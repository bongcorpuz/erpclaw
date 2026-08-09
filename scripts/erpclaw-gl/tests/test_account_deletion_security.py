"""Security tests for the erpclaw-gl `delete-account` action.

Scope is deliberately narrow and adversarial — the functional/behavioural
coverage of the three-condition guard lives in ``test_account_deletion.py``.
This file only asks: can the guard be talked out of the way?

  (d) AC-8 — there is no force/override path. ``--force`` / ``--override``
      must be REJECTED by ``check_unknown_args``, naming the offending flag,
      not silently dropped by ``parse_known_args`` while the delete proceeds.
  (e) ``--account-id`` is parameterized. Classic injection payloads must be
      treated as opaque identifiers that simply do not match a row.
  (f) Cross-company deletion is refused when a company is supplied.
  (g) Audit discipline — a blocked attempt writes zero ``audit_log`` rows; a
      successful delete writes exactly one.
  (h) Deleting an account has no side effect on any OTHER account's
      ``gl_entry`` checksum chain.

Every test runs against the per-test ``tmp_path`` SQLite database provided by
``conftest.py``. The live ``~/.openclaw/erpclaw/data.sqlite`` is never opened;
subprocess tests pass an explicit ``--db-path`` and scrub
``ERPCLAW_TEST_SESSION`` from the environment (with it set, the FOUNDATION
ROUTER logs into the ERPCLAW_HOME database regardless of ``--db-path`` — that
is not this script's behaviour, but the scrub keeps the boundary unambiguous).
"""
import json
import os
import subprocess
import sys
import uuid
from decimal import Decimal

import pytest

from gl_helpers import (
    call_action, ns, is_error, is_ok, load_db_query,
    seed_company, seed_account, seed_fiscal_year, seed_cost_center,
    SCRIPTS_DIR,
)

mod = load_db_query()

_GL_DB_QUERY = os.path.join(SCRIPTS_DIR, "db_query.py")
_LIVE_DB = os.path.join(
    os.path.expanduser(os.environ.get("ERPCLAW_HOME", "~/.openclaw/erpclaw")),
    "data.sqlite",
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def delete_args(account_id, company_id=None, company_name=None):
    """Exactly the argparse surface ``delete_account`` reads — no more."""
    return ns(account_id=account_id, company_id=company_id,
              company_name=company_name)


def run_cli(db_path, *args):
    """Invoke the domain script as a subprocess against an ephemeral DB."""
    assert os.path.abspath(db_path) != os.path.abspath(_LIVE_DB), \
        "refusing to run against the live database"
    env = dict(os.environ)
    env.pop("ERPCLAW_TEST_SESSION", None)
    env["ERPCLAW_DB_PATH"] = db_path
    return subprocess.run(
        [sys.executable, _GL_DB_QUERY, "--db-path", db_path, *args],
        capture_output=True, text=True, env=env,
    )


def account_exists(conn, account_id):
    return conn.execute(
        "SELECT 1 FROM account WHERE id = ?", (account_id,)
    ).fetchone() is not None


def account_count(conn):
    return conn.execute("SELECT COUNT(*) FROM account").fetchone()[0]


def audit_count(conn):
    return conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]


def table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (name,)
    ).fetchone() is not None


@pytest.fixture
def books(conn):
    """A company with a posted, checksum-chained GL plus deletable controls."""
    # Precondition, not a test: the shared conftest `db_path` fixture builds
    # the schema. If it silently didn't, the failure otherwise surfaces as a
    # bare "no such table: company" from gl_helpers, which reads like a
    # security regression and is not one. Observed once on Windows as a
    # transient TEMP filesystem event; kept as a diagnostic, never a retry.
    assert table_exists(conn, "company") and table_exists(conn, "gl_entry"), (
        "test database has no schema — the shared db_path/init_all_tables "
        "fixture did not complete. This is test-infrastructure/environment, "
        "NOT a delete-account guard failure."
    )
    company_id = seed_company(conn, "SecCo", "SC")
    seed_fiscal_year(conn, company_id)
    cost_center = seed_cost_center(conn, company_id, "Main CC")

    cash = seed_account(conn, company_id, "Cash", "asset", "cash", "1000")
    revenue = seed_account(conn, company_id, "Revenue", "income", "revenue", "4000")
    clean = seed_account(conn, company_id, "Clean", "asset", "cash", "1900")
    spare = seed_account(conn, company_id, "Spare", "asset", "cash", "1901")

    entries = json.dumps([
        {"account_id": cash, "debit": "1000.00", "credit": "0"},
        {"account_id": revenue, "debit": "0", "credit": "1000.00",
         "cost_center_id": cost_center},
    ])
    result = call_action(mod.post_gl_entries, conn, ns(
        voucher_type="journal_entry", voucher_id="JE-SEC-1",
        posting_date="2026-06-15", company_id=company_id, entries=entries,
    ))
    assert is_ok(result), result

    company_name = conn.execute(
        "SELECT name FROM company WHERE id = ?", (company_id,)
    ).fetchone()["name"]

    return {
        "company_id": company_id, "company_name": company_name,
        "cash": cash, "revenue": revenue, "clean": clean, "spare": spare,
        "cost_center": cost_center,
    }


# ──────────────────────────────────────────────────────────────────────────────
# (d) AC-8 — no force / override escape hatch
# ──────────────────────────────────────────────────────────────────────────────

class TestNoForceOverridePath:
    """``parse_known_args`` would otherwise swallow an unknown flag silently.
    ``check_unknown_args`` is what turns that into a hard stop, and it is the
    only thing standing between "--force was ignored" and "--force worked"."""

    @pytest.mark.parametrize("flag", [
        "--force", "--override", "--no-check", "--skip-checks", "--yes",
        "--user-confirmed",
    ])
    def test_unknown_override_flag_is_rejected_by_name(self, conn, db_path,
                                                       books, flag):
        target = books["clean"]  # otherwise perfectly deletable
        proc = run_cli(db_path, "--action", "delete-account",
                       "--account-id", target, flag)

        assert proc.returncode == 1, proc.stdout + proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["status"] == "error"
        # Explicitly NAMED, not a vague "invalid arguments".
        assert "Unknown flags" in payload["message"]
        assert flag in payload["message"], \
            f"error did not name the rejected flag: {payload['message']}"
        # And the parse failure happened BEFORE any deletion.
        assert account_exists(conn, target)

    def test_force_does_not_override_a_real_blocker(self, conn, db_path, books):
        """The scenario an attacker actually wants: a blocked account plus
        --force. Must fail on the unknown flag, never on a bypassed guard."""
        proc = run_cli(db_path, "--action", "delete-account",
                       "--account-id", books["cash"], "--force")
        assert proc.returncode == 1
        assert "Unknown flags" in json.loads(proc.stdout)["message"]
        assert account_exists(conn, books["cash"])

    def test_force_flag_value_form_also_rejected(self, conn, db_path, books):
        """``--force=true`` and ``--force true`` are separate argparse shapes;
        neither may slip through."""
        for extra in (["--force=true"], ["--override", "true"]):
            proc = run_cli(db_path, "--action", "delete-account",
                           "--account-id", books["clean"], *extra)
            assert proc.returncode == 1, proc.stdout
            assert "Unknown flags" in json.loads(proc.stdout)["message"]
        assert account_exists(conn, books["clean"])

    def test_delete_account_declares_no_force_argument(self):
        """Defence in depth: the flag must not exist in the parser at all, or
        ``check_unknown_args`` would never see it."""
        import inspect
        source = inspect.getsource(mod.main)
        for banned in ('"--force"', "'--force'", '"--override"', "'--override'"):
            assert banned not in source, \
                f"erpclaw-gl argparse now declares {banned}"


# ──────────────────────────────────────────────────────────────────────────────
# (e) SQL injection probes on --account-id
# ──────────────────────────────────────────────────────────────────────────────

INJECTION_PAYLOADS = [
    "' OR '1'='1",
    "x'; DROP TABLE account; --",
    "' OR 1=1 --",
    "'; DELETE FROM account WHERE '1'='1",
    '" OR ""="',
    "1' UNION SELECT id FROM account --",
    "%' --",
    "\\'; DROP TABLE gl_entry; --",
]


class TestAccountIdInjection:
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    def test_injection_payload_is_treated_as_an_opaque_id(self, conn, books,
                                                          payload):
        """A parameterized query cannot interpret the payload, so the only
        correct outcome is a clean 'not found' — never a partial match, never
        a mass delete, never a leaked SQL fragment."""
        before = account_count(conn)

        result = call_action(mod.delete_account, conn, delete_args(payload))

        assert is_error(result)
        assert "not found" in result["message"]
        # No SQL internals leaked back to the caller. The payload itself is
        # echoed into the "Account <id> not found" message, so strip it first
        # — reflecting the caller's own input in a JSON CLI response is not a
        # leak, but any SQL token AROUND it would be.
        residue = result["message"].replace(payload, "<PAYLOAD>").lower()
        for token in ("select ", "delete from", "drop table", "syntax error",
                      "unrecognized token", "sqlite3.", "traceback",
                      "near \"", "account_id ="):
            assert token not in residue, \
                f"error message leaked {token!r}: {result['message']}"
        # Nothing was destroyed.
        assert account_count(conn) == before
        assert table_exists(conn, "account")
        assert table_exists(conn, "gl_entry")

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    def test_injection_payload_leaves_every_named_account_intact(self, conn,
                                                                 books, payload):
        call_action(mod.delete_account, conn, delete_args(payload))
        for key in ("cash", "revenue", "clean", "spare"):
            assert account_exists(conn, books[key]), \
                f"payload {payload!r} removed account {key!r}"

    def test_injection_payload_writes_no_audit_row(self, conn, books):
        before = audit_count(conn)
        for payload in INJECTION_PAYLOADS:
            call_action(mod.delete_account, conn, delete_args(payload))
        assert audit_count(conn) == before

    def test_injection_in_company_id_does_not_widen_the_match(self, conn, books):
        """``--company-id`` is the other user-controlled value on this path and
        is compared against the account's own company_id in Python, but it is
        also fed to ``resolve_company_id``. It must not become a wildcard."""
        before = account_count(conn)
        result = call_action(mod.delete_account, conn,
                             delete_args(books["clean"], company_id="' OR '1'='1"))
        assert is_error(result)
        assert "does not belong to company" in result["message"]
        assert account_exists(conn, books["clean"])
        assert account_count(conn) == before

    def test_injection_in_company_name_does_not_resolve(self, conn, books):
        """``--company`` goes through ``resolve_company_id``, which does a
        parameterized ``LOWER(name) = ?`` lookup. An injection payload must
        miss and abort before the account is ever touched.

        NOTE for review: this miss path emits ``{"error": ...}`` WITHOUT the
        ``"status": "error"`` key every other response in the codebase uses,
        so ``is_error()`` does not recognise it. That envelope inconsistency
        lives in the shared ``erpclaw_lib.query_helpers`` and predates this
        feature — reported as a LOW finding. Asserted loosely here so the
        test tracks the security property, not the defect.
        """
        before = account_count(conn)
        audit_before = audit_count(conn)
        result = call_action(mod.delete_account, conn,
                             delete_args(books["clean"],
                                         company_name="' OR '1'='1' --"))
        assert is_error(result) or "error" in result, result
        assert "not found" in json.dumps(result).lower()
        # Critically: no deletion, and no wildcard company match.
        assert account_exists(conn, books["clean"])
        assert account_count(conn) == before
        assert audit_count(conn) == audit_before

    def test_oversized_account_id_is_inert(self, conn, books):
        """A 20k-char identifier at the domain-function layer (i.e. with the
        CLI's length check out of the picture) must still be a plain miss."""
        before = account_count(conn)
        result = call_action(mod.delete_account, conn, delete_args("A" * 20000))
        assert is_error(result)
        assert account_count(conn) == before
        assert account_exists(conn, books["clean"])

    def test_oversized_account_id_rejected_at_the_cli_boundary(self, conn,
                                                               db_path, books):
        """``check_input_lengths`` caps --account-id at 5000 chars.

        Asserted as a security property only (rejected, nothing deleted).
        NOTE for review: this rejection currently surfaces as an uncaught
        ValueError traceback on stderr rather than the JSON error contract,
        because ``check_input_lengths`` is called outside main()'s try block.
        That is pre-existing and applies to every action and every string flag
        — reported as a LOW finding, not asserted here, so a future fix that
        converts it to JSON will not break this test."""
        proc = run_cli(db_path, "--action", "delete-account",
                       "--account-id", "A" * 6000)
        assert proc.returncode != 0
        assert '"status": "ok"' not in proc.stdout
        assert account_exists(conn, books["clean"])

    def test_empty_and_whitespace_account_id_rejected(self, conn, books):
        before = account_count(conn)
        for value in ("", None):
            result = call_action(mod.delete_account, conn, delete_args(value))
            assert is_error(result)
            assert "--account-id is required" in result["message"]
        assert account_count(conn) == before


# ──────────────────────────────────────────────────────────────────────────────
# (f) cross-company deletion
# ──────────────────────────────────────────────────────────────────────────────

class TestCrossCompanyIsolation:
    @pytest.fixture
    def two_companies(self, conn):
        a = seed_company(conn, "Alpha", "AA")
        b = seed_company(conn, "Bravo", "BB")
        return {
            "a": a,
            "b": b,
            "a_name": conn.execute("SELECT name FROM company WHERE id = ?",
                                   (a,)).fetchone()["name"],
            "b_name": conn.execute("SELECT name FROM company WHERE id = ?",
                                   (b,)).fetchone()["name"],
            "a_account": seed_account(conn, a, "Alpha Cash", "asset", "cash",
                                      "1000"),
            "b_account": seed_account(conn, b, "Bravo Cash", "asset", "cash",
                                      "1000"),
        }

    def test_company_b_cannot_delete_company_a_account_by_id(self, conn,
                                                             two_companies):
        result = call_action(mod.delete_account, conn,
                             delete_args(two_companies["a_account"],
                                         company_id=two_companies["b"]))
        assert is_error(result)
        assert "does not belong to company" in result["message"]
        assert account_exists(conn, two_companies["a_account"])

    def test_company_b_cannot_delete_company_a_account_by_name(self, conn,
                                                               two_companies):
        result = call_action(mod.delete_account, conn,
                             delete_args(two_companies["a_account"],
                                         company_name=two_companies["b_name"]))
        assert is_error(result)
        assert "does not belong to company" in result["message"]
        assert account_exists(conn, two_companies["a_account"])

    def test_cross_company_refusal_writes_no_audit_row(self, conn,
                                                       two_companies):
        before = audit_count(conn)
        call_action(mod.delete_account, conn,
                    delete_args(two_companies["a_account"],
                                company_id=two_companies["b"]))
        assert audit_count(conn) == before

    def test_cross_company_refusal_precedes_the_blocker_scan(self, conn,
                                                             two_companies):
        """Ownership is checked before blockers are enumerated, so a caller
        scoped to the wrong company learns nothing about the target account's
        balance, history, or children."""
        message = call_action(
            mod.delete_account, conn,
            delete_args(two_companies["a_account"],
                        company_id=two_companies["b"]),
        )["message"]
        assert "Cannot delete account" not in message
        for leak in ("balance", "GL entry", "child account",
                     "journal entry line"):
            assert leak not in message, f"cross-company error leaked {leak!r}"

    def test_owning_company_can_still_delete_its_own_account(self, conn,
                                                             two_companies):
        """The isolation check must not be so broad it blocks the legitimate
        owner — otherwise the feature is 'secure' by being broken."""
        result = call_action(mod.delete_account, conn,
                             delete_args(two_companies["a_account"],
                                         company_id=two_companies["a"]))
        assert is_ok(result), result
        assert not account_exists(conn, two_companies["a_account"])
        # Bravo's identically-numbered account is untouched.
        assert account_exists(conn, two_companies["b_account"])

    def test_deleting_one_company_account_leaves_the_other_company_intact(
            self, conn, two_companies):
        call_action(mod.delete_account, conn,
                    delete_args(two_companies["a_account"],
                                company_id=two_companies["a"]))
        remaining = conn.execute(
            "SELECT COUNT(*) FROM account WHERE company_id = ?",
            (two_companies["b"],)
        ).fetchone()[0]
        assert remaining == 1


# ──────────────────────────────────────────────────────────────────────────────
# (g) audit-row emission discipline
# ──────────────────────────────────────────────────────────────────────────────

class TestAuditDiscipline:
    """What writes where, so the two logs are not confused:

      - ``audit_log``       — written by ``audit()`` INSIDE erpclaw-gl, only
                              after a delete actually succeeds, committed in
                              the same transaction as the DELETE.
      - ``action_call_log`` — written by the FOUNDATION ROUTER's
                              ``_log_action_call``, and only when the
                              ``ERPCLAW_TEST_SESSION`` env var is set (L2 test
                              instrumentation). It is not part of this action
                              and never runs on the direct-invocation path.

    So a blocked delete must leave BOTH untouched here.
    """

    @pytest.mark.parametrize("blocker", ["balance", "history", "children",
                                         "posted_journal"])
    def test_blocked_delete_writes_zero_audit_rows(self, conn, books, blocker):
        if blocker == "balance":
            target = books["cash"]
        elif blocker == "history":
            # net-zero balance but real GL history
            target = books["spare"]
            entries = json.dumps([
                {"account_id": target, "debit": "50.00", "credit": "0"},
                {"account_id": books["revenue"], "debit": "0",
                 "credit": "50.00", "cost_center_id": books["cost_center"]},
            ])
            call_action(mod.post_gl_entries, conn, ns(
                voucher_type="journal_entry", voucher_id="JE-SEC-H",
                posting_date="2026-06-16", company_id=books["company_id"],
                entries=entries))
            entries = json.dumps([
                {"account_id": books["revenue"], "debit": "50.00",
                 "credit": "0", "cost_center_id": books["cost_center"]},
                {"account_id": target, "debit": "0", "credit": "50.00"},
            ])
            call_action(mod.post_gl_entries, conn, ns(
                voucher_type="journal_entry", voucher_id="JE-SEC-H2",
                posting_date="2026-06-16", company_id=books["company_id"],
                entries=entries))
        elif blocker == "children":
            target = books["clean"]
            child = seed_account(conn, books["company_id"], "Kid", "asset",
                                 "cash", "1902")
            conn.execute("UPDATE account SET parent_id = ? WHERE id = ?",
                         (target, child))
            conn.commit()
        else:
            target = books["clean"]
            je_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO journal_entry (id, naming_series, posting_date,
                   company_id, status)
                   VALUES (?, 'JV-SEC', '2026-06-15', ?, 'submitted')""",
                (je_id, books["company_id"]))
            conn.execute(
                """INSERT INTO journal_entry_line (id, journal_entry_id,
                   account_id, debit, credit) VALUES (?, ?, ?, '10.00', '0')""",
                (str(uuid.uuid4()), je_id, target))
            conn.commit()

        before = audit_count(conn)
        result = call_action(mod.delete_account, conn, delete_args(target))

        assert is_error(result), f"{blocker} did not block the delete"
        assert audit_count(conn) == before, \
            f"blocked delete ({blocker}) wrote an audit_log row"
        assert account_exists(conn, target)

    def test_not_found_writes_zero_audit_rows(self, conn, books):
        before = audit_count(conn)
        call_action(mod.delete_account, conn,
                    delete_args(str(uuid.uuid4())))
        assert audit_count(conn) == before

    def test_successful_delete_writes_exactly_one_audit_row(self, conn, books):
        before = audit_count(conn)
        result = call_action(mod.delete_account, conn,
                             delete_args(books["clean"]))
        assert is_ok(result), result
        assert audit_count(conn) == before + 1

        row = conn.execute(
            """SELECT skill, action, entity_type, entity_id, old_values,
                      new_values
               FROM audit_log ORDER BY rowid DESC LIMIT 1"""
        ).fetchone()
        assert row["skill"] == "erpclaw-gl"
        assert row["action"] == "delete"
        assert row["entity_type"] == "account"
        assert row["entity_id"] == books["clean"]
        # The pre-image is retained so the deletion is reconstructable.
        assert row["old_values"], "delete audit row has no pre-image"
        old = json.loads(row["old_values"])
        assert old["account_number"] == "1900"
        assert old["company_id"] == books["company_id"]

    def test_audit_row_carries_no_credentials_or_connection_strings(
            self, conn, books):
        call_action(mod.delete_account, conn, delete_args(books["clean"]))
        row = conn.execute(
            "SELECT * FROM audit_log ORDER BY rowid DESC LIMIT 1").fetchone()
        blob = json.dumps({k: row[k] for k in row.keys()}, default=str).lower()
        for secret_marker in ("password", "secret", "token", "api_key",
                              "apikey", "sqlite:///", "postgresql://",
                              "master_key", "credential"):
            assert secret_marker not in blob, \
                f"audit row contains {secret_marker!r}"
        # The absolute DB path must not be recorded either.
        assert ".sqlite" not in blob

    def test_router_action_call_log_is_untouched(self, conn, books):
        """The domain script never writes action_call_log — that is router
        test-instrumentation. Confirms the two logs stay distinct."""
        call_action(mod.delete_account, conn, delete_args(books["clean"]))
        if table_exists(conn, "action_call_log"):
            assert conn.execute(
                "SELECT COUNT(*) FROM action_call_log").fetchone()[0] == 0

    def test_blocked_then_fixed_then_deleted_logs_once(self, conn, books):
        """A caller who retries after clearing the blocker must end up with
        one audit row, not one per attempt."""
        child = seed_account(conn, books["company_id"], "Kid", "asset", "cash",
                             "1903")
        conn.execute("UPDATE account SET parent_id = ? WHERE id = ?",
                     (books["clean"], child))
        conn.commit()

        before = audit_count(conn)
        assert is_error(call_action(mod.delete_account, conn,
                                    delete_args(books["clean"])))
        assert is_error(call_action(mod.delete_account, conn,
                                    delete_args(books["clean"])))
        assert audit_count(conn) == before

        assert is_ok(call_action(mod.delete_account, conn, delete_args(child)))
        assert is_ok(call_action(mod.delete_account, conn,
                                 delete_args(books["clean"])))
        assert audit_count(conn) == before + 2


# ──────────────────────────────────────────────────────────────────────────────
# (h) gl_entry checksum-chain integrity
# ──────────────────────────────────────────────────────────────────────────────

class TestChecksumChainIntegrity:
    """``check_gl_integrity`` rebuilds the SHA-256 chain by joining gl_entry to
    account and filtering on company. That join is why account deletion and GL
    integrity are coupled: removing an account that owns gl_entry rows would
    silently drop those rows out of the reconstruction and break the chain.

    The ``posted_gl_history`` blocker is what prevents that. These tests pin
    both halves — the blocker holds, and a legitimate delete is inert.
    """

    def _checksums(self, conn):
        return [r["gl_checksum"] for r in conn.execute(
            "SELECT gl_checksum FROM gl_entry ORDER BY created_at, rowid")]

    def _integrity(self, conn, company_id):
        return call_action(mod.check_gl_integrity, conn,
                           ns(company_id=company_id, company_name=None))

    def test_chain_is_intact_before_any_deletion(self, conn, books):
        report = self._integrity(conn, books["company_id"])
        assert is_ok(report), report
        assert report["chain_intact"] is True
        assert report["broken_links"] == 0
        assert report["total_entries"] == 2

    def test_successful_delete_leaves_other_accounts_checksums_byte_identical(
            self, conn, books):
        before = self._checksums(conn)
        assert before and all(before), "fixture produced no chained entries"

        assert is_ok(call_action(mod.delete_account, conn,
                                 delete_args(books["clean"])))

        assert self._checksums(conn) == before

    def test_successful_delete_leaves_the_chain_verifiable(self, conn, books):
        assert is_ok(call_action(mod.delete_account, conn,
                                 delete_args(books["clean"])))
        report = self._integrity(conn, books["company_id"])
        assert is_ok(report), report
        assert report["chain_intact"] is True
        assert report["broken_links"] == 0
        assert report["total_entries"] == 2
        assert report["balanced"] is True

    def test_successful_delete_removes_no_gl_entry_rows(self, conn, books):
        before = conn.execute("SELECT COUNT(*) FROM gl_entry").fetchone()[0]
        assert is_ok(call_action(mod.delete_account, conn,
                                 delete_args(books["clean"])))
        assert conn.execute(
            "SELECT COUNT(*) FROM gl_entry").fetchone()[0] == before

    def test_successful_delete_does_not_move_other_account_balances(
            self, conn, books):
        """Uses the same balance helper the guard itself relies on, so a
        drift between them would show up here."""
        before = {
            key: Decimal(mod._lib_get_account_balance(conn, books[key])["balance"])
            for key in ("cash", "revenue")
        }
        assert before["cash"] == Decimal("1000.00")

        assert is_ok(call_action(mod.delete_account, conn,
                                 delete_args(books["clean"])))

        for key, value in before.items():
            after = Decimal(
                mod._lib_get_account_balance(conn, books[key])["balance"])
            assert after == value, f"{key} balance moved: {value} -> {after}"

    def test_account_owning_gl_entries_cannot_be_deleted_at_all(self, conn,
                                                                books):
        """The guarantee the chain rests on. If this ever regresses, every
        other test in this class becomes meaningless."""
        codes = {b["code"] for b in
                 mod._account_delete_blockers(conn, books["cash"])}
        assert "posted_gl_history" in codes

        result = call_action(mod.delete_account, conn,
                             delete_args(books["cash"]))
        assert is_error(result)
        assert account_exists(conn, books["cash"])
        report = self._integrity(conn, books["company_id"])
        assert report["chain_intact"] is True

    def test_foreign_key_restrict_is_a_second_line_of_defence(self, conn,
                                                              books):
        """Even with the application guard hypothetically removed, the schema
        must refuse the delete. Exercised directly against SQLite so it fails
        loudly if ON DELETE RESTRICT is ever relaxed to CASCADE."""
        import sqlite3
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM account WHERE id = ?", (books["cash"],))
        conn.rollback()
        assert account_exists(conn, books["cash"])
        assert self._checksums(conn)

    def test_cancelled_gl_entries_still_protect_the_chain(self, conn, books):
        """A cancelled entry keeps its row and its link in the chain, so the
        account must stay undeletable even though its balance nets to zero."""
        conn.execute("UPDATE gl_entry SET is_cancelled = 1 WHERE account_id = ?",
                     (books["cash"],))
        conn.commit()

        blockers = {b["code"] for b in
                    mod._account_delete_blockers(conn, books["cash"])}
        assert "posted_gl_history" in blockers
        assert "non_zero_balance" not in blockers

        assert is_error(call_action(mod.delete_account, conn,
                                    delete_args(books["cash"])))
        assert account_exists(conn, books["cash"])
