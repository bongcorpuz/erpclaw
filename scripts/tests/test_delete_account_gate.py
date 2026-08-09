"""Security tests for the `delete-account` DANGEROUS_ACTIONS gate.

Companion to ``test_dangerous_actions_gate.py`` (which owns ``cleanup-backups``
and is deliberately left untouched — this is a separate file for a separate
action). Same loader/invocation pattern: the router
(``<install>/scripts/db_query.py``) is loaded by file location, and the gate is
driven directly via ``_gate_dangerous_action`` with a patched ``sys.argv``,
because the gate reads its per-invocation flag from argv by design.

Three concerns are covered here:

  (a) ``delete-account`` really is gated by the router — blocked without
      ``--user-confirmed``, allowed with it.

  (b) The router diff surface is MINIMAL. Adding a destructive action is the
      moment when an unrelated read-only action can accidentally get dragged
      into ``DANGEROUS_ACTIONS`` (breaking callers) or an unrelated action can
      get silently remapped in ``ACTION_MAP`` (routing to the wrong domain
      script). Both are asserted against.

  (c) THE IMPORTANT ONE. The confirmation gate lives ONLY in the router.
      Invoking ``scripts/erpclaw-gl/db_query.py --action delete-account``
      directly bypasses ``--user-confirmed`` entirely — that is pre-existing
      behaviour shared by every action in ``DANGEROUS_ACTIONS``, not a
      regression introduced by this feature. It does mean the router gate is a
      confirmation/UX control, NOT the security control. The real enforcement
      has to be ``_account_delete_blockers`` inside the domain script. These
      tests prove the three-condition guard still holds on the ungated path.

Every database this module touches is created under pytest's ``tmp_path``.
The live ``~/.openclaw/erpclaw/data.sqlite`` is never opened: subprocesses get
an explicit ``--db-path``, an ``ERPCLAW_DB_PATH`` override, and a scrubbed
``ERPCLAW_TEST_SESSION`` (the router's ``_log_action_call`` writes to the
ERPCLAW_HOME database, ignoring ``--db-path``, whenever that var is set).
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import uuid
from unittest.mock import patch

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_TESTS_DIR)
_ROUTER_PATH = os.path.join(_SCRIPTS_DIR, "db_query.py")
_GL_DB_QUERY = os.path.join(_SCRIPTS_DIR, "erpclaw-gl", "db_query.py")
_INIT_SCHEMA = os.path.join(_SCRIPTS_DIR, "erpclaw-setup", "init_schema.py")

_LIVE_DB = os.path.join(
    os.path.expanduser(os.environ.get("ERPCLAW_HOME", "~/.openclaw/erpclaw")),
    "data.sqlite",
)


def _load_router():
    spec = importlib.util.spec_from_file_location("erpclaw_router", _ROUTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROUTER = _load_router()


# ──────────────────────────────────────────────────────────────────────────────
# (a) delete-account is gated by the router
# ──────────────────────────────────────────────────────────────────────────────

def test_delete_account_is_dangerous():
    """delete-account joins DANGEROUS_ACTIONS (it hard-DELETEs a GL account)."""
    assert "delete-account" in ROUTER.DANGEROUS_ACTIONS


def test_delete_account_blocked_without_flag(capsys):
    with patch.object(sys, "argv",
                      ["db_query.py", "--action", "delete-account",
                       "--account-id", "some-account"]):
        with pytest.raises(SystemExit) as exc:
            ROUTER._gate_dangerous_action("delete-account")
    assert exc.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "user_confirmation_required"
    assert payload["action"] == "delete-account"


def test_delete_account_passes_with_flag():
    with patch.object(sys, "argv",
                      ["db_query.py", "--action", "delete-account",
                       "--account-id", "some-account", "--user-confirmed"]):
        # No exit, no output — the gate lets dispatch proceed.
        ROUTER._gate_dangerous_action("delete-account")


def test_delete_account_gate_is_not_env_bypassable(monkeypatch, capsys):
    """A process-wide env var must NOT stand in for the per-invocation flag.

    Regression guard on ``_is_user_confirmed``: an env bypass would let a
    long-running agent or a cron job enable irreversible account deletion
    globally instead of per call.
    """
    for var in ("ERPCLAW_USER_CONFIRMED", "USER_CONFIRMED", "ERPCLAW_FORCE"):
        monkeypatch.setenv(var, "1")
    with patch.object(sys, "argv",
                      ["db_query.py", "--action", "delete-account"]):
        with pytest.raises(SystemExit) as exc:
            ROUTER._gate_dangerous_action("delete-account")
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "user_confirmation_required"


def test_delete_account_gate_has_no_force_escape_hatch(capsys):
    """`initialize-database` is the ONLY action whose gate is conditional on
    --force. delete-account must not have inherited that carve-out: passing
    --force must still leave it blocked, not skip the gate."""
    with patch.object(sys, "argv",
                      ["db_query.py", "--action", "delete-account", "--force"]):
        with pytest.raises(SystemExit) as exc:
            ROUTER._gate_dangerous_action("delete-account")
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "user_confirmation_required"


# ──────────────────────────────────────────────────────────────────────────────
# (b) the router diff surface is minimal
# ──────────────────────────────────────────────────────────────────────────────

def test_delete_account_routes_to_gl_domain():
    assert ROUTER.ACTION_MAP.get("delete-account") == "erpclaw-gl"


def test_router_mentions_delete_account_exactly_twice():
    """Exactly two insertion points in the router source: the ACTION_MAP entry
    and the DANGEROUS_ACTIONS entry. A third occurrence means an alias, a
    special case, or a carve-out was added that this suite has not reviewed."""
    with open(_ROUTER_PATH, encoding="utf-8") as fh:
        source = fh.read()

    occurrences = re.findall(r'"delete-account"', source)
    assert len(occurrences) == 2, (
        f"expected exactly 2 quoted 'delete-account' literals in the router, "
        f"found {len(occurrences)}"
    )

    map_idx = source.index("ACTION_MAP = {")
    danger_idx = source.index("DANGEROUS_ACTIONS = frozenset({")
    positions = [m.start() for m in re.finditer(r'"delete-account"', source)]
    # One occurrence inside the ACTION_MAP block, one inside DANGEROUS_ACTIONS.
    assert any(map_idx < p < danger_idx for p in positions), \
        "no delete-account entry found inside ACTION_MAP"
    assert any(p > danger_idx for p in positions), \
        "no delete-account entry found inside DANGEROUS_ACTIONS"


def test_delete_account_is_not_aliased():
    """No alias may quietly redirect delete-account to another domain script,
    which would route around erpclaw-gl's guard entirely."""
    assert "delete-account" not in ROUTER.ALIASES
    for alias, target in ROUTER.ALIASES.items():
        assert "delete-account" not in target, \
            f"alias {alias!r} resolves to delete-account: {target!r}"


def test_no_extra_account_actions_became_dangerous():
    """Only the three account-lifecycle mutations may be gated. If this fails,
    a read-only or unrelated account action was dragged into the frozenset
    alongside delete-account and every existing caller of it now breaks."""
    gated_account_actions = {
        a for a in ROUTER.DANGEROUS_ACTIONS if "account" in a
    }
    assert gated_account_actions == {
        "freeze-account", "unfreeze-account", "delete-account",
    }


@pytest.mark.parametrize("action", [
    "get-account", "list-accounts", "add-account", "update-account",
    "get-account-balance", "list-gl-entries", "check-gl-integrity",
    "setup-chart-of-accounts", "import-chart-of-accounts",
])
def test_read_only_and_nondestructive_account_siblings_stay_ungated(action):
    """Reading the chart of accounts must never demand --user-confirmed."""
    assert action not in ROUTER.DANGEROUS_ACTIONS
    with patch.object(sys, "argv", ["db_query.py", "--action", action]):
        ROUTER._gate_dangerous_action(action)  # no exit


def test_gl_account_actions_still_route_to_gl():
    """delete-account's ACTION_MAP insertion must not have displaced or
    retargeted its neighbours."""
    for action in ("add-account", "update-account", "get-account",
                   "list-accounts", "freeze-account", "unfreeze-account"):
        assert ROUTER.ACTION_MAP.get(action) == "erpclaw-gl", \
            f"{action} no longer routes to erpclaw-gl"


# ──────────────────────────────────────────────────────────────────────────────
# (c) the guard survives a router bypass — the important test
# ──────────────────────────────────────────────────────────────────────────────

def _init_schema(db_path):
    spec = importlib.util.spec_from_file_location("init_schema_for_gate_tests",
                                                  _INIT_SCHEMA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.init_db(db_path)


def _run_gl_direct(db_path, *args):
    """Invoke the erpclaw-gl domain script DIRECTLY, bypassing the router.

    This is the attack path under test: the ``--user-confirmed`` gate is never
    consulted because the router is never in the call chain.
    """
    assert os.path.abspath(db_path) != os.path.abspath(_LIVE_DB), \
        "refusing to run against the live database"
    env = dict(os.environ)
    # The router's _log_action_call() writes into the ERPCLAW_HOME database
    # regardless of --db-path when this is set. Never let it be set here.
    env.pop("ERPCLAW_TEST_SESSION", None)
    env["ERPCLAW_DB_PATH"] = db_path
    return subprocess.run(
        [sys.executable, _GL_DB_QUERY, "--db-path", db_path, *args],
        capture_output=True, text=True, env=env,
    )


def _u():
    return str(uuid.uuid4())


@pytest.fixture
def bypass_db(tmp_path):
    """Ephemeral DB seeded with one account per blocker plus a clean control.

    Rows are inserted with plain parameterized SQL rather than through the
    domain script so the fixture cannot be invalidated by the very code under
    test.
    """
    import sqlite3

    db_path = str(tmp_path / "bypass.sqlite")
    _init_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    # Precondition, not a test: if schema creation silently didn't land, the
    # failure below reads like a guard regression and is not one.
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
        "AND name IN ('company','account','gl_entry','journal_entry',"
        "'journal_entry_line','audit_log')").fetchone()[0] == 6, (
        "test database has no schema — init_schema did not complete. "
        "Test-infrastructure/environment, NOT a delete-account guard failure.")

    company_id = _u()
    conn.execute(
        """INSERT INTO company (id, name, abbr, default_currency, country,
           fiscal_year_start_month) VALUES (?, ?, ?, 'USD', 'United States', 1)""",
        (company_id, "Bypass Co", "BYP"),
    )

    def add_account(name, number, parent_id=None, is_group=0):
        aid = _u()
        conn.execute(
            """INSERT INTO account (id, name, account_number, root_type,
               account_type, balance_direction, company_id, depth, is_group,
               parent_id)
               VALUES (?, ?, ?, 'asset', 'cash', 'debit_normal', ?, 0, ?, ?)""",
            (aid, name, number, company_id, is_group, parent_id),
        )
        return aid

    ids = {
        "clean": add_account("Clean Control", "9000"),
        "balance": add_account("Has Balance", "9100"),
        "history": add_account("Has GL History", "9200"),
        "parent": add_account("Has Children", "9300", is_group=1),
        "posted_je": add_account("Posted JE Ref", "9400"),
    }
    ids["child"] = add_account("Child", "9301", parent_id=ids["parent"])

    # Non-zero balance: a single uncancelled debit.
    conn.execute(
        """INSERT INTO gl_entry (id, posting_date, account_id, debit, credit,
           voucher_type, voucher_id, is_cancelled)
           VALUES (?, '2026-06-15', ?, '250.00', '0', 'journal_entry',
                   'JE-BAL', 0)""",
        (_u(), ids["balance"]),
    )
    # GL history with a NET-ZERO balance: proves the guard is not balance-only.
    for debit, credit in (("100.00", "0"), ("0", "100.00")):
        conn.execute(
            """INSERT INTO gl_entry (id, posting_date, account_id, debit, credit,
               voucher_type, voucher_id, is_cancelled)
               VALUES (?, '2026-06-15', ?, ?, ?, 'journal_entry', 'JE-WASH',
                       0)""",
            (_u(), ids["history"], debit, credit),
        )
    # Submitted journal entry line referencing an account.
    je_id = _u()
    conn.execute(
        """INSERT INTO journal_entry (id, naming_series, posting_date,
           company_id, status)
           VALUES (?, 'JV-0001', '2026-06-15', ?, 'submitted')""",
        (je_id, company_id),
    )
    conn.execute(
        """INSERT INTO journal_entry_line (id, journal_entry_id, account_id,
           debit, credit) VALUES (?, ?, ?, '75.00', '0')""",
        (_u(), je_id, ids["posted_je"]),
    )

    conn.commit()
    conn.close()
    return {"db_path": db_path, "company_id": company_id, "ids": ids}


def _account_exists(db_path, account_id):
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT 1 FROM account WHERE id = ?", (account_id,)
        ).fetchone() is not None
    finally:
        conn.close()


def test_router_gate_really_is_bypassable_by_direct_invocation(bypass_db):
    """Document the pre-existing architecture honestly rather than pretend.

    A clean account deletes with exit 0 through the domain script with NO
    --user-confirmed anywhere. That is the shared behaviour of all gated
    actions. It is why the tests below matter: the router gate cannot be the
    thing enforcing the three-condition guard.
    """
    proc = _run_gl_direct(bypass_db["db_path"], "--action", "delete-account",
                          "--account-id", bypass_db["ids"]["clean"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["status"] == "ok"
    assert not _account_exists(bypass_db["db_path"], bypass_db["ids"]["clean"])


@pytest.mark.parametrize("key,expected_fragment", [
    ("balance", "non-zero balance"),
    ("history", "GL entry record"),
    ("parent", "child account"),
    ("posted_je", "posted journal entry line"),
])
def test_guard_holds_when_router_gate_is_bypassed(bypass_db, key,
                                                  expected_fragment):
    """THE load-bearing test.

    Straight into erpclaw-gl/db_query.py — no router, no --user-confirmed, no
    confirmation of any kind — and the delete is still refused for every
    blocker class, with the account still present afterwards. If any of these
    ever start passing with exit 0, the three-condition guard has become a
    router-level UX prompt and nothing more.
    """
    account_id = bypass_db["ids"][key]
    proc = _run_gl_direct(bypass_db["db_path"], "--action", "delete-account",
                          "--account-id", account_id)

    assert proc.returncode == 1, (
        f"blocker {key!r} did NOT block a direct invocation: "
        f"exit={proc.returncode} stdout={proc.stdout}"
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "error"
    assert "Cannot delete account" in payload["message"]
    assert expected_fragment in payload["message"]
    assert _account_exists(bypass_db["db_path"], account_id), \
        f"account was deleted despite blocker {key!r}"


def test_bypass_path_reports_all_blockers_not_just_the_first(bypass_db):
    """An account with a balance AND GL history reports both, so a caller
    cannot clear one condition, retry, and be surprised by the next."""
    proc = _run_gl_direct(bypass_db["db_path"], "--action", "delete-account",
                          "--account-id", bypass_db["ids"]["balance"])
    assert proc.returncode == 1
    message = json.loads(proc.stdout)["message"]
    assert "non-zero balance" in message
    assert "GL entry record" in message


def test_bypass_path_rejects_user_confirmed_as_an_unknown_flag(bypass_db):
    """The domain script does not know --user-confirmed; passing it directly
    must be rejected as an unknown flag rather than silently accepted as some
    kind of local override."""
    proc = _run_gl_direct(bypass_db["db_path"], "--action", "delete-account",
                          "--account-id", bypass_db["ids"]["parent"],
                          "--user-confirmed")
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert "Unknown flags" in payload["message"]
    assert "--user-confirmed" in payload["message"]
    assert _account_exists(bypass_db["db_path"], bypass_db["ids"]["parent"])


def test_blocked_bypass_attempt_writes_no_audit_row(bypass_db):
    """A refused deletion must leave no audit_log trace claiming a delete."""
    import sqlite3

    db_path = bypass_db["db_path"]
    conn = sqlite3.connect(db_path)
    before = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    conn.close()

    proc = _run_gl_direct(db_path, "--action", "delete-account",
                          "--account-id", bypass_db["ids"]["history"])
    assert proc.returncode == 1

    conn = sqlite3.connect(db_path)
    after = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    conn.close()
    assert after == before, "a blocked delete wrote an audit_log row"
