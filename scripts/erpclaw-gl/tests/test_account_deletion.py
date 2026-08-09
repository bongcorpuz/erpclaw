"""Functional tests for the erpclaw-gl ``delete-account`` guard.

Covers Coder 1's ``_account_delete_blockers`` / ``delete_account`` in
``scripts/erpclaw-gl/db_query.py``. Acceptance criteria under test (mapped
1:1 to the swarm-coder blueprint):

  AC-1  non-zero balance blocks deletion; Decimal comparison regression
        (balance string "0.00" must NOT be treated as non-zero).
  AC-2  posted GL history (even is_cancelled=1 rows) blocks deletion at a
        net-zero balance.
  AC-2b journal_entry_line under a submitted/cancelled/amended parent
        triggers posted_journal_history.
  AC-3  journal_entry_line under a draft parent triggers the DISTINCT
        draft_journal_reference code, not posted_journal_history, and does
        not crash with a raw FK error.
  AC-4  child accounts (via parent_id) block deletion regardless of the
        child's disabled/is_frozen flags or bogus lft/rgt values.
  AC-5  an empty is_group=1 account with no children/balance/history IS
        deletable (is_group alone must not block).
  AC-6  an account failing multiple checks at once reports ALL reasons in
        one composed message.
  AC-7  a RESTRICT-FK reference from a table outside the 5 named checks
        (budget.account_id) is blocked via the generic IntegrityError
        safety net — no raw exception text or table/column name leaks.
  AC-9  happy-path deletion: exactly one audit_log row (skill=erpclaw-gl,
        action=delete, entity_type=account) with old_values populated, and
        the deletion is actually committed (visible on a fresh connection).
  AC-10 any BLOCKED deletion attempt leaves the account, its children, its
        gl_entry rows, and its journal_entry_line rows completely
        unchanged, and writes NO audit row.

Uses the existing ``conn``/``db_path`` fixtures from conftest.py and the
seed_company/seed_account/seed_fiscal_year helpers from gl_helpers.py. Raw
SQL seed helpers for gl_entry / journal_entry / journal_entry_line / budget
are added locally below since gl_helpers.py does not yet provide them for
this module (matching the module's existing use of direct SQL for seeding
in test_gl_entries.py's fixtures).
"""
import json
import uuid
from decimal import Decimal

import pytest
from gl_helpers import (
    call_action, ns, is_error, is_ok, load_db_query,
    seed_company, seed_account, seed_fiscal_year, get_conn,
)

mod = load_db_query()


# ──────────────────────────────────────────────────────────────────────────────
# Local seed helpers (gl_entry / journal_entry / journal_entry_line / budget
# have no shared seed_* helpers yet in gl_helpers.py for this test module)
# ──────────────────────────────────────────────────────────────────────────────

def _uid() -> str:
    return str(uuid.uuid4())


def seed_gl_entry_raw(conn, account_id, *, debit="0", credit="0", is_cancelled=0,
                       voucher_type="journal_entry", voucher_id=None,
                       posting_date="2026-01-15"):
    """Insert a gl_entry row directly via SQL, bypassing post-gl-entries'
    balanced-double-entry validation, so tests can build ledger states
    (e.g. cancelled-only history) the normal action would refuse to create."""
    gid = _uid()
    voucher_id = voucher_id or f"V-{gid[:8]}"
    conn.execute(
        """INSERT INTO gl_entry (id, posting_date, account_id, debit, credit,
           debit_base, credit_base, voucher_type, voucher_id, is_cancelled)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (gid, posting_date, account_id, debit, credit, debit, credit,
         voucher_type, voucher_id, is_cancelled),
    )
    conn.commit()
    return gid


def seed_journal_entry_raw(conn, company_id, status="draft", posting_date="2026-01-15"):
    jid = _uid()
    conn.execute(
        """INSERT INTO journal_entry (id, posting_date, status, company_id)
           VALUES (?, ?, ?, ?)""",
        (jid, posting_date, status, company_id),
    )
    conn.commit()
    return jid


def seed_journal_entry_line_raw(conn, journal_entry_id, account_id, *,
                                 debit="0", credit="0"):
    lid = _uid()
    conn.execute(
        """INSERT INTO journal_entry_line (id, journal_entry_id, account_id, debit, credit)
           VALUES (?, ?, ?, ?, ?)""",
        (lid, journal_entry_id, account_id, debit, credit),
    )
    conn.commit()
    return lid


def seed_budget_raw(conn, company_id, fiscal_year_id, account_id):
    bid = _uid()
    conn.execute(
        """INSERT INTO budget (id, fiscal_year_id, account_id, company_id)
           VALUES (?, ?, ?, ?)""",
        (bid, fiscal_year_id, account_id, company_id),
    )
    conn.commit()
    return bid


def _set_account_fields(conn, account_id, **fields):
    """Directly update account columns the seed_account() helper doesn't
    expose (parent_id, disabled, is_frozen, lft, rgt) — test-only shortcut,
    field names are hardcoded by the caller, never user input."""
    set_clause = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE account SET {set_clause} WHERE id=?",
                 (*fields.values(), account_id))
    conn.commit()


def audit_rows_for(conn, entity_id):
    return conn.execute(
        "SELECT * FROM audit_log WHERE entity_type='account' AND entity_id=?",
        (entity_id,),
    ).fetchall()


def _full_state_snapshot(conn, account_id):
    """Capture the account row, its children, its gl_entry rows, and its
    journal_entry_line rows so a blocked delete can be proven to leave all
    of them byte-identical."""
    account = conn.execute(
        "SELECT * FROM account WHERE id=?", (account_id,)
    ).fetchone()
    children = conn.execute(
        "SELECT * FROM account WHERE parent_id=? ORDER BY id", (account_id,)
    ).fetchall()
    gl_rows = conn.execute(
        "SELECT * FROM gl_entry WHERE account_id=? ORDER BY id", (account_id,)
    ).fetchall()
    jel_rows = conn.execute(
        "SELECT * FROM journal_entry_line WHERE account_id=? ORDER BY id", (account_id,)
    ).fetchall()
    return {
        "account": dict(account) if account else None,
        "children": [dict(r) for r in children],
        "gl_entry": [dict(r) for r in gl_rows],
        "journal_entry_line": [dict(r) for r in jel_rows],
    }


def _delete(conn, account_id, company_id=None, company_name=None):
    return call_action(mod.delete_account, conn, ns(
        account_id=account_id, company_id=company_id, company_name=company_name,
    ))


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def env(conn):
    """Company + fiscal year, mirroring the pattern used by other GL test
    modules (see test_gl_entries.py's gl_setup fixture)."""
    cid = seed_company(conn)
    fyid = seed_fiscal_year(conn, cid)
    return {"company_id": cid, "fiscal_year_id": fyid}


# ──────────────────────────────────────────────────────────────────────────────
# AC-1: non-zero balance blocks deletion + Decimal-not-string regression
# ──────────────────────────────────────────────────────────────────────────────

class TestNonZeroBalanceBlocks:
    def test_nonzero_balance_blocks_deletion(self, conn, env):
        """AC-1: a plain non-zero balance must block deletion."""
        aid = seed_account(conn, env["company_id"], name="Nonzero Balance Acct")
        seed_gl_entry_raw(conn, aid, debit="100.00", credit="0")

        blockers = mod._account_delete_blockers(conn, aid)
        codes = {b["code"] for b in blockers}
        assert "non_zero_balance" in codes

        result = _delete(conn, aid)
        assert is_error(result)
        assert "Cannot delete account:" in result["message"]

    def test_zero_dot_zero_balance_string_is_not_treated_as_nonzero(self, conn, env):
        """AC-1 regression: balance is always formatted as a 2-decimal-place
        string (e.g. "0.00"), never bare "0". A naive
        ``balance_str != "0"`` check would incorrectly fire here. The
        comparison must be done as Decimal, where Decimal("0.00") ==
        Decimal("0"). Seed via a single cancelled gl_entry row: it is
        excluded from the balance sum entirely (is_cancelled=1), which is
        the simplest way to reach a "0.00"-formatted zero balance while
        real gl_entry rows exist for the account."""
        aid = seed_account(conn, env["company_id"], name="Cancelled-Zero Acct")
        seed_gl_entry_raw(conn, aid, debit="500.00", credit="0", is_cancelled=1)

        balance = mod._lib_get_account_balance(conn, aid)
        # Confirm the seeded scenario actually produces the "0.00" string
        # form this regression test targets (not just numeric zero).
        assert balance["balance"] == "0.00"
        assert Decimal(balance["balance"]) == Decimal("0")

        blockers = mod._account_delete_blockers(conn, aid)
        codes = {b["code"] for b in blockers}
        assert "non_zero_balance" not in codes


# ──────────────────────────────────────────────────────────────────────────────
# AC-2: posted GL history blocks even at zero net balance
# ──────────────────────────────────────────────────────────────────────────────

class TestPostedGLHistoryBlocks:
    def test_cancelled_only_gl_entry_still_blocks_via_posted_gl_history(self, conn, env):
        """AC-2: a single is_cancelled=1 gl_entry row (net balance zero)
        must still block deletion via posted_gl_history — gl_entry is
        append-only/checksum-chained, cancellation does not erase history."""
        aid = seed_account(conn, env["company_id"], name="GL History Acct")
        seed_gl_entry_raw(conn, aid, debit="500.00", credit="0", is_cancelled=1)

        blockers = mod._account_delete_blockers(conn, aid)
        codes = {b["code"] for b in blockers}
        assert "posted_gl_history" in codes

        result = _delete(conn, aid)
        assert is_error(result)


# ──────────────────────────────────────────────────────────────────────────────
# AC-2b: posted journal_entry_line (submitted/cancelled/amended) blocks
# ──────────────────────────────────────────────────────────────────────────────

class TestPostedJournalHistoryBlocks:
    @pytest.mark.parametrize("status", ["submitted", "cancelled", "amended"])
    def test_posted_status_triggers_posted_journal_history(self, conn, env, status):
        aid = seed_account(conn, env["company_id"], name=f"JE-{status} Acct")
        jid = seed_journal_entry_raw(conn, env["company_id"], status=status)
        seed_journal_entry_line_raw(conn, jid, aid, debit="100.00", credit="0")

        blockers = mod._account_delete_blockers(conn, aid)
        codes = {b["code"] for b in blockers}
        assert "posted_journal_history" in codes
        assert "draft_journal_reference" not in codes

        result = _delete(conn, aid)
        assert is_error(result)


# ──────────────────────────────────────────────────────────────────────────────
# AC-3: draft journal_entry_line -> distinct draft_journal_reference code
# ──────────────────────────────────────────────────────────────────────────────

class TestDraftJournalReferenceIsDistinctCode:
    def test_draft_parent_triggers_draft_code_not_posted_code(self, conn, env):
        aid = seed_account(conn, env["company_id"], name="JE-draft Acct")
        jid = seed_journal_entry_raw(conn, env["company_id"], status="draft")
        seed_journal_entry_line_raw(conn, jid, aid, debit="50.00", credit="0")

        blockers = mod._account_delete_blockers(conn, aid)
        codes = {b["code"] for b in blockers}
        assert "draft_journal_reference" in codes
        assert "posted_journal_history" not in codes

    def test_draft_reference_blocks_without_raising_raw_fk_error(self, conn, env):
        """The FK from journal_entry_line to account is a live RESTRICT FK,
        so a naive implementation might let sqlite3.IntegrityError bubble
        straight out of delete_account. Calling through call_action (which
        only swallows SystemExit) means any unhandled exception here would
        fail this test with a traceback — that is the crash we're
        checking does NOT happen."""
        aid = seed_account(conn, env["company_id"], name="JE-draft Acct 2")
        jid = seed_journal_entry_raw(conn, env["company_id"], status="draft")
        seed_journal_entry_line_raw(conn, jid, aid, debit="50.00", credit="0")

        result = _delete(conn, aid)
        assert is_error(result)
        assert "FOREIGN KEY" not in result["message"]
        row = conn.execute("SELECT * FROM account WHERE id=?", (aid,)).fetchone()
        assert row is not None


# ──────────────────────────────────────────────────────────────────────────────
# AC-4: child accounts block deletion regardless of disabled/is_frozen/lft/rgt
# ──────────────────────────────────────────────────────────────────────────────

class TestChildAccountsBlock:
    def test_child_blocks_regardless_of_disabled_and_frozen(self, conn, env):
        parent = seed_account(conn, env["company_id"], name="Parent Group", is_group=1)
        child = seed_account(conn, env["company_id"], name="Child Account")
        _set_account_fields(conn, child, parent_id=parent, disabled=1, is_frozen=1)

        blockers = mod._account_delete_blockers(conn, parent)
        codes = {b["code"] for b in blockers}
        assert "child_accounts" in codes

        result = _delete(conn, parent)
        assert is_error(result)

    def test_child_blocks_even_with_bogus_lft_rgt(self, conn, env):
        """lft/rgt are nullable and must not influence the child-account
        check at all — only parent_id matters. Set them to nonsensical
        values (rgt < lft) and confirm the blocker still fires exactly the
        same, and also confirm a NULL lft/rgt child still blocks."""
        parent = seed_account(conn, env["company_id"], name="Parent Group 2", is_group=1)
        child = seed_account(conn, env["company_id"], name="Child Account 2")
        _set_account_fields(conn, child, parent_id=parent, lft=999, rgt=1)

        blockers = mod._account_delete_blockers(conn, parent)
        codes = {b["code"] for b in blockers}
        assert "child_accounts" in codes

        # NULL lft/rgt (the seed_account() default) must also still block.
        parent2 = seed_account(conn, env["company_id"], name="Parent Group 3", is_group=1)
        child2 = seed_account(conn, env["company_id"], name="Child Account 3")
        _set_account_fields(conn, child2, parent_id=parent2, lft=None, rgt=None)
        blockers2 = mod._account_delete_blockers(conn, parent2)
        codes2 = {b["code"] for b in blockers2}
        assert "child_accounts" in codes2


# ──────────────────────────────────────────────────────────────────────────────
# AC-5: is_group alone must NOT block (negative test)
# ──────────────────────────────────────────────────────────────────────────────

class TestEmptyGroupAccountIsDeletable:
    def test_is_group_alone_does_not_block(self, conn, env):
        aid = seed_account(conn, env["company_id"], name="Empty Group", is_group=1)

        blockers = mod._account_delete_blockers(conn, aid)
        assert blockers == []

        result = _delete(conn, aid)
        assert is_ok(result)
        assert result["deleted"] is True

        row = conn.execute("SELECT * FROM account WHERE id=?", (aid,)).fetchone()
        assert row is None


# ──────────────────────────────────────────────────────────────────────────────
# AC-6: multiple simultaneous blockers -> one composed message with ALL reasons
# ──────────────────────────────────────────────────────────────────────────────

class TestMultipleBlockersComposedInOneMessage:
    def test_balance_and_child_both_reported_together(self, conn, env):
        aid = seed_account(conn, env["company_id"], name="Multi-blocker Acct")
        seed_gl_entry_raw(conn, aid, debit="500.00", credit="0")
        child = seed_account(conn, env["company_id"], name="Child of Multi-blocker")
        _set_account_fields(conn, child, parent_id=aid)

        blockers = mod._account_delete_blockers(conn, aid)
        codes = {b["code"] for b in blockers}
        # Non-zero balance implies posted GL history too (balance is only
        # ever derived from gl_entry rows), so at least these two plus
        # child_accounts must all be present simultaneously.
        assert "non_zero_balance" in codes
        assert "child_accounts" in codes
        assert len(blockers) >= 2

        result = _delete(conn, aid)
        assert is_error(result)
        msg = result["message"]
        assert msg.startswith("Cannot delete account:")
        # Every individual blocker message must appear in the composed
        # message — not just the first one found.
        for b in blockers:
            assert b["message"] in msg
        assert msg.count(";") == len(blockers) - 1


# ──────────────────────────────────────────────────────────────────────────────
# AC-7: generic IntegrityError safety net for FK tables outside the 5 checks
# ──────────────────────────────────────────────────────────────────────────────

class TestGenericSafetyNetForUncoveredForeignKey:
    def test_budget_reference_blocks_with_single_generic_message(self, conn, env):
        aid = seed_account(conn, env["company_id"], name="Budgeted Account")
        seed_budget_raw(conn, env["company_id"], env["fiscal_year_id"], aid)

        # Sanity check: none of the 5 named pre-checks should fire — this
        # proves the block below comes from the raw-FK safety net, not one
        # of the explicit blocker codes.
        blockers = mod._account_delete_blockers(conn, aid)
        assert blockers == []

        result = _delete(conn, aid)
        assert is_error(result)
        msg = result["message"]
        assert msg == "account is still referenced elsewhere and cannot be deleted"
        assert "FOREIGN KEY constraint failed" not in msg
        assert "budget" not in msg.lower()
        assert "account_id" not in msg.lower()

        # DELETE must have been rolled back — account row still present.
        row = conn.execute("SELECT * FROM account WHERE id=?", (aid,)).fetchone()
        assert row is not None


# ──────────────────────────────────────────────────────────────────────────────
# AC-9: happy path — single audit row, old_values populated, single commit
# ──────────────────────────────────────────────────────────────────────────────

class TestHappyPathDeletion:
    def test_clean_account_deletes_with_single_audit_row_and_commits(self, conn, env, db_path):
        aid = seed_account(conn, env["company_id"], name="Clean Deletable Account",
                            account_number="9999")
        before_audit_total = conn.execute(
            "SELECT COUNT(*) as c FROM audit_log"
        ).fetchone()["c"]

        result = _delete(conn, aid)
        assert is_ok(result)
        assert result["deleted"] is True
        assert result["account_id"] == aid

        audit_rows = audit_rows_for(conn, aid)
        assert len(audit_rows) == 1
        row = audit_rows[0]
        assert row["skill"] == "erpclaw-gl"
        assert row["action"] == "delete"
        assert row["entity_type"] == "account"
        assert row["old_values"] is not None
        old_values = json.loads(row["old_values"])
        assert old_values.get("name") == "Clean Deletable Account"

        after_audit_total = conn.execute(
            "SELECT COUNT(*) as c FROM audit_log"
        ).fetchone()["c"]
        assert after_audit_total == before_audit_total + 1

        # Committed for real: a brand-new connection to the same on-disk
        # database must also see the account gone.
        fresh = get_conn(db_path)
        try:
            fresh_row = fresh.execute(
                "SELECT * FROM account WHERE id=?", (aid,)
            ).fetchone()
            assert fresh_row is None
        finally:
            fresh.close()


# ──────────────────────────────────────────────────────────────────────────────
# AC-10: blocked deletions leave account/children/gl_entry/journal_entry_line
# state fully unchanged and write no audit row
# ──────────────────────────────────────────────────────────────────────────────

class TestBlockedDeletionLeavesStateUntouched:
    def _assert_blocked_and_untouched(self, conn, aid):
        before = _full_state_snapshot(conn, aid)
        before_audit_count = len(audit_rows_for(conn, aid))

        result = _delete(conn, aid)
        assert is_error(result)

        after = _full_state_snapshot(conn, aid)
        assert after == before
        assert len(audit_rows_for(conn, aid)) == before_audit_count

    def test_non_zero_balance_block_leaves_state_untouched(self, conn, env):
        aid = seed_account(conn, env["company_id"], name="AC10-balance")
        seed_gl_entry_raw(conn, aid, debit="250.00", credit="0")
        self._assert_blocked_and_untouched(conn, aid)

    def test_cancelled_gl_history_block_leaves_state_untouched(self, conn, env):
        aid = seed_account(conn, env["company_id"], name="AC10-gl-history")
        seed_gl_entry_raw(conn, aid, debit="250.00", credit="0", is_cancelled=1)
        self._assert_blocked_and_untouched(conn, aid)

    def test_posted_journal_block_leaves_state_untouched(self, conn, env):
        aid = seed_account(conn, env["company_id"], name="AC10-posted-je")
        jid = seed_journal_entry_raw(conn, env["company_id"], status="submitted")
        seed_journal_entry_line_raw(conn, jid, aid, debit="10.00", credit="0")
        self._assert_blocked_and_untouched(conn, aid)

    def test_draft_journal_block_leaves_state_untouched(self, conn, env):
        aid = seed_account(conn, env["company_id"], name="AC10-draft-je")
        jid = seed_journal_entry_raw(conn, env["company_id"], status="draft")
        seed_journal_entry_line_raw(conn, jid, aid, debit="10.00", credit="0")
        self._assert_blocked_and_untouched(conn, aid)

    def test_child_account_block_leaves_state_untouched(self, conn, env):
        parent = seed_account(conn, env["company_id"], name="AC10-parent", is_group=1)
        child = seed_account(conn, env["company_id"], name="AC10-child")
        _set_account_fields(conn, child, parent_id=parent)
        self._assert_blocked_and_untouched(conn, parent)

    def test_fk_safety_net_block_leaves_state_untouched(self, conn, env):
        aid = seed_account(conn, env["company_id"], name="AC10-budget-fk")
        seed_budget_raw(conn, env["company_id"], env["fiscal_year_id"], aid)
        self._assert_blocked_and_untouched(conn, aid)
