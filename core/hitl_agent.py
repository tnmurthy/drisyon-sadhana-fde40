"""
Drisyon Sadhana · Forward Deployed AI Engineer Starter Kit
Module 4: Human-in-the-Loop (HITL) Execution Controller
"""

from typing import Dict, Any, List


class HITLController:
    """Enforces human approval gates on state-mutating enterprise tool calls."""

    READ_ONLY_TOOLS = {"search_docs", "get_user_info", "query_database", "check_balance"}
    STATE_MUTATING_TOOLS = {"issue_refund", "update_erp_record", "delete_file", "send_email"}

    def __init__(self):
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}

    def intercept_tool_call(self, call_id: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intercepts tool call.
        If read-only -> executes immediately (status: 'authorized').
        If mutating -> queues for human review (status: 'approval_required').
        """
        if tool_name in self.READ_ONLY_TOOLS:
            return {
                "call_id": call_id,
                "status": "authorized",
                "tool_name": tool_name,
                "requires_human": False
            }

        if tool_name in self.STATE_MUTATING_TOOLS:
            self.pending_approvals[call_id] = {
                "tool_name": tool_name,
                "args": args,
                "approved": False
            }
            return {
                "call_id": call_id,
                "status": "approval_required",
                "tool_name": tool_name,
                "requires_human": True,
                "message": f"Action '{tool_name}' requires operator approval."
            }

        return {
            "call_id": call_id,
            "status": "rejected",
            "message": f"Unrecognized tool: {tool_name}"
        }

    def approve_action(self, call_id: str) -> bool:
        """Approves a pending mutating action."""
        if call_id in self.pending_approvals:
            self.pending_approvals[call_id]["approved"] = True
            return True
        return False
