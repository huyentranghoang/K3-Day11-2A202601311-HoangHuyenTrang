"""
Lab 11 — Part 4: Human-in-the-Loop Design
  TODO 11: Confidence Router
  TODO 12: Design 3 HITL decision points
"""
from dataclasses import dataclass


# ============================================================
# TODO 11: Implement ConfidenceRouter
#
# Route agent responses based on confidence scores:
#   - HIGH (>= 0.9): Auto-send to user
#   - MEDIUM (0.7 - 0.9): Queue for human review
#   - LOW (< 0.7): Escalate to human immediately
#
# Special case: if the action is HIGH_RISK (e.g., money transfer,
# account deletion), ALWAYS escalate regardless of confidence.
#
# Implement the route() method.
# ============================================================

HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Thresholds:
        HIGH:   confidence >= 0.9 -> auto-send
        MEDIUM: 0.7 <= confidence < 0.9 -> queue for review
        LOW:    confidence < 0.7 -> escalate to human

    High-risk actions always escalate regardless of confidence.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        if action_type in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {action_type}",
                priority="high",
                requires_human=True,
            )

        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )

        if confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence — needs review",
                priority="normal",
                requires_human=True,
            )

        return RoutingDecision(
            action="escalate",
            confidence=confidence,
            reason="Low confidence — escalating",
            priority="high",
            requires_human=True,
        )


# ============================================================
# TODO 12: Design 3 HITL decision points + a review lifecycle
#
# For each decision point, define:
# - trigger: What condition activates this HITL check?
# - hitl_model: Which model? (human-in-the-loop, human-on-the-loop,
#   human-as-tiebreaker)
# - context_needed: What info does the human reviewer need?
# - example: A concrete scenario
# - approval_path: What approve/reject/timeout decision is recorded?
# - audit_fields: Which correlation ID, intent and proposed action/diff are logged?
#
# Think about real banking scenarios where human judgment is critical.
# ============================================================

@dataclass
class ReviewRecord:
    """One human decision linked to a correlation/request ID."""
    correlation_id: str
    intent: str
    proposed_diff: str
    decision: str  # approve | reject | timeout
    reviewer_id: str | None
    reason: str


class ReviewLifecycle:
    """Minimal fail-closed HITL store for high-risk actions.

    Why: high-risk banking actions must never auto-send; each decision needs
    intent + diff + approve/reject/timeout and an audit correlation ID.
    """

    VALID = frozenset({"approve", "reject", "timeout"})

    def __init__(self):
        self.records: list[ReviewRecord] = []

    def decide(
        self,
        *,
        correlation_id: str,
        intent: str,
        proposed_diff: str,
        decision: str,
        reviewer_id: str | None = None,
        reason: str = "",
    ) -> ReviewRecord:
        """Record a human decision; unknown decisions become timeout (fail closed)."""
        normalized = (decision or "").lower().strip()
        if normalized not in self.VALID:
            normalized = "timeout"
            reason = reason or "invalid decision → fail closed as timeout"
        if normalized == "timeout" and not reviewer_id:
            reason = reason or "no reviewer response before SLA"
        record = ReviewRecord(
            correlation_id=correlation_id,
            intent=intent,
            proposed_diff=proposed_diff,
            decision=normalized,
            reviewer_id=reviewer_id,
            reason=reason,
        )
        self.records.append(record)
        return record

    def is_approved(self, correlation_id: str) -> bool:
        """Return True only if the latest decision for this ID is approve."""
        for record in reversed(self.records):
            if record.correlation_id == correlation_id:
                return record.decision == "approve"
        return False


hitl_decision_points = [
    {
        "id": 1,
        "name": "High-risk money transfer approval",
        "trigger": (
            "Agent proposes transfer_money (or any HIGH_RISK_ACTIONS money movement) "
            "to an external/beneficiary account - never auto-send."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Customer identity & auth level, source/destination accounts, amount, "
            "currency, fee, recent login/device risk signals, proposed tool call diff, "
            "and egress destination URL."
        ),
        "example": (
            "Customer asks to transfer 50,000,000 VND to a new beneficiary added today. "
            "Router escalates despite high model confidence."
        ),
        "approval_path": (
            "approve → execute transfer with approval_id; reject → cancel and notify customer; "
            "timeout (e.g. 15 min) → fail closed, no egress, ask customer to retry."
        ),
        "audit_fields": (
            "correlation_id/request_id, customer_id, intent=transfer_money, "
            "proposed_action_diff (from→to, amount), destination allowlist result, "
            "reviewer_id, decision (approve|reject|timeout), timestamp."
        ),
    },
    {
        "id": 2,
        "name": "Account closure / credential change review",
        "trigger": (
            "Action type is close_account, change_password, delete_data, or "
            "update_personal_info - irreversible or identity-changing operations."
        ),
        "hitl_model": "human-in-the-loop",
        "context_needed": (
            "Requested change vs current profile (field-level diff), verification steps "
            "already completed (OTP/KYC), channel used, and whether request came from "
            "untrusted email/RAG content."
        ),
        "example": (
            "User (or injected email text) asks to close a joint savings account and "
            "change the linked phone number in one turn."
        ),
        "approval_path": (
            "approve → apply changes and send confirmation; reject → keep prior state and "
            "log reason; timeout → fail closed, leave account unchanged."
        ),
        "audit_fields": (
            "correlation_id, intent, before/after personal_info diff, risk flags, "
            "reviewer_id, decision, SLA/timeout deadline."
        ),
    },
    {
        "id": 3,
        "name": "Medium-confidence response / egress gate",
        "trigger": (
            "ConfidenceRouter returns queue_review (0.7-0.9) OR pipeline wants to call "
            "is_egress_allowed for outbound API with ambiguous/sensitive payload."
        ),
        "hitl_model": "human-on-the-loop",
        "context_needed": (
            "Original user question, draft agent reply (PII-redacted preview), confidence "
            "score, judge/guardrail signals, destination URL, payload summary without secrets."
        ),
        "example": (
            "Agent drafts a reply summarizing an external invoice email that mentions "
            "account numbers; confidence 0.82 before posting to VinBank API."
        ),
        "approval_path": (
            "approve → send reply and/or allow egress; reject → block egress and return "
            "safe fallback; timeout → fail closed (no send / no egress)."
        ),
        "audit_fields": (
            "correlation_id, intent, draft_response_hash, confidence, "
            "egress_destination, is_egress_allowed result, reviewer decision, block_reason."
        ),
    },
]


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    Example:  {point['example']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
