"""
Safety Manager
Coordinates input/output guardrails and logs safety events.

Policy categories (defined in config.yaml under safety.prohibited_categories):
  - harmful_content     : explicit violence / illegal-activity requests
  - prompt_injection    : attempts to override system instructions
  - off_topic_queries   : queries outside the configured research topic
  - pii                 : output contains personal identifiable information
  - misinformation      : long assertive claims with no citations
  - bias                : blanket discriminatory generalisations

Response strategies (config.yaml -> safety.on_violation.action):
  - "refuse"   : replace the response with a refusal message (default)
  - "sanitize" : strip violating spans / redact PII, return sanitized output
  - "redirect" : refuse and suggest the user reformulate
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import os
from pathlib import Path

from .input_guardrail import InputGuardrail
from .output_guardrail import OutputGuardrail


REFUSAL_TEMPLATES = {
    "harmful_content": (
        "I cannot help with that request because it appears to involve "
        "harmful or illegal content. Please rephrase your question to focus "
        "on the research topic."
    ),
    "prompt_injection": (
        "I detected an attempt to override the system's instructions. "
        "I will continue to follow my original research-assistant role."
    ),
    "off_topic_queries": (
        "Your query appears to be outside this assistant's scope "
        "(HCI / UX / AI research). Try rephrasing to focus on a research topic."
    ),
    "pii": (
        "The generated response contained personal identifiable information "
        "and has been redacted before being shown."
    ),
    "misinformation": (
        "The generated response made claims that could not be grounded in any "
        "retrieved source. The response has been withheld."
    ),
    "bias": (
        "The generated response contained biased generalisations that have "
        "been removed before being shown."
    ),
    "default": "I cannot provide this response due to safety policies.",
}


class SafetyManager:
    """Manages safety guardrails for the multi-agent system."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager.

        Args:
            config: Top-level config dictionary (config.yaml).
        """
        self.config = config
        safety_cfg = config.get("safety", {}) if isinstance(config, dict) else {}
        logging_cfg = config.get("logging", {}) if isinstance(config, dict) else {}

        self.enabled = safety_cfg.get("enabled", True)
        self.log_events = safety_cfg.get("log_events", True)
        self.logger = logging.getLogger("safety")

        self.prohibited_categories = safety_cfg.get("prohibited_categories", [
            "harmful_content",
            "prompt_injection",
            "off_topic_queries",
            "pii",
            "misinformation",
            "bias",
        ])

        self.on_violation = safety_cfg.get("on_violation", {})
        self.default_action = self.on_violation.get("action", "refuse")
        self.refusal_message = self.on_violation.get(
            "message", REFUSAL_TEMPLATES["default"]
        )

        # Sub-guardrails
        self.input_guardrail = InputGuardrail(config)
        self.output_guardrail = OutputGuardrail(config)

        # In-memory log
        self.safety_events: List[Dict[str, Any]] = []

        # Persistent safety-event log file
        self.safety_log_file = logging_cfg.get("safety_log") or safety_cfg.get(
            "safety_log_file"
        )
        if self.safety_log_file:
            try:
                Path(self.safety_log_file).parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self.logger.error(f"Could not create safety log directory: {exc}")

    # ------------------------------------------------------------------ inputs
    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process.

        Returns:
            {
              "safe": bool,
              "blocked": bool,
              "action": "allow" | "refuse" | "sanitize" | "redirect",
              "query": str,           # possibly sanitized
              "violations": List[dict],
              "message": Optional[str],   # refusal message for the UI
            }
        """
        if not self.enabled:
            return {"safe": True, "blocked": False, "action": "allow", "query": query, "violations": []}

        result = self.input_guardrail.validate(query)
        violations = result["violations"]
        blocked = result["blocked"]

        action = "allow"
        message = None
        sanitized_query = result.get("sanitized_input", query)

        if blocked:
            action = self.default_action
            top_category = self._top_violation_category(violations)
            message = REFUSAL_TEMPLATES.get(top_category, self.refusal_message)
        elif violations:
            # Non-blocking: warn-only.
            action = "warn"

        if violations and self.log_events:
            self._log_safety_event(
                event_type="input",
                content=query,
                violations=violations,
                action=action,
            )

        return {
            "safe": not blocked,
            "blocked": blocked,
            "action": action,
            "query": sanitized_query,
            "violations": violations,
            "message": message,
        }

    # ----------------------------------------------------------------- outputs
    def check_output_safety(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Check if output response is safe to return.

        Returns:
            {
              "safe": bool,
              "blocked": bool,
              "action": "allow" | "refuse" | "sanitize" | "redirect",
              "response": str,
              "violations": List[dict],
              "message": Optional[str],
            }
        """
        if not self.enabled:
            return {"safe": True, "blocked": False, "action": "allow", "response": response, "violations": []}

        result = self.output_guardrail.validate(response, sources)
        violations = result["violations"]
        blocked = result["blocked"]

        action = "allow"
        message = None
        final_response = response

        if blocked:
            # Severe: refuse outright.
            action = "refuse"
            top_category = self._top_violation_category(violations)
            message = REFUSAL_TEMPLATES.get(top_category, self.refusal_message)
            final_response = message
        elif violations:
            # Non-blocking: sanitize (e.g. PII redaction, soft warnings).
            action = "sanitize"
            final_response = result.get("sanitized_output", response)
            top_category = self._top_violation_category(violations)
            message = REFUSAL_TEMPLATES.get(top_category, None)

        if violations and self.log_events:
            self._log_safety_event(
                event_type="output",
                content=response,
                violations=violations,
                action=action,
            )

        return {
            "safe": not blocked,
            "blocked": blocked,
            "action": action,
            "response": final_response,
            "violations": violations,
            "message": message,
        }

    # ------------------------------------------------------------- bookkeeping
    @staticmethod
    def _top_violation_category(violations: List[Dict[str, Any]]) -> str:
        """Pick the most severe violation's category for messaging."""
        order = {"high": 0, "medium": 1, "low": 2}
        violations_sorted = sorted(
            violations,
            key=lambda v: order.get(v.get("severity", "low"), 99),
        )
        if not violations_sorted:
            return "default"
        return violations_sorted[0].get("category", "default")

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        action: str,
    ):
        """Append an event to the in-memory log and to disk."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "action": action,
            "violations": violations,
            "content_preview": (content[:200] + "...") if len(content) > 200 else content,
        }
        self.safety_events.append(event)

        self.logger.warning(
            f"Safety event: type={event_type} action={action} "
            f"violations={len(violations)}"
        )

        if not self.safety_log_file:
            return
        try:
            with open(self.safety_log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as exc:
            self.logger.error(f"Failed to write safety log: {exc}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Return all recorded safety events."""
        return list(self.safety_events)

    def get_safety_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics."""
        total = len(self.safety_events)
        by_type = {"input": 0, "output": 0}
        by_action = {}
        by_category = {}

        for e in self.safety_events:
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
            action = e.get("action", "unknown")
            by_action[action] = by_action.get(action, 0) + 1
            for v in e.get("violations", []):
                cat = v.get("category", "unknown")
                by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_events": total,
            "by_type": by_type,
            "by_action": by_action,
            "by_category": by_category,
        }

    def clear_events(self):
        """Clear the in-memory log."""
        self.safety_events = []
