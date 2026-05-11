"""
Input Guardrail
Checks user inputs for safety violations.

This implements a custom rule-based policy filter that runs before the
multi-agent orchestrator processes any query. It detects four broad
violation categories:

  1. harmful_content    - explicit requests for violence, weapons, illegal acts
  2. prompt_injection   - attempts to override the system instructions
  3. off_topic_queries  - queries far outside the configured research topic
  4. malformed_input    - empty, too short, too long, or non-text input

Each violation includes a severity (low / medium / high) so the
SafetyManager can decide whether to refuse, sanitize, or simply warn.
"""

from typing import Dict, Any, List
import re


class InputGuardrail:
    """
    Guardrail for checking input safety.

    The guardrail is intentionally lightweight (rule-based) so it runs
    quickly and deterministically without an external API key. It can be
    extended to call Guardrails AI or NeMo Guardrails — see the
    `_check_with_external_validator` hook below.
    """

    # Keyword/phrase lists used by the rule-based policy.
    HARMFUL_PATTERNS = [
        r"\bhow\s+to\s+(make|build|create)\s+a?\s*(bomb|explosive|weapon)",
        r"\bkill\s+(myself|someone|people)",
        r"\bself[-\s]?harm\b",
        r"\bchild\s+(porn|sexual)",
        r"\bsynthes(ize|is)\s+(meth|fentanyl|cocaine|heroin)",
        r"\bdoxx?\s+",
        r"\bhack\s+into\s+(someone|a\s+\w+)",
        r"\b(launder|laundering)\s+money\b",
    ]

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions",
        r"disregard\s+(your|the|all)\s+(instructions|rules|guidelines)",
        r"forget\s+(everything|all\s+rules|previous)",
        r"reveal\s+(your|the)\s+system\s+prompt",
        r"\bact\s+as\s+(if\s+you\s+have\s+no\s+restrictions|dan|developer\s+mode)",
        r"\bjailbreak\b",
        r"<\s*\|.*system.*\|\s*>",
    ]

    # Topic-relevance keywords for HCI / UX / AI research.
    ON_TOPIC_KEYWORDS = {
        "hci", "human-computer", "human computer", "ui", "ux", "user interface",
        "user experience", "interaction", "interface", "design", "usability",
        "accessibility", "ai", "artificial intelligence", "ml", "machine learning",
        "agent", "chatbot", "conversational", "voice", "speech", "ar", "vr", "xr",
        "augmented", "virtual reality", "research", "study", "evaluation",
        "prototype", "prototyping", "visualization", "dashboard", "tool",
        "wcag", "accessibility", "inclusive", "ethics", "ethical", "privacy",
        "explainable", "transparency", "trust", "education", "healthcare",
        "design pattern", "system", "user", "users", "human", "cognitive",
        "perception", "experience", "guidelines", "principles", "best practices",
        "trends", "framework", "model", "models", "review", "literature",
    }

    SEVERITY_BLOCK_THRESHOLD = "high"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Top-level config dictionary (config.yaml). Reads from
                config["safety"] and config["system"] when present.
        """
        self.config = config
        safety_cfg = config.get("safety", {}) if isinstance(config, dict) else {}
        system_cfg = config.get("system", {}) if isinstance(config, dict) else {}

        self.min_length = safety_cfg.get("min_query_length", 5)
        self.max_length = safety_cfg.get("max_query_length", 2000)
        self.topic = system_cfg.get("topic", "HCI Research")
        self.enforce_topic = safety_cfg.get("enforce_topic", False)

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query against the policy.

        Args:
            query: User input to validate

        Returns:
            {
              "valid": bool,                # passes all checks
              "blocked": bool,              # at least one high-severity violation
              "violations": List[dict],     # each {validator, category, reason, severity}
              "sanitized_input": str,       # query with injection markers stripped
            }
        """
        violations: List[Dict[str, Any]] = []

        if not isinstance(query, str):
            violations.append({
                "validator": "format",
                "category": "malformed_input",
                "reason": "Input is not a string",
                "severity": "high",
            })
            return {
                "valid": False,
                "blocked": True,
                "violations": violations,
                "sanitized_input": "",
            }

        normalized = query.strip()

        # 1) Length checks
        if len(normalized) < self.min_length:
            violations.append({
                "validator": "length",
                "category": "malformed_input",
                "reason": f"Query is too short (< {self.min_length} chars)",
                "severity": "low",
            })
        if len(normalized) > self.max_length:
            violations.append({
                "validator": "length",
                "category": "malformed_input",
                "reason": f"Query exceeds max length ({self.max_length} chars)",
                "severity": "medium",
            })

        # 2) Harmful content
        violations.extend(self._check_toxic_language(normalized))

        # 3) Prompt injection
        injection_violations, sanitized = self._check_prompt_injection(normalized)
        violations.extend(injection_violations)

        # 4) Topic relevance (only flagged, not auto-blocked, unless enforce_topic)
        violations.extend(self._check_relevance(normalized))

        # Determine final disposition
        blocked = any(v.get("severity") == self.SEVERITY_BLOCK_THRESHOLD for v in violations)
        valid = len(violations) == 0

        return {
            "valid": valid,
            "blocked": blocked,
            "violations": violations,
            "sanitized_input": sanitized,
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """Detect explicit harmful / illegal requests via regex."""
        violations: List[Dict[str, Any]] = []
        lowered = text.lower()
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, lowered):
                violations.append({
                    "validator": "harmful_content",
                    "category": "harmful_content",
                    "reason": f"Matched harmful pattern: {pattern}",
                    "severity": "high",
                })
        return violations

    def _check_prompt_injection(self, text: str) -> (List[Dict[str, Any]], str):
        """Detect prompt-injection attempts and return a sanitized copy."""
        violations: List[Dict[str, Any]] = []
        sanitized = text
        lowered = text.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, lowered):
                violations.append({
                    "validator": "prompt_injection",
                    "category": "prompt_injection",
                    "reason": f"Matched injection pattern: {pattern}",
                    "severity": "high",
                })
                # Remove injection-suspect lines from sanitized version.
                sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)
        return violations, sanitized

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Lightweight topic-relevance heuristic. We flag only if no on-topic
        keyword is detected; never high-severity unless enforce_topic is set.
        """
        violations: List[Dict[str, Any]] = []
        lowered = query.lower()
        if any(kw in lowered for kw in self.ON_TOPIC_KEYWORDS):
            return violations

        severity = "high" if self.enforce_topic else "low"
        violations.append({
            "validator": "topic_relevance",
            "category": "off_topic_queries",
            "reason": f"Query does not appear related to '{self.topic}'",
            "severity": severity,
        })
        return violations

    def _check_with_external_validator(self, query: str) -> List[Dict[str, Any]]:
        """
        Hook for plugging in Guardrails AI / NeMo Guardrails / a moderation API.
        Returns an empty list by default so the system works offline.
        """
        return []
