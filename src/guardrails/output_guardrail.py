"""
Output Guardrail
Checks system outputs for safety violations before they are surfaced to the user.

Validation categories:
  1. pii              - emails, phone numbers, SSNs, credit-card-shaped numbers
  2. harmful_content  - explicit instructions for violence / illegal activity
  3. unsupported_claims - flags responses that make hard claims with zero
                         citations (a weak proxy for misinformation risk)
  4. bias             - blanket discriminatory generalisations

Each violation has a severity. The output may be sanitized (PII redacted) or
the response may be refused outright when a high-severity violation appears.
"""

from typing import Dict, Any, List
import re


class OutputGuardrail:
    """Guardrail for inspecting and sanitizing model outputs."""

    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone": r"\b(?:\+?\d{1,2}\s*)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    }

    HARMFUL_OUTPUT_PATTERNS = [
        r"\b(here\s+is|here'?s)\s+how\s+to\s+(make|build)\s+a?\s*(bomb|explosive|weapon)",
        r"\bstep[-\s]?by[-\s]?step\s+guide\s+to\s+(hack|exploit)\b",
        r"\bsynthes(ize|is)\s+(meth|fentanyl|cocaine|heroin)",
        r"\b(kill|murder)\s+(someone|people|a\s+person)\b",
        r"\binstructions?\s+for\s+(making|building|creating)\s+(a\s+)?(bomb|explosive|weapon)",
    ]

    BIAS_PATTERNS = [
        r"\ball\s+(women|men|asians|africans|whites|blacks|jews|muslims|christians)\s+are\b",
        r"\b(women|men)\s+are\s+(inferior|superior)\b",
    ]

    SEVERITY_BLOCK_THRESHOLD = "high"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Top-level config dictionary (config.yaml).
        """
        self.config = config
        safety_cfg = config.get("safety", {}) if isinstance(config, dict) else {}
        self.redact_pii = safety_cfg.get("redact_pii", True)
        self.require_citations = safety_cfg.get("require_citations", False)

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources: Optional list of sources used (for fact-checking)

        Returns:
            {
              "valid": bool,
              "blocked": bool,
              "violations": List[dict],
              "sanitized_output": str,
            }
        """
        violations: List[Dict[str, Any]] = []

        if not isinstance(response, str):
            return {
                "valid": False,
                "blocked": True,
                "violations": [{
                    "validator": "format",
                    "category": "malformed_output",
                    "reason": "Output is not a string",
                    "severity": "high",
                }],
                "sanitized_output": "",
            }

        violations.extend(self._check_pii(response))
        violations.extend(self._check_harmful_content(response))
        violations.extend(self._check_bias(response))

        if sources is not None:
            violations.extend(self._check_factual_consistency(response, sources))

        sanitized = self._sanitize(response, violations) if violations else response
        blocked = any(v.get("severity") == self.SEVERITY_BLOCK_THRESHOLD
                      and v.get("category") in {"harmful_content", "bias", "malformed_output"}
                      for v in violations)

        return {
            "valid": len(violations) == 0,
            "blocked": blocked,
            "violations": violations,
            "sanitized_output": sanitized,
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect common PII patterns."""
        violations: List[Dict[str, Any]] = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            # Filter obvious false-positives for credit_card (must have >= 13 digits).
            if pii_type == "credit_card":
                matches = [m for m in matches if sum(c.isdigit() for c in m) >= 13]
            if matches:
                violations.append({
                    "validator": "pii",
                    "category": "pii",
                    "pii_type": pii_type,
                    "reason": f"Output contains {pii_type}",
                    "severity": "high" if pii_type in {"ssn", "credit_card"} else "medium",
                    "matches": matches,
                })
        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """Detect explicit harmful instructions in the output."""
        violations: List[Dict[str, Any]] = []
        lowered = text.lower()
        for pattern in self.HARMFUL_OUTPUT_PATTERNS:
            if re.search(pattern, lowered):
                violations.append({
                    "validator": "harmful_content",
                    "category": "harmful_content",
                    "reason": f"Matched harmful pattern: {pattern}",
                    "severity": "high",
                })
        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """Detect blanket discriminatory generalisations."""
        violations: List[Dict[str, Any]] = []
        lowered = text.lower()
        for pattern in self.BIAS_PATTERNS:
            if re.search(pattern, lowered):
                violations.append({
                    "validator": "bias",
                    "category": "bias",
                    "reason": f"Possible biased generalisation: {pattern}",
                    "severity": "medium",
                })
        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Heuristic citation check: if the response makes assertive claims but
        no URL/citation appears anywhere, flag a low-severity violation. This
        is a weak misinformation proxy; a deeper check would call an LLM
        verifier and ground each claim against sources.
        """
        violations: List[Dict[str, Any]] = []
        if not self.require_citations:
            return violations

        has_citation = bool(re.search(r"https?://", response)) or "[Source:" in response
        looks_assertive = len(response.split()) > 80
        if looks_assertive and not has_citation and not sources:
            violations.append({
                "validator": "citation_grounding",
                "category": "misinformation",
                "reason": "Long assertive response with no citations or sources",
                "severity": "low",
            })
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Redact PII and strip lines matching harmful/bias patterns."""
        sanitized = text

        # Redact PII spans.
        for v in violations:
            if v.get("validator") == "pii":
                for match in v.get("matches", []):
                    sanitized = sanitized.replace(match, f"[REDACTED:{v.get('pii_type','pii').upper()}]")

        # Strip harmful/bias lines.
        for v in violations:
            if v.get("category") in {"harmful_content", "bias"}:
                pattern = v.get("reason", "").replace("Matched harmful pattern: ", "").replace("Possible biased generalisation: ", "")
                try:
                    sanitized = re.sub(
                        rf".*{pattern}.*",
                        "[REMOVED FOR SAFETY]",
                        sanitized,
                        flags=re.IGNORECASE,
                    )
                except re.error:
                    pass

        return sanitized
