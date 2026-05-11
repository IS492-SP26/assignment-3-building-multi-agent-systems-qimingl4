"""
LLM-as-a-Judge

Implements *two independent judging perspectives* — a supportive academic
reviewer and a strict critical reviewer — and aggregates their scores per
criterion. Two perspectives reduce single-rubric bias and produce a more
robust overall score (the "judge ensemble" pattern).

Example usage:
    judge = LLMJudge(config)
    result = await judge.evaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        sources=[],
        ground_truth="Paris",
    )
    print(result["overall_score"])             # weighted average across criteria/perspectives
    print(result["criterion_scores"])          # per-criterion, per-perspective scores
    print(result["perspective_scores"])        # average score per perspective

Provider support:
    config["models"]["judge"]["provider"] = "groq" | "openai" | "vllm"
    Falls back to Groq if "GROQ_API_KEY" is set; else tries the OpenAI-compatible
    client at OPENAI_BASE_URL.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import os
import re
from datetime import datetime
from pathlib import Path


# Two independent judging perspectives.
JUDGE_PERSPECTIVES: List[Dict[str, str]] = [
    {
        "name": "supportive_reviewer",
        "system": (
            "You are a supportive but rigorous academic reviewer. "
            "Reward responses that are well-organised, draw on evidence, and "
            "directly address the user's research question. Penalise vague or "
            "off-topic responses but give partial credit for partially correct "
            "answers. Always return strictly valid JSON."
        ),
        "rubric_hint": (
            "Scoring scale (0.0–1.0):\n"
            "  1.0 = exemplary, fully meets the criterion\n"
            "  0.7 = good, minor weaknesses\n"
            "  0.4 = mediocre, partial credit\n"
            "  0.0 = does not meet the criterion at all"
        ),
    },
    {
        "name": "strict_reviewer",
        "system": (
            "You are a strict, sceptical peer reviewer at a top HCI venue. "
            "Demand citations for every non-trivial claim, penalise unsupported "
            "assertions, vague language, and scope drift. Be conservative when "
            "scoring. Always return strictly valid JSON."
        ),
        "rubric_hint": (
            "Scoring scale (0.0–1.0), be conservative:\n"
            "  1.0 = publishable as-is\n"
            "  0.7 = needs minor revisions\n"
            "  0.4 = needs major revisions\n"
            "  0.0 = reject"
        ),
    },
]


class LLMJudge:
    """LLM-based judge for evaluating system responses."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.

        Args:
            config: Top-level configuration dictionary (config.yaml).
        """
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        self.model_config = config.get("models", {}).get("judge", {})
        self.criteria = config.get("evaluation", {}).get("criteria", [])

        self.provider = self.model_config.get("provider", "groq")
        self.model_name = self.model_config.get("name", "llama-3.1-8b-instant")
        self.temperature = self.model_config.get("temperature", 0.3)
        self.max_tokens = self.model_config.get("max_tokens", 1024)

        # Lazily initialise clients.
        self._groq_client = None
        self._openai_client = None
        self._init_client()

        # Optional raw-prompt/response log for repo artifacts.
        # Activate by setting JUDGE_DEBUG_LOG=path or via config.evaluation.debug_log.
        debug_log = os.getenv("JUDGE_DEBUG_LOG") or (
            config.get("evaluation", {}).get("debug_log") if isinstance(config, dict) else None
        )
        self._debug_log_path: Optional[Path] = Path(debug_log) if debug_log else None
        if self._debug_log_path:
            self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Per-evaluation raw call list (always populated; small, in-memory).
        self.last_raw_calls: List[Dict[str, Any]] = []

        self.logger.info(
            f"LLMJudge ready: provider={self.provider} model={self.model_name} "
            f"criteria={len(self.criteria)} perspectives={len(JUDGE_PERSPECTIVES)}"
        )

    def _init_client(self):
        """Initialise either a Groq or OpenAI-compatible client based on env."""
        if self.provider == "groq" or os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq

                api_key = os.getenv("GROQ_API_KEY")
                if api_key:
                    self._groq_client = Groq(api_key=api_key)
                    return
            except Exception as exc:
                self.logger.warning(f"Could not init Groq client: {exc}")

        if self.provider in {"openai", "vllm"} or os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                base_url = os.getenv("OPENAI_BASE_URL")
                if api_key:
                    self._openai_client = OpenAI(api_key=api_key, base_url=base_url)
            except Exception as exc:
                self.logger.warning(f"Could not init OpenAI-compat client: {exc}")

    # ------------------------------------------------------------------ public
    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a response across all criteria and judge perspectives.

        Returns a dict shaped like:
            {
              "query": str,
              "overall_score": float,
              "criterion_scores": {
                  <criterion>: {
                      "score": <avg across perspectives>,
                      "perspectives": {
                          <perspective>: {"score": float, "reasoning": str}
                      },
                      "criterion": <criterion>
                  }
              },
              "perspective_scores": {<perspective>: float}, # avg across criteria
              "feedback": List[str],
            }
        """
        self.logger.info(f"Evaluating response (query='{query[:60]}...')")
        # Reset per-evaluation raw call list.
        self.last_raw_calls = []

        results: Dict[str, Any] = {
            "query": query,
            "overall_score": 0.0,
            "criterion_scores": {},
            "perspective_scores": {p["name"]: [] for p in JUDGE_PERSPECTIVES},
            "feedback": [],
        }

        total_weight = sum(c.get("weight", 1.0) for c in self.criteria) or 1.0
        weighted_score = 0.0

        for criterion in self.criteria:
            criterion_name = criterion.get("name", "unknown")
            weight = criterion.get("weight", 1.0)

            perspective_results: Dict[str, Dict[str, Any]] = {}
            perspective_scores: List[float] = []
            for perspective in JUDGE_PERSPECTIVES:
                pres = await self._judge_criterion(
                    criterion=criterion,
                    perspective=perspective,
                    query=query,
                    response=response,
                    sources=sources,
                    ground_truth=ground_truth,
                )
                perspective_results[perspective["name"]] = pres
                perspective_scores.append(pres.get("score", 0.0))
                results["perspective_scores"][perspective["name"]].append(
                    pres.get("score", 0.0)
                )

            # Average across perspectives for this criterion.
            avg_score = (
                sum(perspective_scores) / len(perspective_scores)
                if perspective_scores
                else 0.0
            )
            results["criterion_scores"][criterion_name] = {
                "score": avg_score,
                "perspectives": perspective_results,
                "criterion": criterion_name,
            }
            weighted_score += avg_score * weight

        # Average per-perspective scores across criteria.
        results["perspective_scores"] = {
            name: (sum(scores) / len(scores) if scores else 0.0)
            for name, scores in results["perspective_scores"].items()
        }
        results["overall_score"] = weighted_score / total_weight
        return results

    # ----------------------------------------------------------- per-criterion
    async def _judge_criterion(
        self,
        criterion: Dict[str, Any],
        perspective: Dict[str, str],
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
    ) -> Dict[str, Any]:
        """Score a single (criterion, perspective) pair."""
        criterion_name = criterion.get("name", "unknown")
        description = criterion.get("description", "")

        prompt = self._create_judge_prompt(
            criterion_name=criterion_name,
            description=description,
            query=query,
            response=response,
            sources=sources,
            ground_truth=ground_truth,
            perspective=perspective,
        )

        try:
            judgment = await self._call_judge_llm(
                system_prompt=perspective["system"],
                user_prompt=prompt,
            )
            score, reasoning = self._parse_judgment(judgment)
            raw_call = {
                "timestamp": datetime.now().isoformat(),
                "criterion": criterion_name,
                "perspective": perspective["name"],
                "system_prompt": perspective["system"],
                "user_prompt": prompt,
                "raw_response": judgment,
                "parsed_score": score,
                "parsed_reasoning": reasoning,
            }
            self.last_raw_calls.append(raw_call)
            if self._debug_log_path:
                with open(self._debug_log_path, "a") as f:
                    f.write(json.dumps(raw_call) + "\n")
            return {
                "score": score,
                "reasoning": reasoning,
                "criterion": criterion_name,
                "perspective": perspective["name"],
            }
        except Exception as exc:
            self.logger.error(
                f"Judge error ({perspective['name']}/{criterion_name}): {exc}"
            )
            return {
                "score": 0.0,
                "reasoning": f"Error during evaluation: {exc}",
                "criterion": criterion_name,
                "perspective": perspective["name"],
            }

    def _create_judge_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
        perspective: Dict[str, str],
    ) -> str:
        """Build the user prompt for the judge."""
        prompt = (
            f"Evaluate the following research-assistant response for the "
            f"criterion: **{criterion_name}**.\n\n"
            f"Criterion description: {description}\n\n"
            f"{perspective.get('rubric_hint', '')}\n\n"
            f"User query:\n{query}\n\n"
            f"System response:\n{response}\n"
        )

        if sources:
            prompt += f"\nSources used (count={len(sources)}):\n"
            for i, s in enumerate(sources[:5], 1):
                if isinstance(s, dict):
                    title = s.get("title") or s.get("url") or str(s)[:80]
                    prompt += f"  {i}. {title}\n"
                else:
                    prompt += f"  {i}. {str(s)[:120]}\n"

        if ground_truth:
            prompt += f"\nReference / expected answer:\n{ground_truth}\n"

        prompt += (
            "\nReturn STRICTLY this JSON object (no commentary, no markdown fence):\n"
            "{\n"
            '  "score": <float between 0.0 and 1.0>,\n'
            '  "reasoning": "<one to three sentences>"\n'
            "}\n"
        )
        return prompt

    async def _call_judge_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the judge LLM via Groq or OpenAI-compatible client."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self._groq_client is not None:
            completion = self._groq_client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.content

        if self._openai_client is not None:
            completion = self._openai_client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.content

        raise ValueError(
            "No LLM client available for the judge. Set GROQ_API_KEY or "
            "OPENAI_API_KEY (+ OPENAI_BASE_URL for vLLM)."
        )

    @staticmethod
    def _parse_judgment(judgment: str) -> Tuple[float, str]:
        """Extract a numeric score and reasoning from the judge's response."""
        try:
            cleaned = judgment.strip()
            # Strip ```json / ``` fences if present.
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

            # Extract the first {...} JSON object even if surrounded by prose.
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                cleaned = match.group(0)

            obj = json.loads(cleaned)
            score = float(obj.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reasoning = str(obj.get("reasoning", ""))
            return score, reasoning
        except Exception as exc:
            return 0.0, f"Could not parse judge output: {exc}"


# ----------------------------------------------------------------------- demos
async def example_basic_evaluation():
    """Quick smoke test: judge a short Q/A pair."""
    import yaml
    from dotenv import load_dotenv

    load_dotenv()
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    judge = LLMJudge(config)
    result = await judge.evaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France. It is known for the Eiffel Tower.",
        sources=[],
        ground_truth="Paris",
    )
    print(f"Overall: {result['overall_score']:.3f}")
    for p, s in result["perspective_scores"].items():
        print(f"  perspective {p}: {s:.3f}")
    for c, s in result["criterion_scores"].items():
        print(f"  criterion {c}: {s['score']:.3f}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_basic_evaluation())
