"""
Main Entry Point
Run the multi-agent research system.

Usage:
  python main.py --mode cli           # Interactive CLI
  python main.py --mode web           # Streamlit web UI
  python main.py --mode evaluate      # Batch evaluation (LLM-as-a-Judge)
  python main.py --mode autogen       # Run the AutoGen example script
  python main.py --mode demo          # Single end-to-end query + judge

Optional flags:
  --config PATH       config file (default: config.yaml)
  --queries PATH      evaluation dataset (default: data/example_queries.json)
  --query "TEXT"      single-query override for `demo` mode
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime


def run_cli():
    """Interactive CLI."""
    from src.ui.cli import main as cli_main
    cli_main()


def run_web():
    """Launch the Streamlit web app."""
    import subprocess
    print("Starting Streamlit web interface...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


def run_autogen():
    """Run the bundled AutoGen example."""
    import subprocess
    print("Running AutoGen example...")
    subprocess.run([sys.executable, "example_autogen.py"])


async def run_evaluation(config_path: str, queries_path: str):
    """Batch evaluation with LLM-as-a-Judge."""
    import yaml
    from dotenv import load_dotenv

    from src.autogen_orchestrator import AutoGenOrchestrator
    from src.evaluation.evaluator import SystemEvaluator

    load_dotenv()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("Initializing AutoGen orchestrator...")
    orchestrator = AutoGenOrchestrator(config)

    print(f"Loading evaluation dataset: {queries_path}")
    evaluator = SystemEvaluator(config, orchestrator=orchestrator)

    report = await evaluator.evaluate_system(queries_path)
    if "error" in report:
        print(f"Evaluation error: {report['error']}")
        return

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    summary = report.get("summary", {})
    scores = report.get("scores", {})
    print(f"Total queries:       {summary.get('total_queries', 0)}")
    print(f"Successful:          {summary.get('successful', 0)}")
    print(f"Failed:              {summary.get('failed', 0)}")
    print(f"Success rate:        {summary.get('success_rate', 0.0):.2%}")
    print(f"Overall score (avg): {scores.get('overall_average', 0.0):.3f}\n")

    print("Average score by criterion:")
    for criterion, score in scores.get("by_criterion", {}).items():
        print(f"  {criterion:25s} {score:.3f}")

    print("\nAverage score by judge perspective:")
    for perspective, score in scores.get("by_perspective", {}).items():
        print(f"  {perspective:25s} {score:.3f}")

    best = report.get("best_result")
    worst = report.get("worst_result")
    if best:
        print(f"\nBest:  ({best['score']:.3f}) {best['query'][:80]}")
    if worst:
        print(f"Worst: ({worst['score']:.3f}) {worst['query'][:80]}")
    print("\nDetailed results saved under outputs/")


async def run_demo(config_path: str, query: str):
    """Single-query demo: query -> agents -> response -> judge -> export."""
    import yaml
    from dotenv import load_dotenv

    from src.autogen_orchestrator import AutoGenOrchestrator
    from src.evaluation.judge import LLMJudge

    load_dotenv()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    orchestrator = AutoGenOrchestrator(config)
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    result = orchestrator.process_query(query)
    print("\nFINAL RESPONSE:\n")
    print(result.get("response", "<no response>"))

    metadata = result.get("metadata", {})
    safety = metadata.get("safety") or {}
    input_check = safety.get("input") or {}
    output_check = safety.get("output") or {}

    if input_check.get("blocked"):
        print("\nInput was BLOCKED by safety policy.")
        return

    if "error" in result:
        print(f"\nOrchestrator error: {result['error']}")
        return

    print(f"\nMessages: {metadata.get('num_messages', 0)} | "
          f"Sources: {metadata.get('num_sources', 0)} | "
          f"Agents: {', '.join(metadata.get('agents_involved', []))}")

    if output_check.get("action") in {"refuse", "sanitize"}:
        print(
            f"Output safety action: {output_check['action']} "
            f"({len(output_check.get('violations', []))} violations)"
        )

    # Judge — also log raw prompts/responses for the assignment artefact.
    print("\nJudging response...")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_log = out_dir / f"judge_raw_{timestamp}.jsonl"
    import os
    os.environ["JUDGE_DEBUG_LOG"] = str(raw_log)
    judge = LLMJudge(config)
    judge_result = await judge.evaluate(
        query=query,
        response=result.get("response", ""),
        sources=metadata.get("research_findings", []),
    )
    print(f"Raw judge prompts/responses logged to: {raw_log}")
    print(f"\nOverall score: {judge_result['overall_score']:.3f}")
    print("Per-criterion:")
    for c, s in judge_result["criterion_scores"].items():
        print(f"  {c:25s} {s['score']:.3f}")
    print("Per-perspective:")
    for p, s in judge_result["perspective_scores"].items():
        print(f"  {p:25s} {s:.3f}")

    # Export this session.
    session_file = out_dir / f"session_{timestamp}.json"
    with open(session_file, "w") as f:
        json.dump(
            {
                "query": query,
                "response": result.get("response", ""),
                "conversation_history": result.get("conversation_history", []),
                "metadata": metadata,
                "judge": judge_result,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSession exported to: {session_file}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant")
    parser.add_argument(
        "--mode",
        choices=["cli", "web", "evaluate", "autogen", "demo"],
        default="autogen",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--queries", default="data/example_queries.json")
    parser.add_argument(
        "--query",
        default="What are the key principles of accessible user interface design?",
        help="Query to use in --mode demo",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    elif args.mode == "web":
        run_web()
    elif args.mode == "autogen":
        run_autogen()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation(args.config, args.queries))
    elif args.mode == "demo":
        asyncio.run(run_demo(args.config, args.query))


if __name__ == "__main__":
    main()
