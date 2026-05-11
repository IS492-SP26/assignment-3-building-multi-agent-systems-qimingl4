"""
Command Line Interface
Interactive CLI for the multi-agent research system.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from typing import Dict, Any
import yaml
import logging
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator

# Load environment variables
load_dotenv()

class CLI:
    """
    Command-line interface for the research assistant.

    TODO: YOUR CODE HERE
    - Implement interactive prompt loop
    - Display agent traces clearly
    - Show citations and sources
    - Indicate safety events (blocked/sanitized)
    - Handle user commands (help, quit, clear, etc.)
    - Format output nicely
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize CLI.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self._setup_logging()

        # Initialize AutoGen orchestrator
        try:
            self.orchestrator = AutoGenOrchestrator(self.config)
            self.logger = logging.getLogger("cli")
            self.logger.info("AutoGen orchestrator initialized successfully")
        except Exception as e:
            self.logger = logging.getLogger("cli")
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise

        self.running = True
        self.query_count = 0

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_format = log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )

    async def run(self):
        """
        Main CLI loop.

        TODO: YOUR CODE HERE
        - Implement interactive loop
        - Handle user input
        - Process queries through orchestrator
        - Display results
        - Handle errors gracefully
        """
        self._print_welcome()

        while self.running:
            try:
                # Get user input
                query = input("\nEnter your research query (or 'help' for commands): ").strip()

                if not query:
                    continue

                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    self._print_goodbye()
                    break
                elif query.lower() == 'help':
                    self._print_help()
                    continue
                elif query.lower() == 'clear':
                    self._clear_screen()
                    continue
                elif query.lower() == 'stats':
                    self._print_stats()
                    continue

                # Process query
                print("\n" + "=" * 70)
                print("Processing your query...")
                print("=" * 70)
                
                try:
                    # Process through orchestrator (synchronous call, not async)
                    result = self.orchestrator.process_query(query)
                    self.query_count += 1
                    
                    # Display result
                    self._display_result(result)
                    
                except Exception as e:
                    print(f"\nError processing query: {e}")
                    logging.exception("Error processing query")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                self._print_goodbye()
                break
            except Exception as e:
                print(f"\nError: {e}")
                logging.exception("Error in CLI loop")

    def _print_welcome(self):
        """Print welcome message."""
        print("=" * 70)
        print(f"  {self.config['system']['name']}")
        print(f"  Topic: {self.config['system']['topic']}")
        print("=" * 70)
        print("\nWelcome! Ask me anything about your research topic.")
        print("Type 'help' for available commands, or 'quit' to exit.\n")

    def _print_help(self):
        """Print help message."""
        print("\nAvailable commands:")
        print("  help    - Show this help message")
        print("  clear   - Clear the screen")
        print("  stats   - Show system statistics")
        print("  quit    - Exit the application")
        print("\nOr enter a research query to get started!")

    def _print_goodbye(self):
        """Print goodbye message."""
        print("\nThank you for using the Multi-Agent Research Assistant!")
        print("Goodbye!\n")

    def _clear_screen(self):
        """Clear the terminal screen."""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')

    def _print_stats(self):
        """Print system statistics."""
        print("\nSystem Statistics:")
        print(f"  Queries processed: {self.query_count}")
        print(f"  System: {self.config.get('system', {}).get('name', 'Unknown')}")
        print(f"  Topic: {self.config.get('system', {}).get('topic', 'Unknown')}")
        print(f"  Model: {self.config.get('models', {}).get('default', {}).get('name', 'Unknown')}")

    def _display_result(self, result: Dict[str, Any]):
        """Display query result with formatting."""
        metadata = result.get("metadata", {})
        safety = metadata.get("safety", {}) or {}

        # If the input was blocked, surface that prominently and stop.
        input_check = safety.get("input") or {}
        if input_check.get("blocked"):
            print("\n" + "=" * 70)
            print("SAFETY: INPUT BLOCKED")
            print("=" * 70)
            print(f"\n{input_check.get('message') or result.get('response')}\n")
            self._print_violations(input_check.get("violations", []))
            return

        print("\n" + "=" * 70)
        print("RESPONSE")
        print("=" * 70)

        # Check for errors
        if "error" in result:
            print(f"\nError: {result['error']}")
            return

        # If the output was refused/sanitized, label it.
        output_check = safety.get("output") or {}
        action = output_check.get("action", "allow")
        if action == "refuse":
            print("\nSAFETY: OUTPUT REFUSED")
        elif action == "sanitize":
            print("\nSAFETY: OUTPUT SANITIZED (PII / unsafe spans redacted)")

        # Display response
        response = result.get("response", "")
        print(f"\n{response}\n")

        # Extract and display citations from conversation
        citations = self._extract_citations(result)
        if citations:
            print("\n" + "-" * 70)
            print("📚 CITATIONS")
            print("-" * 70)
            for i, citation in enumerate(citations, 1):
                print(f"[{i}] {citation}")

        # Display metadata
        if metadata:
            print("\n" + "-" * 70)
            print("METADATA")
            print("-" * 70)
            print(f"  - Messages exchanged: {metadata.get('num_messages', 0)}")
            print(f"  - Sources gathered: {metadata.get('num_sources', 0)}")
            print(f"  - Agents involved: {', '.join(metadata.get('agents_involved', []))}")

            # Safety summary from the orchestrator.
            stats = safety.get("stats", {}) if safety else {}
            if stats and stats.get("total_events"):
                print(
                    f"  - Safety events: {stats.get('total_events', 0)} "
                    f"(input={stats.get('by_type', {}).get('input', 0)}, "
                    f"output={stats.get('by_type', {}).get('output', 0)})"
                )
                cats = stats.get("by_category", {})
                if cats:
                    pretty = ", ".join(f"{k}={v}" for k, v in cats.items())
                    print(f"  - Triggered categories: {pretty}")

            output_violations = (output_check or {}).get("violations", [])
            if output_violations:
                print("\nOutput safety violations:")
                self._print_violations(output_violations)

        # Display conversation summary if verbose mode
        if self._should_show_traces():
            self._display_conversation_summary(result.get("conversation_history", []))

        print("=" * 70 + "\n")
    
    def _print_violations(self, violations: list):
        """Pretty-print a list of safety violations."""
        for v in violations:
            print(
                f"    [{v.get('severity', 'low').upper()}] "
                f"{v.get('category', 'unknown')}: {v.get('reason', '')}"
            )

    def _extract_citations(self, result: Dict[str, Any]) -> list:
        """Extract citations/URLs from conversation history."""
        citations = []
        
        for msg in result.get("conversation_history", []):
            content = msg.get("content", "")
            
            # Find URLs in content
            import re
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
            
            for url in urls:
                if url not in citations:
                    citations.append(url)
        
        return citations[:10]  # Limit to top 10

    def _should_show_traces(self) -> bool:
        """Check if agent traces should be displayed."""
        # Check config for verbose mode
        return self.config.get("ui", {}).get("verbose", False)

    def _display_conversation_summary(self, conversation_history: list):
        """Display a summary of the agent conversation."""
        if not conversation_history:
            return
            
        print("\n" + "-" * 70)
        print("🔍 CONVERSATION SUMMARY")
        print("-" * 70)
        
        for i, msg in enumerate(conversation_history, 1):
            agent = msg.get("source", "Unknown")
            content = msg.get("content", "")
            
            # Truncate long content
            preview = content[:150] + "..." if len(content) > 150 else content
            preview = preview.replace("\n", " ")
            
            print(f"\n{i}. {agent}:")
            print(f"   {preview}")


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant CLI"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Run CLI
    cli = CLI(config_path=args.config)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
