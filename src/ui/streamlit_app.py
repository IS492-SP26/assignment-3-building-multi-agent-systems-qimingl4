"""
Streamlit Web Interface
Web UI for the multi-agent research system.

Run with: streamlit run src/ui/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import asyncio
import json
import yaml
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator
from src.evaluation.judge import LLMJudge

# Load environment variables
load_dotenv()


def load_config():
    """Load configuration file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'history' not in st.session_state:
        st.session_state.history = []

    if 'orchestrator' not in st.session_state:
        config = load_config()
        # Initialize AutoGen orchestrator
        try:
            st.session_state.orchestrator = AutoGenOrchestrator(config)
        except Exception as e:
            st.error(f"Failed to initialize orchestrator: {e}")
            st.session_state.orchestrator = None

    if 'show_traces' not in st.session_state:
        st.session_state.show_traces = False

    if 'show_safety_log' not in st.session_state:
        st.session_state.show_safety_log = False

    if 'run_judge' not in st.session_state:
        st.session_state.run_judge = True

    if 'judge' not in st.session_state:
        config = load_config()
        try:
            st.session_state.judge = LLMJudge(config)
        except Exception as e:
            st.session_state.judge = None
            st.warning(f"Judge unavailable: {e}")

async def process_query(query: str) -> Dict[str, Any]:
    """
    Process a query through the orchestrator.
    
    Args:
        query: Research query to process
        
    Returns:
        Result dictionary with response, citations, and metadata
    """
    orchestrator = st.session_state.orchestrator
    
    if orchestrator is None:
        return {
            "query": query,
            "error": "Orchestrator not initialized",
            "response": "Error: System not properly initialized. Please check your configuration.",
            "citations": [],
            "metadata": {}
        }
    
    try:
        # Process query through AutoGen orchestrator
        result = orchestrator.process_query(query)

        # Check for errors
        if "error" in result:
            return result

        # Optional LLM-as-a-Judge scoring. Skip when the orchestrator refused
        # or blocked the request — there is nothing meaningful to judge.
        judge_result = None
        result_meta = result.get("metadata", {}) or {}
        was_refused = result_meta.get("refused") or (
            (result_meta.get("safety", {}) or {}).get("input", {}) or {}
        ).get("blocked")
        if (
            st.session_state.get("run_judge")
            and st.session_state.get("judge")
            and not was_refused
        ):
            try:
                judge_result = await st.session_state.judge.evaluate(
                    query=query,
                    response=result.get("response", ""),
                    sources=result_meta.get("research_findings", []),
                )
            except Exception as exc:
                judge_result = {"error": str(exc)}
        
        # Extract citations from conversation history
        citations = extract_citations(result)
        
        # Extract agent traces for display
        agent_traces = extract_agent_traces(result)
        
        # Format metadata
        metadata = result.get("metadata", {})
        metadata["agent_traces"] = agent_traces
        metadata["citations"] = citations
        metadata["critique_score"] = calculate_quality_score(result)
        # safety information is already nested under metadata["safety"] by
        # the orchestrator; surface it for the UI helpers below.
        metadata["safety_events"] = (metadata.get("safety", {}) or {}).get(
            "events", []
        )
        metadata["judge"] = judge_result

        return {
            "query": query,
            "response": result.get("response", ""),
            "citations": citations,
            "metadata": metadata
        }
        
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "response": f"An error occurred: {str(e)}",
            "citations": [],
            "metadata": {"error": True}
        }


def extract_citations(result: Dict[str, Any]) -> list:
    """Extract citations from research result."""
    citations = []
    
    # Look through conversation history for citations
    for msg in result.get("conversation_history", []):
        content = msg.get("content", "")
        # Tool-call entries are lists of dicts; flatten to a string.
        if not isinstance(content, str):
            content = str(content)

        # Find URLs in content
        import re
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)

        # Find citation patterns like [Source: Title]
        citation_patterns = re.findall(r'\[Source: ([^\]]+)\]', content)
        
        for url in urls:
            if url not in citations:
                citations.append(url)
        
        for citation in citation_patterns:
            if citation not in citations:
                citations.append(citation)
    
    return citations[:10]  # Limit to top 10


def extract_agent_traces(result: Dict[str, Any]) -> Dict[str, list]:
    """Extract agent execution traces from conversation history."""
    traces = {}
    
    for msg in result.get("conversation_history", []):
        agent = msg.get("source", "Unknown")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        content = content[:200]  # First 200 chars
        
        if agent not in traces:
            traces[agent] = []
        
        traces[agent].append({
            "action_type": "message",
            "details": content
        })
    
    return traces


def calculate_quality_score(result: Dict[str, Any]) -> float:
    """Calculate a quality score based on various factors."""
    score = 5.0  # Base score
    
    metadata = result.get("metadata", {})
    
    # Add points for sources
    num_sources = metadata.get("num_sources", 0)
    score += min(num_sources * 0.5, 2.0)
    
    # Add points for critique
    if metadata.get("critique"):
        score += 1.0
    
    # Add points for conversation length (indicates thorough discussion)
    num_messages = metadata.get("num_messages", 0)
    score += min(num_messages * 0.1, 2.0)
    
    return min(score, 10.0)  # Cap at 10


def display_response(result: Dict[str, Any]):
    """
    Display query response.

    TODO: YOUR CODE HERE
    - Format response nicely
    - Show citations with links
    - Display sources
    - Show safety events if any
    """
    # Check for errors
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return

    metadata = result.get("metadata", {})
    safety = metadata.get("safety", {}) or {}
    input_check = safety.get("input") or {}
    output_check = safety.get("output") or {}

    # Strong banner for blocked input.
    if input_check.get("blocked"):
        st.error(
            "Input blocked by a safety policy. "
            f"Triggered categories: {', '.join({v.get('category','?') for v in input_check.get('violations', [])})}"
        )
        st.markdown(input_check.get("message") or result.get("response", ""))
        with st.expander("Input violation details", expanded=True):
            for v in input_check.get("violations", []):
                st.warning(
                    f"[{v.get('severity','low').upper()}] {v.get('category','?')}: {v.get('reason','')}"
                )
        return

    # Banner if output was refused or sanitized.
    action = output_check.get("action", "allow")
    if action == "refuse":
        st.error("Output refused by the safety policy.")
    elif action == "sanitize":
        st.warning("Output was sanitized (PII or unsafe spans redacted).")

    # Display response
    st.markdown("### Response")
    response = result.get("response", "")
    st.markdown(response)

    # Display citations
    citations = result.get("citations", [])
    if citations:
        with st.expander("📚 Citations", expanded=False):
            for i, citation in enumerate(citations, 1):
                st.markdown(f"**[{i}]** {citation}")

    # Display metadata
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sources Used", metadata.get("num_sources", 0))
    with col2:
        score = metadata.get("critique_score", 0)
        st.metric("Quality Score", f"{score:.2f}")

    # Safety events
    safety_events = metadata.get("safety_events", [])
    if safety_events:
        with st.expander(f"Safety Events ({len(safety_events)})", expanded=True):
            for event in safety_events:
                event_type = event.get("type", "unknown")
                event_action = event.get("action", "allow")
                violations = event.get("violations", [])
                st.warning(
                    f"{event_type.upper()} - action={event_action.upper()} - "
                    f"{len(violations)} violation(s)"
                )
                for violation in violations:
                    st.text(
                        f"  [{violation.get('severity','low').upper()}] "
                        f"{violation.get('category','?')}: "
                        f"{violation.get('reason','')}"
                    )

    # LLM-as-a-Judge scores
    judge_result = metadata.get("judge")
    if judge_result and not judge_result.get("error"):
        display_judge_result(judge_result)
    elif judge_result and judge_result.get("error"):
        st.warning(f"Judge error: {judge_result['error']}")

    # Agent traces
    if st.session_state.show_traces:
        agent_traces = metadata.get("agent_traces", {})
        if agent_traces:
            display_agent_traces(agent_traces)


def display_judge_result(judge_result: Dict[str, Any]):
    """Render the LLM-as-a-Judge evaluation result."""
    st.markdown("### LLM-as-a-Judge Evaluation")
    overall = judge_result.get("overall_score", 0.0)
    st.metric("Overall Weighted Score", f"{overall:.3f}")

    perspective_scores = judge_result.get("perspective_scores", {}) or {}
    if perspective_scores:
        cols = st.columns(len(perspective_scores))
        for col, (name, score) in zip(cols, perspective_scores.items()):
            col.metric(name.replace("_", " ").title(), f"{score:.3f}")

    criterion_scores = judge_result.get("criterion_scores", {}) or {}
    if criterion_scores:
        rows = []
        for crit, data in criterion_scores.items():
            perspectives = data.get("perspectives", {})
            row = {"criterion": crit, "average": round(data.get("score", 0.0), 3)}
            for p_name, p_data in perspectives.items():
                row[p_name] = round(p_data.get("score", 0.0), 3)
            rows.append(row)
        st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.expander("Judge reasoning (per perspective × criterion)", expanded=False):
            for crit, data in criterion_scores.items():
                st.markdown(f"**{crit}**  —  average {data.get('score', 0.0):.3f}")
                for p_name, p_data in data.get("perspectives", {}).items():
                    st.markdown(f"- *{p_name}* `{p_data.get('score', 0.0):.2f}` — {p_data.get('reasoning','')}")


def display_agent_traces(traces: Dict[str, Any]):
    """
    Display agent execution traces.

    TODO: YOUR CODE HERE
    - Format traces nicely
    - Show agent workflow
    - Display timing information
    """
    with st.expander("🔍 Agent Traces", expanded=False):
        for agent_name, actions in traces.items():
            st.markdown(f"**{agent_name.upper()}**")
            for action in actions:
                action_type = action.get("action_type", "unknown")
                details = action.get("details", {})
                st.text(f"  → {action_type}: {details}")


def display_sidebar():
    """Display sidebar with settings and statistics."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # Show traces toggle
        st.session_state.show_traces = st.checkbox(
            "Show Agent Traces",
            value=st.session_state.show_traces
        )

        # Show safety log toggle
        st.session_state.show_safety_log = st.checkbox(
            "Show Safety Log",
            value=st.session_state.show_safety_log
        )

        # Run LLM-as-a-Judge after each query
        st.session_state.run_judge = st.checkbox(
            "Run LLM-as-a-Judge",
            value=st.session_state.run_judge,
            help="Score the response with two independent rubrics (supportive + strict).",
        )

        st.divider()

        st.title("📊 Statistics")

        st.metric("Total Queries", len(st.session_state.history))

        # Sum safety events across queries.
        total_events = 0
        for item in st.session_state.history:
            md = (item.get("result", {}) or {}).get("metadata", {}) or {}
            total_events += len(md.get("safety_events", []) or [])
        st.metric("Safety Events", total_events)

        st.divider()

        # Clear history button
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

        # About section
        st.divider()
        st.markdown("### About")
        config = load_config()
        system_name = config.get("system", {}).get("name", "Research Assistant")
        topic = config.get("system", {}).get("topic", "General")
        st.markdown(f"**System:** {system_name}")
        st.markdown(f"**Topic:** {topic}")


def display_history():
    """Display query history."""
    if not st.session_state.history:
        return

    with st.expander("📜 Query History", expanded=False):
        for i, item in enumerate(reversed(st.session_state.history), 1):
            timestamp = item.get("timestamp", "")
            query = item.get("query", "")
            st.markdown(f"**{i}.** [{timestamp}] {query}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Multi-Agent Research Assistant",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session_state()

    # Header
    st.title("🤖 Multi-Agent Research Assistant")
    st.markdown("Ask me anything about your research topic!")

    # Sidebar
    display_sidebar()

    # Main area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input
        query = st.text_area(
            "Enter your research query:",
            height=100,
            placeholder="e.g., What are the latest developments in explainable AI for novice users?"
        )

        # "Load demo session" — display a saved live run without re-running it.
        # Useful for graders / screenshots when API keys are unavailable.
        demo_paths = sorted(Path("outputs").glob("session_*.json"), reverse=True)
        if demo_paths:
            if st.button("📂 Load most recent demo session", use_container_width=True):
                try:
                    with open(demo_paths[0]) as f:
                        saved = json.load(f)
                    metadata = saved.get("metadata", {}) or {}
                    citations = extract_citations(saved)
                    metadata["agent_traces"] = extract_agent_traces(saved)
                    metadata["citations"] = citations
                    metadata["critique_score"] = calculate_quality_score(saved)
                    metadata["safety_events"] = (
                        (metadata.get("safety") or {}).get("events", [])
                    )
                    # If the session was exported with a judge field, surface it.
                    metadata["judge"] = saved.get("judge")
                    loaded_result = {
                        "query": saved.get("query", ""),
                        "response": saved.get("response", ""),
                        "citations": citations,
                        "metadata": metadata,
                    }
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": loaded_result["query"],
                        "result": loaded_result,
                    })
                    st.session_state.preview_result = loaded_result
                    st.success(f"Loaded {demo_paths[0].name}")
                except Exception as exc:
                    st.error(f"Could not load demo session: {exc}")

        # Submit button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Processing your query..."):
                    # Process query
                    result = asyncio.run(process_query(query))

                    # Add to history
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query,
                        "result": result
                    })

                    # Display result
                    st.divider()
                    display_response(result)
            else:
                st.warning("Please enter a query.")

        # If a session was loaded via "Load demo session", display it here.
        if st.session_state.get("preview_result"):
            st.divider()
            display_response(st.session_state.preview_result)

        # History
        display_history()

    with col2:
        st.markdown("### 💡 Example Queries")
        examples = [
            "What are the key principles of user-centered design?",
            "Explain recent advances in AR usability research",
            "Compare different approaches to AI transparency",
            "What are ethical considerations in AI for education?",
        ]

        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.example_query = example
                st.rerun()

        # If example was clicked, populate the text area
        if 'example_query' in st.session_state:
            st.info(f"Example query selected: {st.session_state.example_query}")
            del st.session_state.example_query

        st.divider()

        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        1. **Planner** breaks down your query
        2. **Researcher** gathers evidence
        3. **Writer** synthesizes findings
        4. **Critic** verifies quality
        5. **Safety** checks ensure appropriate content
        """)

    # Safety log (if enabled)
    if st.session_state.show_safety_log:
        st.divider()
        st.markdown("### Safety Event Log")
        all_events = []
        for item in st.session_state.history:
            md = (item.get("result", {}) or {}).get("metadata", {}) or {}
            all_events.extend(md.get("safety_events", []) or [])
        if not all_events:
            st.info("No safety events recorded.")
        else:
            for event in all_events[-50:]:
                st.warning(
                    f"[{event.get('timestamp','')}] {event.get('type','?').upper()} "
                    f"- action={event.get('action','?').upper()} - "
                    f"{len(event.get('violations', []))} violation(s)"
                )
                for v in event.get("violations", []):
                    st.text(
                        f"  [{v.get('severity','low').upper()}] "
                        f"{v.get('category','?')}: {v.get('reason','')}"
                    )


if __name__ == "__main__":
    main()
