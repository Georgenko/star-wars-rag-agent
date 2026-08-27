"""
LangGraph agent: routes a question to either vector RAG or knowledge-graph
search (or both), then generates a grounded answer.

Flow:
    query -> [router] -> [run_vector | run_graph | run_both] -> [generate] -> answer
"""

import os
from typing import TypedDict, Literal, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from retrieval import vector_search, graph_search

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
llm = ChatGroq(model=GROQ_MODEL, temperature=0)


# ---------- State ----------
class AgentState(TypedDict):
    query: str
    route: Optional[Literal["vector", "graph", "both"]]
    context: Optional[str]
    source: Optional[str]
    answer: Optional[str]
    tried_both: Optional[bool]
    needs_retry: Optional[bool]


# ---------- Nodes ----------
def route_query(state: AgentState) -> AgentState:
    """LLM decides whether the question needs descriptive lore (vector RAG),
    relational facts (knowledge graph), or both."""
    prompt = f"""You are a router deciding how to answer a Star Wars question.

Two tools are available:
- "vector": search descriptive lore/story text (good for "what happened",
  "explain", "describe" questions)
- "graph": search a knowledge graph of relationships like family ties,
  memberships, alliances (good for "who is X's father", "who trained X",
  "who leads X" questions)
- "both": if the question needs both a relationship and descriptive context

Question: "{state['query']}"

Reply with exactly one word: vector, graph, or both."""

    response = llm.invoke(prompt).content.strip().lower()
    route = "vector"
    if "both" in response:
        route = "both"
    elif "graph" in response:
        route = "graph"
    print(f"  [router decided: {route}]")
    return {**state, "route": route}


def run_vector(state: AgentState) -> AgentState:
    results = vector_search(state["query"])
    context = "\n\n".join(f"[{r['title']}]: {r['text']}" for r in results)
    return {**state, "context": context, "source": "vector (docs)"}


def run_graph(state: AgentState) -> AgentState:
    results = graph_search(state["query"])
    if results:
        context = "\n".join(
            f"{t['subject']} --{t['relation']}--> {t['object']}" for t in results
        )
    else:
        context = "No matching relationships found in the knowledge graph."
    return {**state, "context": context, "source": "graph (knowledge graph)"}


def run_both(state: AgentState) -> AgentState:
    vec = run_vector(state)
    graph = run_graph(state)
    context = f"VECTOR CONTEXT:\n{vec['context']}\n\nGRAPH CONTEXT:\n{graph['context']}"
    return {**state, "context": context, "source": "vector + graph"}

def grade_context(state: AgentState) -> AgentState:
    """Check if retrieved context actually answers the question."""
    prompt = f"""Does this context contain enough info to answer the question?
Context:
{state['context']}

Question: {state['query']}

Reply with exactly one word: yes or no."""
    verdict = llm.invoke(prompt).content.strip().lower()
    ok = "yes" in verdict
    print(f"  [grader: {'sufficient' if ok else 'insufficient'}]")
    return {**state, "needs_retry": not ok}


def retry_other(state: AgentState) -> AgentState:
    """First attempt failed grading - try the other source."""
    other = "graph" if state["route"] == "vector" else "vector"
    print(f"  [retrying with: {other}]")
    fn = run_graph if other == "graph" else run_vector
    result = fn(state)
    return {**result, "tried_both": True}



def generate(state: AgentState) -> AgentState:
    prompt = f"""Answer the question using ONLY the context below. Be concise (2-3 sentences).

Context:
{state['context']}

Question: {state['query']}

Answer:"""
    response = llm.invoke(prompt).content.strip()
    response += f"\n\nSource: {state['source']}"
    return {**state, "answer": response}


# ---------- Graph wiring ----------
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("route", route_query)
    graph.add_node("run_vector", run_vector)
    graph.add_node("run_graph", run_graph)
    graph.add_node("run_both", run_both)
    graph.add_node("grade_context", grade_context)
    graph.add_node("retry_other", retry_other)
    graph.add_node("generate", generate)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"vector": "run_vector", "graph": "run_graph", "both": "run_both"},
    )

    graph.add_edge("run_vector", "grade_context")
    graph.add_edge("run_graph", "grade_context")
    graph.add_edge("run_both", "generate")

    graph.add_conditional_edges(
        "grade_context",
        lambda state: "retry" if (state.get("needs_retry") and not state.get("tried_both")) else "generate",
        {"retry": "retry_other", "generate": "generate"},
    )

    graph.add_edge("retry_other", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


if __name__ == "__main__":
    agent = build_agent()
    result = agent.invoke({"query": "Who is Luke Skywalker's father?"})
    print(result["answer"])
