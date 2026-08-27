"""CLI to test the Star Wars RAG + Knowledge Graph agent."""

from agent import build_agent

TEST_QUESTIONS = [
    "Who is Luke Skywalker's father?",          # graph
    "Who trained Luke Skywalker?",               # graph
    "What happened at the Battle of Yavin?",     # vector
    "What is the Force?",                        # vector
    "How did Darth Vader die and who is he related to?",  # both
]


def run_tests(agent):
    for q in TEST_QUESTIONS:
        print(f"\nQ: {q}")
        result = agent.invoke({"query": q})
        print(f"A: {result['answer']}")


def run_interactive(agent):
    print("Star Wars Agent (RAG + Knowledge Graph). Type 'quit' to exit.\n")
    while True:
        q = input("Ask: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"query": q})
        print(f"\n{result['answer']}\n")


if __name__ == "__main__":
    agent = build_agent()
    print("=== Running test questions ===")
    run_tests(agent)
    print("\n=== Interactive mode ===")
    run_interactive(agent)
