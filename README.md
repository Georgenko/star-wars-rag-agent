# Star Wars RAG + Knowledge Graph Agent
 
A small **agentic AI** system built with **LangGraph** that answers Star Wars questions by dynamically routing between **vector-based RAG (Retrieval-Augmented Generation)** and a **knowledge graph**, with a self-correcting **grader/retry loop** to reduce hallucination and improve grounding.
 
Built as a hands-on exploration of agentic orchestration, retrieval pipelines, and hybrid retrieval architectures.
 
## What it does
 
Instead of relying on a single fixed retrieval method, the agent **decides at query time** which knowledge source is best suited to answer the question:
 
- **Vector search (RAG)** - semantic/lexical similarity search over unstructured lore text, ideal for descriptive "what happened" or "explain" questions.
- **Knowledge graph search** - traversal over a directed graph of entity relationships; handles direct lookups ("who is X's father") and multi-hop connections ("how is X related to Y").
- **Both** - when a question needs relational facts *and* narrative context.

                                
If a single source (**vector** or **graph**) is chosen, a **grader node** then evaluates whether the retrieved context is actually sufficient to answer the question. If not, the agent **automatically retries with the alternate source** before generating a final, grounded answer with source attribution.
 
## Architecture
 
```
         ┌─────────┐
  query →│ Router  │   LLM decides: vector / graph / both
         └────┬────┘
   ┌──────────┼────────────┐
   ▼          ▼            ▼
run_vector  run_graph  run_both
   │          │            │
   └────┬─────┘            │
        ▼                  │
  ┌─────────────┐          │
  │Grade Context│          │
  └──────┬──────┘          │
    insufficient?          │
         ▼                 │
  ┌─────────────┐          │
  │ Retry Other │          │
  │   Source    │          │
  └──────┬──────┘          │
         └────────┬────────┘
                  ▼
             ┌──────────┐
             │ Generate │  Grounded answer + source citation
             └──────────┘
```


## Tech stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** - stateful, graph-based agent orchestration with conditional edges
- **[Groq](https://groq.com/)** (via `langchain-groq`) - fast LLM inference for routing, grading, and generation
- **scikit-learn (TF-IDF)** - lightweight vector similarity search, standing in for a full embedding model
- **[NetworkX](https://networkx.org/)** - real directed graph with multi-hop path traversal between entities

## Running it
 
```bash
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY
python main.py
```
 
Runs a batch of test questions first, then drops into an interactive prompt.
 
## Example
 
```
Ask: Who is Yoda?
  [router decided: vector]
  [grader: insufficient]
  [retrying with: graph]
Yoda is a member of the Jedi Order. He trained Luke Skywalker.
Source: graph (knowledge graph)
```
