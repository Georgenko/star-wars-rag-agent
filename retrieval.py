"""
Two retrieval tools:
1. vector_search  -> similarity search over doc paragraphs (RAG)
                     Uses TF-IDF vectors for speed/simplicity in this demo.
                     Swap in sentence-transformers or OpenAI embeddings for
                     real semantic search in a production version.
2. graph_search   -> keyword lookup over knowledge graph triples
"""

import json
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load data ----------
with open("data/docs.json") as f:
    DOCS = json.load(f)

with open("data/triples.json") as f:
    TRIPLES = json.load(f)

# ---------- Vector RAG setup ----------
_doc_texts = [d["text"] for d in DOCS]
_vectorizer = TfidfVectorizer(stop_words="english")
_doc_matrix = _vectorizer.fit_transform(_doc_texts)


def vector_search(query: str, top_k: int = 2):
    """Return the top_k most similar docs to the query (TF-IDF cosine similarity)."""
    query_vec = _vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _doc_matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({
            "title": DOCS[i]["title"],
            "text": DOCS[i]["text"],
            "score": float(sims[i])
        })
    return results


# ---------- Knowledge graph setup ----------
# Build a real directed graph: nodes = characters/factions, edges = relations
G = nx.DiGraph()
for t in TRIPLES:
    G.add_edge(t["subject"], t["object"], relation=t["relation"])

_ALL_NODES = list(G.nodes())

_RELATION_SYNONYMS = {
    "father": "child_of", "mother": "child_of", "parent": "child_of",
    "child": "child_of", "son": "child_of", "daughter": "child_of",
    "sibling": "sibling_of", "brother": "sibling_of", "sister": "sibling_of",
    "trained": "trained", "teacher": "trained", "mentor": "trained",
    "leads": "leads", "leader": "leads", "commands": "leads",
    "married": "married_to", "husband": "married_to", "wife": "married_to",
    "enemy": "enemy_of", "rival": "enemy_of", "opposes": "enemy_of",
    "killed": "killed", "kill": "killed",
    "member": "member_of", "belongs": "member_of", "part": "member_of",
}


def _clean_word(w: str) -> str:
    w = w.lower().strip("?.,'\"")
    return w[:-2] if w.endswith("'s") else w


def _find_entities_in_query(query: str):
    """Which known graph nodes (character/faction names) are mentioned?"""
    query_words = {_clean_word(w) for w in query.split()}
    found = []
    for node in _ALL_NODES:
        node_words = {w.lower() for w in node.split()}
        if node_words & query_words:
            found.append(node)
    return found


def graph_search(query: str, top_k: int = 5):
    """Traverse the knowledge graph:
    - find which entities are mentioned in the query
    - if 2+ entities mentioned, look for a path connecting them (multi-hop)
    - otherwise, return that entity's direct edges (1-hop neighbors),
      optionally filtered by an implied relation (e.g. "father" -> child_of)
    """
    entities = _find_entities_in_query(query)
    query_words = {_clean_word(w) for w in query.split()}
    implied_relations = {
        rel for word, rel in _RELATION_SYNONYMS.items()
        if word in query_words or word + "s" in query_words # tolerate plurals as well
    }

    matches = []

    # Multi-hop: if two known entities are both mentioned, find a path between them
    if len(entities) >= 2:
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue
                try:
                    path = nx.shortest_path(G.to_undirected(), entities[i], entities[j])
                    for a, b in zip(path, path[1:]):
                        if G.has_edge(a, b):
                            relation = G.get_edge_data(a, b)["relation"]
                            matches.append({"subject": a, "relation": relation, "object": b})
                        else:
                            relation = G.get_edge_data(b, a)["relation"]
                            matches.append({"subject": b, "relation": relation, "object": a})
                except nx.NetworkXNoPath:
                    continue

    # 1-hop: direct edges (outgoing + incoming) for each mentioned entity
    if not matches:
        for entity in entities:
            for _, obj, data in G.out_edges(entity, data=True):
                if not implied_relations or data["relation"] in implied_relations:
                    matches.append({"subject": entity, "relation": data["relation"], "object": obj})
            for subj, _, data in G.in_edges(entity, data=True):
                if not implied_relations or data["relation"] in implied_relations:
                    matches.append({"subject": subj, "relation": data["relation"], "object": entity})

    if not matches and implied_relations:
        for u, v, data in G.edges(data=True):
            if data["relation"] in implied_relations:
                matches.append({"subject": u, "relation": data["relation"], "object": v})

    return matches[:top_k]


if __name__ == "__main__":
    # quick manual sanity check
    print("VECTOR:", vector_search("What happened at the Battle of Yavin?"))
    print()
    print("GRAPH:", graph_search("Who is Luke's father?"))
