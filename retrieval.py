"""
Two retrieval tools:
1. vector_search  -> similarity search over doc paragraphs (RAG)
                     Uses TF-IDF vectors for speed/simplicity in this demo.
                     Swap in sentence-transformers or OpenAI embeddings for
                     real semantic search in a production version.
2. graph_search   -> keyword lookup over knowledge graph triples
"""

import json
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


# ---------- Knowledge graph search ----------
# Maps everyday query words to the relation labels used in the graph
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


def graph_search(query: str, top_k: int = 5):
    """Return triples where subject/object matches a query word, or the
    query implies one of the known relations (via synonym mapping)."""
    query_words = set(w.lower().strip("?.,'\"") for w in query.split())

    implied_relations = {
        rel for word, rel in _RELATION_SYNONYMS.items() if word in query_words
    }

    matches = []
    for t in TRIPLES:
        subj_words = set(t["subject"].lower().split())
        obj_words = set(t["object"].lower().split())
        name_match = bool(query_words & subj_words or query_words & obj_words)
        relation_match = t["relation"] in implied_relations
        if name_match or relation_match:
            matches.append(t)
    return matches[:top_k]


if __name__ == "__main__":
    # quick manual sanity check
    print("VECTOR:", vector_search("What happened at the Battle of Yavin?"))
    print()
    print("GRAPH:", graph_search("Who is Luke's father?"))
