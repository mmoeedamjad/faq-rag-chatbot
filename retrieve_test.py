import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('data/faqs.json', 'r') as f:
    faqs = json.load(f)

# Reload cached embeddings (or recompute - both fine)
faq_embeddings = np.load('data/faq_embeddings.npy')


def cosine_similarity(query_vec, faq_matrix):
    # Normalize each vector to unit length, then dot product = cosine similarity
    query_norm = query_vec / np.linalg.norm(query_vec)
    faq_norms = faq_matrix / np.linalg.norm(faq_matrix, axis=1, keepdims=True)
    return faq_norms @ query_norm  # shape: (num_faqs,)


def retrieve(query, top_k=3):
    query_vec = model.encode(query)
    scores = cosine_similarity(query_vec, faq_embeddings)

    # Get indices of top_k highest scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "question": faqs[idx]["question"],
            "answer": faqs[idx]["answer"],
            "score": float(scores[idx])
        })
    return results


# Test it with a few queries
test_queries = [
    "how can I change my password",
    "what does it cost to use this",
    "do you have an app for my phone",   # try something NOT in your FAQs
]

for q in test_queries:
    print(f"\nQuery: {q}")
    results = retrieve(q)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['question']}")
