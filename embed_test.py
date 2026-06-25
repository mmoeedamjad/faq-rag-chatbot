import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model (downloads once, then cached locally)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAQ data
with open('data/faqs.json', 'r') as f:
    faqs = json.load(f)

print(f"Loaded {len(faqs)} FAQs")

# Embed all questions (you could also embed question+answer combined - try both later)
questions = [faq['question'] for faq in faqs]
embeddings = model.encode(questions)

print(f"Embeddings shape: {embeddings.shape}")  # should be (num_faqs, 384)
print(f"First embedding (first 10 dims): {embeddings[0][:10]}")

# Save embeddings so we don't recompute every time during dev (optional but handy)
np.save('data/faq_embeddings.npy', embeddings)
print("Saved embeddings to data/faq_embeddings.npy")
