from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Sample documents
documents = [
    "I bought a cheap phone yesterday.",
    "This smartphone is very affordable.",
    "Apple iPhone is expensive but high quality.",
    "Looking for a low-cost mobile device."
]

# 1. Keyword Search (BM25)
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

query = "cheap phone"
tokenized_query = query.lower().split()
keyword_scores = bm25.get_scores(tokenized_query)

# Normalize keyword scores (0–1 range)
keyword_scores = (keyword_scores - np.min(keyword_scores)) / (np.max(keyword_scores) - np.min(keyword_scores) + 1e-9)

# 2. Semantic Search (Embeddings)
model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

semantic_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()
# Normalize semantic scores
semantic_scores = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores) + 1e-9)
# 3. Hybrid Scoring
alpha = 0.5  # weight factor (0.5 = equal importance)
hybrid_scores = alpha * keyword_scores + (1 - alpha) * semantic_scores
# 4. Show Results
print("Keyword Search Results:")
for doc, score in sorted(zip(documents, keyword_scores), key=lambda x: x[1], reverse=True):
    print(f"Score={score:.4f} | {doc}")

print("\nSemantic Search Results:")
for doc, score in sorted(zip(documents, semantic_scores), key=lambda x: x[1], reverse=True):
    print(f"Score={score:.4f} | {doc}")

print("\nHybrid Search Results:")
for doc, score in sorted(zip(documents, hybrid_scores), key=lambda x: x[1], reverse=True):
    print(f"Score={score:.4f} | {doc}")
