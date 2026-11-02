import json
import numpy as np
from sentence_transformers import SentenceTransformer

class LocalVectorSearch:
    def __init__(self, kb_path: str = "kb_pairs.json"):
        # Load lightweight, high-performance local embedding model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.kb_path = kb_path
        self.documents = []
        self.embeddings = None
        self._initialize_kb()

    def _initialize_kb(self):
        """Loads kb_pairs.json and computes normalized embeddings across enriched text fields."""
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list formats or dictionary keys
        if isinstance(data, dict):
            self.documents = data.get("chunks", [])
        else:
            self.documents = data

        texts_to_embed = []
        for doc in self.documents:
            # Extract primary question, semantic variations, and answer for rich vector matching
            q = doc.get("question", "")
            variations = " ".join(doc.get("variations", []))
            answer = doc.get("answer", "")
            
            # Combine into a unified embedding text corpus
            combined_text = f"Question: {q} | Variations: {variations} | Answer: {answer}"
            texts_to_embed.append(combined_text)

        if not texts_to_embed:
            print("⚠️ Warning: Knowledge base is empty.")
            return

        # Precompute vector embeddings for the knowledge base
        print(f"Embedding {len(texts_to_embed)} knowledge base chunks locally...")
        self.embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True)
        
        # Normalize vectors for fast cosine similarity via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # Prevent division by zero if vectors are zero-length
        norms[norms == 0.0] = 1.0
        self.embeddings = self.embeddings / norms

    def search(self, query: str, top_k: int = 3) -> list:
        """Performs semantic similarity search against the user query."""
        if not self.documents or self.embeddings is None:
            return []

        # Encode and normalize query vector
        query_vector = self.model.encode([query], convert_to_numpy=True)
        q_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        q_norm[q_norm == 0.0] = 1.0
        query_vector = query_vector / q_norm

        # Compute cosine similarities via matrix multiplication
        similarities = np.dot(self.embeddings, query_vector.T).flatten()

        # Sort indices by highest similarity score
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = float(similarities[idx])
            results.append({
                "content": doc.get("answer", doc.get("content", "")),
                "metadata": doc,
                "score": score
            })

        return results