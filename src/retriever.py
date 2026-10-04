import json
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name,device="cpu")
        self.index = None
        self.chunks = None
        self.bm25 = None
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            device="cpu"
        )
        
    def build_index(self, chunks):
        self.chunks = chunks

        tokenized_chunks = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        return self.index

    def semantic_search(self, query, top_k=20):
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks[int(idx)]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "semantic_score": float(score)
            })

        return results


    def save_index(self, path):
        if self.index is None:
            raise ValueError("Index has not been built yet.")

        faiss.write_index(self.index, path)

    def save_chunks(self, path):
        if self.chunks is None:
            raise ValueError("No chunks to save.")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self, index_path, chunks_path):
        self.index = faiss.read_index(index_path)

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        tokenized_chunks = [
            chunk["text"].lower().split()
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def lexical_search(self, query, top_k=20):
        if self.bm25 is None:
            raise ValueError("BM25 index has not been built.")

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:top_k]

        results = []

        for idx in top_indices:
            chunk = self.chunks[int(idx)]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "bm25_score": float(scores[idx])
            })

        return results

    def reciprocal_rank_fusion(self, semantic_results, lexical_results, k=60):
        rrf_scores = {}

        for rank, result in enumerate(semantic_results, start=1):
            idx = result["chunk_id"]

            rrf_scores[idx] = (
                rrf_scores.get(idx, 0)
                + 1 / (k + rank)
            )

        for rank, result in enumerate(lexical_results, start=1):
            idx = result["chunk_id"]

            rrf_scores[idx] = (
                rrf_scores.get(idx, 0)
                + 1 / (k + rank)
            )

        ranked_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked_results

    def hybrid_search(self, query, top_k=20):
        semantic_results = self.semantic_search(query, top_k)
        lexical_results = self.lexical_search(query, top_k)

        fused_results = self.reciprocal_rank_fusion(
            semantic_results,
            lexical_results
        )

        results = []

        for chunk_id, rrf_score in fused_results[:top_k]:
            chunk = self.chunks[chunk_id]

            results.append({
                "chunk_id": chunk_id,
                "page": chunk["page"],
                "text": chunk["text"],
                "rrf_score": float(rrf_score)
            })

        return results

    def rerank(self, query, candidates, top_k=5):
        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        scores = self.reranker.predict(pairs)

        intent = self.classify_intent(query)

        for candidate, score in zip(candidates, scores):
            bonus = self.intent_bonus(
                intent,
                candidate["text"]
            )

            candidate["rerank_score"] = float(score)
            candidate["intent_bonus"] = bonus
            candidate["final_score"] = float(score) + bonus

        candidates.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return candidates[:top_k]

    def classify_intent(self, query):
        q = query.lower().strip()

        if q.startswith(("what is", "what are", "define", "definition")):
            return "definition"

        elif q.startswith(("derive", "show that", "prove")):
            return "derivation"

        elif q.startswith(("calculate", "find", "compute", "solve")):
            return "numerical"

        elif q.startswith(("compare", "difference between", "distinguish")):
            return "comparison"

        elif q.startswith((
            "what are the properties",
            "what properties",
            "list the properties"
        )):
            return "properties"

        return "general"




    def intent_bonus(self, intent, text):
        text = text.lower()

        if intent == "definition":
            patterns = [
                "is defined as",
                "is called",
                "we define",
                "definition",
                "is a projection operator if"
            ]

            if any(pattern in text for pattern in patterns):
                return 2.0

        return 0.0