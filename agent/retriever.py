import os
import json
from typing import List, Dict, Any, Callable
from rank_bm25 import BM25Okapi
import numpy as np

# ======================
# Tokenization Strategy (Enables Cross-Lingual Extension)
# ======================
def default_tokenizer(text: str) -> List[str]:
    """
    Default tokenizer: whitespace split (works for English/Chinese w/o spaces).
    In real cross-lingual deployment, replace with:
      - Chinese: jieba.cut()
      - Japanese: MeCab
      - Multilingual: spaCy + language detection
    """
    return text.replace("，", ",").replace("。", ".").split()

# ======================
# Layer 1: Lexical Retriever (BM25) — Base Knowledge
# ======================
class BM25Retriever:
    """Layer 1: Fast keyword matching for symptoms, lab names, and critical values."""
    
    def __init__(self, evidence_dir: str, tokenizer: Callable = default_tokenizer):
        self.docs: List[str] = []
        self.keys: List[str] = []
        self.metadata: List[Dict] = []
        self.tokenizer = tokenizer
        self._load_and_index(evidence_dir)
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def _load_and_index(self, evidence_dir: str):
        for fname in os.listdir(evidence_dir):
            if fname.endswith(".json"):
                path = os.path.join(evidence_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Construct searchable text: title + evidence + red_flags
                    searchable_text = f"{data['title']} {data['evidence_text']}"
                    if "red_flags" in data:
                        flags = " ".join(data["red_flags"])
                        searchable_text += f" 危急值: {flags}"
                    
                    self.docs.append(searchable_text)
                    self.keys.append(data["key"])
                    self.metadata.append(data)
        
        self.tokenized_docs = [self.tokenizer(doc) for doc in self.docs]
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [
            {
                "key": self.keys[i],
                "text": self.docs[i],
                "score": float(scores[i]),
                "metadata": self.metadata[i],
                "retrieval_layer": "lexical"  # 标记为基础层
            }
            for i in top_k_idx
        ]

# ======================
# Layer 2: Semantic Retriever (FAISS Stub) — Expert Knowledge
# ======================
class DummyFAISSRetriever:
    """
    Layer 2: Semantic retrieval stub for clinical guidelines.
    In production, replace with:
      - embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
      - faiss.IndexFlatIP(embedding_dim)
    """
    
    def __init__(self, evidence_dir: str):
        self.evidence_chunks: List[Dict] = []
        self._load_evidence(evidence_dir)
    
    def _load_evidence(self, evidence_dir: str):
        for fname in os.listdir(evidence_dir):
            if fname.endswith(".json"):
                path = os.path.join(evidence_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.evidence_chunks.append({
                        "key": data["key"],
                        "text": data["evidence_text"],
                        "title": data["title"],
                        "metadata": data,
                        "retrieval_layer": "semantic"  # 🔹 标记为专家层
                    })
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        semantic ranking:
        - Critical topics (e.g., 'shock', 'acid_base') ranked higher
        """
        # Simple heuristic: prioritize life-threatening conditions
        priority_keys = {"shock_differential", "acid_base_disorders", "critical_lab_values"}
        prioritized = [c for c in self.evidence_chunks if c["key"] in priority_keys]
        others = [c for c in self.evidence_chunks if c["key"] not in priority_keys]
        combined = prioritized + others
        return combined[:k]

# ======================
# Fusion: Reciprocal Rank Fusion (RRF)
# ======================
def reciprocal_rank_fusion(
    faiss_results: List[Dict],
    bm25_results: List[Dict],
    k_rrf: int = 60
) -> List[Dict]:
    """
    🌟 RRF Algorithm: fuses lexical (BM25) and semantic (FAISS) results.
    Formula: score(doc) = Σ(1 / (k_rrf + rank))
    - Robust to score scale differences between retrievers
    - SOTA in multi-retriever fusion (used in MS MARCO, etc.)
    """
    fused_scores: Dict[str, float] = {}
    all_keys = set()

    # Process FAISS results (rank starts at 1)
    for rank, item in enumerate(faiss_results, start=1):
        key = item["key"]
        fused_scores[key] = fused_scores.get(key, 0) + 1.0 / (k_rrf + rank)
        all_keys.add(key)

    # Process BM25 results
    for rank, item in enumerate(bm25_results, start=1):
        key = item["key"]
        fused_scores[key] = fused_scores.get(key, 0) + 1.0 / (k_rrf + rank)
        all_keys.add(key)

    # Sort by fused score (descending)
    sorted_keys = sorted(all_keys, key=lambda x: fused_scores[x], reverse=True)
    
    # Reconstruct full result objects (preserve metadata & layer info)
    key_to_doc = {}
    for item in faiss_results + bm25_results:
        if item["key"] not in key_to_doc:
            key_to_doc[item["key"]] = item

    return [key_to_doc[key] for key in sorted_keys]

# ======================
# Unified Medical Evidence Retriever
# ======================
class MedicalEvidenceRetriever:
    """
    Cross-Lingual Hierarchical RAG Engine
    
    Usage:
        retriever = MedicalEvidenceRetriever("data/evidence_repository")
        results = retriever.retrieve("患者乳酸升高至5.0，血压低", top_k=3)
    
    Output includes:
        - Fused evidence from both lexical & semantic layers
        - Metadata for The Critic to validate reasoning
        - Language-agnostic tokenization ready for multilingual extension
    """
    
    def __init__(self, evidence_dir: str, tokenizer: Callable = default_tokenizer):
        self.bm25_retriever = BM25Retriever(evidence_dir, tokenizer=tokenizer)
        self.faiss_retriever = DummyFAISSRetriever(evidence_dir)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve and fuse evidence using hierarchical RAG strategy.
        Returns top_k most relevant medical knowledge snippets.
        """
        bm25_hits = self.bm25_retriever.search(query, k=top_k)
        faiss_hits = self.faiss_retriever.search(query, k=top_k)
        fused_results = reciprocal_rank_fusion(faiss_hits, bm25_hits, k_rrf=60)
        return fused_results[:top_k]