import os
import json
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

# FAISS
class DummyFAISSRetriever:
    def __init__(self, evidence_dir: str):
        self.evidence_chunks = []
        self._load_evidence(evidence_dir)
    
    def _load_evidence(self, evidence_dir: str):
        for fname in os.listdir(evidence_dir):
            if fname.endswith(".json"):
                with open(os.path.join(evidence_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.evidence_chunks.append({
                        "key": data["key"],
                        "text": data["evidence_text"],
                        "title": data["title"]
                    })
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        # 返回前 k 个（实际应计算 embedding 相似度）
        return self.evidence_chunks[:k]

# 真实 BM25 检索器
class BM25Retriever:
    def __init__(self, evidence_dir: str):
        self.docs = []
        self.keys = []
        self._load_and_tokenize(evidence_dir)
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def _load_and_tokenize(self, evidence_dir: str):
        for fname in os.listdir(evidence_dir):
            if fname.endswith(".json"):
                with open(os.path.join(evidence_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    text = f"{data['title']} {data['evidence_text']}"
                    self.docs.append(text)
                    self.keys.append(data["key"])
        self.tokenized_docs = [doc.split() for doc in self.docs]
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_doc=tokenized_query)
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [
            {"key": self.keys[i], "text": self.docs[i], "score": float(scores[i])}
            for i in top_k_idx
        ]

# Reciprocal Rank Fusion (RRF) 核心公式
def reciprocal_rank_fusion(
    faiss_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    RRF = Σ(1 / (k + rank))
    合并两路结果，rank 从 1 开始
    """
    fused_scores = {}
    all_keys = set()

    # FAISS 路（假设已按相关性排序）
    for rank, item in enumerate(faiss_results, start=1):
        key = item["key"]
        fused_scores[key] = fused_scores.get(key, 0) + 1.0 / (k + rank)
        all_keys.add(key)

    # BM25 路
    for rank, item in enumerate(bm25_results, start=1):
        key = item["key"]
        fused_scores[key] = fused_scores.get(key, 0) + 1.0 / (k + rank)
        all_keys.add(key)

    # 按融合分排序
    sorted_keys = sorted(all_keys, key=lambda x: fused_scores[x], reverse=True)
    
    # 重建结果（从任一源取完整文本）
    key_to_doc = {}
    for item in faiss_results + bm25_results:
        if item["key"] not in key_to_doc:
            key_to_doc[item["key"]] = item

    return [key_to_doc[key] for key in sorted_keys]

# 主检索类
class MedicalEvidenceRetriever:
    def __init__(self, evidence_dir: str):
        self.faiss_retriever = DummyFAISSRetriever(evidence_dir)
        self.bm25_retriever = BM25Retriever(evidence_dir)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        faiss_hits = self.faiss_retriever.search(query, k=top_k)
        bm25_hits = self.bm25_retriever.search(query, k=top_k)
        fused = reciprocal_rank_fusion(faiss_hits, bm25_hits)
        return fused[:top_k]