from typing import Any, Dict, List, cast
import chromadb
from indexer.embedder import embed

_client = chromadb.PersistentClient("vector_store")
COL = _client.get_collection("code_chunks")

def retrieve(question: str, k: int = 10) -> List[Dict[str, Any]]:
    q_vec = embed([question])[0]
    res = COL.query(
        query_embeddings=[q_vec],
        n_results=k,
        include=["documents", "metadatas"],
    )

    docs_raw: List[List[str]] = cast(List[List[str]], res.get("documents", [[]]))
    metas_raw: List[List[Dict[str, Any]]] = cast(List[List[Dict[str, Any]]], res.get("metadatas", [[]]))

    docs: List[str] = docs_raw[0] if docs_raw and docs_raw[0] else []
    metas: List[Dict[str, Any]] = metas_raw[0] if metas_raw and metas_raw[0] else []

    return [{"code": doc, **meta} for doc, meta in zip(docs, metas)]
