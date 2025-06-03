import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union, cast
from uuid import uuid4
import chromadb
from chromadb.api.models.Collection import Collection
from .chunker import yield_chunks
from .embedder import embed
import sys

Metadata = Mapping[str, Union[str, int, float, bool, None]]


_client = chromadb.PersistentClient("vector_store")
COL: Collection = _client.get_or_create_collection("code_chunks")

def _existing_chunks(path: str) -> Dict[str, str]:
    """Return {sha256: doc_id} for chunks of *path* already in DB."""
    raw: Any = COL.get(where={"file_path": path}, include=["metadatas"])
    ids: Sequence[str] = raw.get("ids", [])
    metas: Sequence[Metadata] = raw.get("metadatas", [])  # type: ignore[assignment]

    out: Dict[str, str] = {}
    for doc_id, meta in zip(ids, metas):
        if not meta:
            continue
        sha_val = meta.get("sha256")
        if isinstance(sha_val, str):
            out[sha_val] = doc_id
    return out

def index_file(file_path: str | Path) -> None:
    """Index or update *file_path* within the Chroma collection."""
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)

    rel = p.relative_to(Path.cwd()).as_posix()
    mod_ts = int(p.stat().st_mtime)

    existing = _existing_chunks(rel)

    ids_add: List[str] = []
    docs_add: List[str] = []
    metas_add: List[Metadata] = []
    present: set[str] = set()

    for ch in yield_chunks(p):
        h = ch["sha256"]
        present.add(h)
        if h in existing:
            continue
        ids_add.append(str(uuid4()))
        docs_add.append(ch["code"])
        metas_add.append(
            {
                "file_path": rel,
                "mod_ts": mod_ts,
                "line_start": ch["line_start"],
                "line_end": ch["line_end"],
                "sha256": h,
                "access_ts": 0,
                "access_n": 0,
            }
        )

    ids_del = [doc_id for h, doc_id in existing.items() if h not in present]

    if ids_add:
        COL.upsert(
            ids=ids_add,
            embeddings=embed(docs_add),
            documents=docs_add,
            metadatas=cast(List[Metadata], metas_add),  
        )

    if ids_del:
        COL.delete(ids=ids_del)



if __name__ == "__main__":
    start = time.perf_counter()
    index_file(sys.argv[1])
    print(f"Indexed {sys.argv[1]} in {(time.perf_counter()-start)*1_000:.1f} ms")