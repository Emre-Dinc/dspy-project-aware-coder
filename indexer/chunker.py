from pathlib import Path
from hashlib import sha256
from typing import Iterator, Dict, List

from llama_cpp import Llama
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())        
MODEL_PATH = Path(os.environ["DEEPSEEK_GGUF"]).expanduser()
LLM = Llama(model_path=str(MODEL_PATH), n_ctx=2048, n_threads=0, verbose=False)

TOKEN_LIMIT = 2048        
OVERLAP      = 1024       

def _encode(text: str) -> List[int]:
    return LLM.tokenize(text.encode("utf-8"), add_bos=False)

def _decode(tokens: List[int]) -> str:
    return LLM.detokenize(tokens).decode("utf-8", errors="replace")


def yield_chunks(file_path: Path) -> Iterator[Dict]:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    toks = _encode(text)
    i = 0
    while i < len(toks):
        block = toks[i : i + TOKEN_LIMIT]
        if not block:
            break
        chunk_txt = _decode(block)
        yield {
            "code": chunk_txt,
            "line_start": i,
            "line_end": i + len(block) - 1,
            "sha256": sha256(chunk_txt.encode()).hexdigest(),
        }
        i += OVERLAP