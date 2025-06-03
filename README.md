# DSPyProjectAwareCoder

Local, project‑aware coding assistant built with **DSPy 2.6.25** and **DeepSeek‑R1‑0528‑Qwen3‑8B (Q6\_K, YaRN 131 k)** served by `llama‑server`.  All inference, retrieval, and storage run on‑device—no cloud calls.

---

## 1 Current capabilities (v0.1)

| Stage             | Module                               | Highlights                                                                                                                                               |
| ----------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Index & Chunk** | `indexer/`                           | 2 k‑token, 50 %‑overlap chunks · tokenizer from GGUF via `llama‑cpp` · embeddings with **bge‑small‑en‑v1.5** · differential upsert & delete into Chroma. |
| **Retrieve**      | `dspy_agent/retrieval.py`            | k‑NN search (*k=10* default), max prompt ≤ 40 k tokens.                                                                                                  |
| **Reason**        | `dspy_agent/pipeline.py`             | DSPy Chain‑of‑Thought + JSON adapter compiled via **BootstrapFewShot**.                                                                                  |
| **Validate**      | `dspy_agent.assertions.RefAssertion` | Fails if cited references absent in answer.                                                                                                              |
| **CLI**           | `python ‑m dspy_agent.pipeline`      | Streams structured JSON.                                                                                                                                 |

### Example session

```bash
$ llama-server --model "$DEEPSEEK_GGUF" --port 8080 --n_ctx 65536 &
$ find indexer -name "*.py" -print0 | xargs -0 -n1 python -m indexer.upsert
$ python -m dspy_agent.pipeline "Explain indexer.upsert()"
{
  "solution": "The indexer.upsert function is used to index or update a file within the ChromaDB collection. It works by reading the file, splitting it into chunks, generating embeddings for the chunks, and then using the ChromaDB collection's upsert method to insert new chunks or update existing ones. Additionally, it deletes chunks that are no longer present in the file to keep the database consistent. The function is defined in the first snippet and relies on the chunking and embedding functions from the other snippets.",
  "references": [
    "The first snippet defines the index_file function which uses the COL.upsert method to update the ChromaDB collection with new embeddings and documents, while also deleting outdated ones.",
    "The second snippet provides the chunking function that splits the file content into manageable pieces and computes their sha256 hashes for tracking.",
    "The third snippet defines the embed function that converts text into vectors using a SentenceTransformer model, which is used in the upsert operation."
  ]
}
```

---

## 2 Roadmap

| Priority | Feature                                                                                                                                                                                                                                           | Purpose                                                               |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **1**    | **Persistent chat sessions** backed by MongoDB.                                                                                                                                                                                                   | Maintain dialogue context without rerunning full pipeline every turn. |
| **2**    | **Self‑refreshing index**: assistant triggers incremental upserts when files change during a session.                                                                                                                                             | Keeps retrieval in sync live.                                         |
| **3**    | **MongoDB memory store** for summaries, tool outputs, and long‑term notes.                                                                                                                                                                        | Enables iterative design loops.                                       |
| **4**    | **Tool‑calling (********`apply_patch`****\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*, ********************************************`run_tests`********************************************)** via DSPy tools. | Move from Q\&A to active pair‑programming.                            |

---

## 3 Environment (.env template)

```dotenv
# Quantised DeepSeek GGUF (YaRN 131k)
DEEPSEEK_GGUF="/…/unsloth_DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf"

# Embedding model
_EMB_MODEL="BAAI/bge-small-en-v1.5"

# llama‑server endpoint (OpenAI format)
LLM_API_BASE="http://127.0.0.1:8080/v1"
LLM_MODEL="openai/unsloth_DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf"
MAX_TOKEN=8000
TEMPERATURE=0.6

# Retrieval
RETRIEVE_K=10
```

---

## 4 Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)

# start inference server
llama-server --model "$DEEPSEEK_GGUF" --port 8080 --n_ctx 65536 &

# index your project
find my_project -name "*.py" -print0 | xargs -0 -n1 python -m indexer.upsert

# ask something
python -m dspy_agent.pipeline "How does foo.bar.validate() work?"
```

---

---

## 5 Why this project exists

This repository is purely for \*\*learning and exploration of how the **DSPy** programming model.What the **DeepSeek‑R1‑0528‑Qwen3‑8B (Q6\_K, YaRN 131 k)** model can achieve.

All findings will be documented in future commits; nothing here should be considered production‑ready.
