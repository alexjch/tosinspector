
## ToSInspector Design
Build a Python project named "tosinspector" that satisfies the following specification. If any requirement is ambiguous or missing, ask specific, targeted questions before writing code.

### Project summary
- Purpose: accept a Terms of Service (ToS) document (text/HTML/PDF), index it, and provide a CLI that lets a user ask questions in natural language about the ToS and receive answers with source citations.
- Architecture: CLI entrypoint backed by a well-organized API layer so the core logic can be reused by a future web UI.
- LLM/embeddings: remote Ollama server (different host) provides embeddings (/api/embed) and generation (/api/generate).
- Indexing/retrieval: Use LlamaIndex for orchestration and in-memory Chroma (chromadb) as the vector store. Keep the vector DB in memory (no persistent storage), but design code so persistence could be added later.
- Embeddings: call Ollama for embeddings; ensure consistent model name is configurable.
- Language: Python 3.11+, use type hints and Pydantic for configs/schemas.
- CLI: build with Typer; commands: index, query, summarize, list-sources, export(optional).
- Testing: include unit tests for core components.

### Non-functional requirements
- Modular: Clean package layout separating CLI, API layer, ingestion, indexing, embeddings adapter (Ollama), retriever/QA, and utils.
- Configurable: All external endpoints, model names, chunk sizes, and top_k values must be configurable via a single settings module or env vars (use Pydantic BaseSettings).
- Robust: Validate inputs, handle common errors (network failures to Ollama, empty docs), and implement sensible defaults.
- Secure defaults: Use HTTPS for Ollama calls, allow configuring timeouts and verify SSL.

### Behavioral & prompt rules for the QA agent
- The LLM prompt sent to Ollama for generation must:
  - Include only the top-k retrieved context chunks with metadata (id, source, page/start-end offsets).
  - Start with a short system instruction: you are an assistant answering only from the context, if answer not present reply exactly: "Not in document."
  - Include the user question and ask for concise answers with citations in the form [chunk-id].
  - Limit overall prompt size; if combined tokens exceed the model's context, include highest-relevance chunks until limit reached and note that context was truncated.
- Provide the exact prompt template to use (include notation for where contexts, metadata, and question are inserted).

### Implementation details and constraints
- Chunking: default chunk_size=1000 chars, overlap=200; store metadata: source filename, page number if available, char offsets.
- Embeddings adapter:
  - Implement an OllamaEmbeddings class with methods embed(texts: list[str]) -> np.ndarray and embed_query(text: str) -> np.ndarray.
  - Batch requests where possible; handle rate-limiting/backoff.
  - Configurable endpoint (base_url), model name, timeout, and optional API key header.
- Vector store:
  - Use chromadb in-memory client for collections; collection name per-index.
  - Store texts, metadatas, ids, and embeddings.
  - For queries, compute cosine similarity via chroma's query using query_embeddings from Ollama.
- LlamaIndex usage:
  - Use LlamaIndex only for document/node orchestration if desired; the vector store and embeddings are handled explicitly. Ensure LlamaIndex is integrated so future features (synthesizers, chains) are easy to add.
  - Save a minimal manifest for in-memory indexes (JSON) that records index name, source file, embedding model, and creation timestamp — allow reloading collections into memory if persisted externally later.
- CLI behavior:
  - index <file> --name <index_name>: loads, chunks, gets embeddings from Ollama, builds an in-memory chroma collection, writes a manifest to disk (manifest.json) in an index folder, and prints summary (num chunks).
  - query <index_folder> --question "<q>" [--top_k 5]: loads manifest and in-memory collection, embeds query via Ollama, retrieves top_k, assembles prompt, calls Ollama generate, and prints answer + citations and source metadata.
  - summarize: uses retrieved chunks (or a dedicated summary prompt) to produce a short summary of the ToS.
  - list-sources: prints chunk ids and metadata.
- Testing:
  - Provide unit tests for chunking, embeddings adapter (mocked HTTP), and retrieval logic.
- Logging: include structured logging (info, warn, error).

### Deliverables
- A ready-to-run project scaffold with:
  - package modules (tos-inspect, api, ingestion, indexer, embeddings, qa, utils)
  - example index workflow showing how to index sample_tos.pdf and run a query
  - README with setup and usage
  - minimal tests
- Annotated code comments explaining key decisions and where to change Ollama endpoints/models.

