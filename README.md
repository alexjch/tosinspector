# ToS Inspector

A Python CLI tool for indexing and querying Terms of Service documents using LLM-powered semantic search. Built with LlamaIndex, ChromaDB, and Ollama for embeddings and generation.

## Features

- 📄 **Multiple Format Support**: Index text, HTML, and PDF documents
- 🔍 **Semantic Search**: Query documents using natural language
- 📚 **Citation Support**: Get answers with source references (chunk IDs, page numbers, character offsets)
- 🤖 **LLM-Powered QA**: Uses remote Ollama server for embeddings and generation
- 💾 **In-Memory Vector Store**: Fast retrieval using ChromaDB
- 🎯 **Modular Architecture**: Clean separation between CLI, API, and core logic
- ⚙️ **Fully Configurable**: All settings via environment variables or .env file

## Architecture

```
tosinspector/
├── cli.py           # Typer-based CLI interface
├── config.py        # Pydantic settings and configuration
├── embeddings.py    # Ollama embeddings adapter
├── ingestion.py     # Document loading and chunking
├── indexer.py       # ChromaDB vector index management
├── qa.py            # Question answering with prompt engineering
└── utils.py         # Logging and utilities
```

## Requirements

- Python 3.11+
- Access to a remote Ollama server with:
  - Embedding model (e.g., `nomic-embed-text`)
  - Generation model (e.g., `llama2`, `mistral`)

## Installation

### 1. Clone the repository

```bash
cd tosinspector
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

### 4. Configure Ollama connection

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Edit `.env` with your Ollama server details:

```env
# Ollama Configuration
OLLAMA_BASE_URL=https://your-ollama-server.com
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_GENERATION_MODEL=llama2
OLLAMA_TIMEOUT=60
OLLAMA_VERIFY_SSL=true

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval Configuration
TOP_K=5
MAX_CONTEXT_TOKENS=4000
```

**Important**: Update `OLLAMA_BASE_URL` to point to your Ollama server. The server must expose:
- `/api/embeddings` endpoint for embeddings
- `/api/generate` endpoint for text generation

## Usage

### Index a Document

Index a ToS document (text, HTML, or PDF):

```bash
tosinspector index sample_tos.pdf --name my_tos
```

This will:
1. Load and parse the document
2. Split it into chunks (default: 1000 chars with 200 char overlap)
3. Generate embeddings via Ollama
4. Store in an in-memory ChromaDB collection
5. Save a manifest to `indexes/my_tos/manifest.json`

### Query an Index

Ask questions about the indexed document:

```bash
tosinspector query my_tos --question "What is the refund policy?"
```

Example output:
```
Answer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The refund policy allows returns within 30 days 
of purchase [chunk_5]. No refunds are provided 
after 60 days [chunk_8].
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Chunks used: 5/5

Sources:
┏━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━┓
┃ Chunk ID ┃ Source       ┃ Page ┃ Char Range   ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━┩
│ chunk_5  │ sample.pdf   │ 3    │ 2000-3000    │
│ chunk_8  │ sample.pdf   │ 5    │ 4000-5000    │
└──────────┴──────────────┴──────┴──────────────┘
```

### Summarize a Document

Generate a summary of the indexed document:

```bash
tosinspector summarize my_tos
```

### List Sources

View all chunks and their metadata:

```bash
tosinspector list-sources my_tos --limit 10
```

### List All Indexes

Show all available indexes:

```bash
tosinspector list-indexes
```

## CLI Commands

### `index`

Index a ToS document file.

```bash
tosinspector index <file> --name <index_name> [options]
```

**Arguments**:
- `file`: Path to the document (text/HTML/PDF)

**Options**:
- `--name, -n`: Name for the index (required)
- `--chunk-size`: Override default chunk size
- `--chunk-overlap`: Override default chunk overlap

**Example**:
```bash
tosinspector index terms.pdf --name terms_v1 --chunk-size 800
```

### `query`

Query an indexed document.

```bash
tosinspector query <index_name> --question <question> [options]
```

**Arguments**:
- `index_name`: Name of the index to query

**Options**:
- `--question, -q`: Question to ask (required)
- `--top-k, -k`: Number of chunks to retrieve (default: 5)
- `--show-sources/--no-sources`: Show/hide source citations (default: show)

**Example**:
```bash
tosinspector query terms_v1 --question "Can I cancel my subscription?" --top-k 3
```

### `summarize`

Generate a summary of the indexed document.

```bash
tosinspector summarize <index_name>
```

**Example**:
```bash
tosinspector summarize terms_v1
```

### `list-sources`

List all chunks and metadata from an index.

```bash
tosinspector list-sources <index_name> [--limit N]
```

**Options**:
- `--limit, -l`: Maximum number of chunks to display

**Example**:
```bash
tosinspector list-sources terms_v1 --limit 20
```

### `list-indexes`

Show all available indexes.

```bash
tosinspector list-indexes
```

## Configuration

All configuration is handled via environment variables or a `.env` file. See [.env.example](.env.example) for all available options.

### Key Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama server URL | `https://localhost:11434` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model name | `nomic-embed-text` |
| `OLLAMA_GENERATION_MODEL` | Generation model name | `llama2` |
| `OLLAMA_TIMEOUT` | Request timeout (seconds) | `60` |
| `OLLAMA_VERIFY_SSL` | Verify SSL certificates | `true` |
| `CHUNK_SIZE` | Characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Overlapping characters | `200` |
| `TOP_K` | Default chunks to retrieve | `5` |
| `MAX_CONTEXT_TOKENS` | Max prompt context tokens | `4000` |

### Changing Ollama Endpoints/Models

To use a different Ollama server or models:

1. **Update `.env` file**:
   ```env
   OLLAMA_BASE_URL=https://my-ollama-server.com
   OLLAMA_EMBEDDING_MODEL=my-custom-embed-model
   OLLAMA_GENERATION_MODEL=my-custom-gen-model
   ```

2. **Or set environment variables**:
   ```bash
   export OLLAMA_BASE_URL=https://my-ollama-server.com
   export OLLAMA_EMBEDDING_MODEL=my-custom-embed-model
   ```

The settings are automatically loaded from [tosinspector/config.py](tosinspector/config.py).

## Development

### Running Tests

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=tosinspector --cov-report=html
```

### Code Quality

Format code with Black:

```bash
black tosinspector/ tests/
```

Lint with Ruff:

```bash
ruff check tosinspector/ tests/
```

Type check with mypy:

```bash
mypy tosinspector/
```

## Example Workflow

Here's a complete example of indexing and querying a sample ToS document:

```bash
# 1. Create a sample document
cat > sample_tos.txt << 'EOF'
Terms of Service

1. Refund Policy
Refunds are available within 30 days of purchase. After 30 days, 
no refunds will be issued. All refund requests must be submitted 
via email to support@example.com.

2. Privacy Policy
We collect user data including email addresses, usage statistics, 
and device information. This data is used to improve our services 
and may be shared with third-party analytics providers.

3. User Obligations
Users must not misuse the service, share accounts, or engage in 
fraudulent activities. Violation of these terms may result in 
account suspension or termination.
EOF

# 2. Index the document
tosinspector index sample_tos.txt --name sample

# 3. Query the index
tosinspector query sample --question "What is the refund policy?"

# 4. Get a summary
tosinspector summarize sample

# 5. List all chunks
tosinspector list-sources sample --limit 5
```

## Prompt Engineering

The QA system uses a carefully designed prompt template (see [tosinspector/qa.py](tosinspector/qa.py)):

```python
SYSTEM_INSTRUCTION = """You are an assistant that answers questions using 
only the provided context. If the answer is not present in the context, 
reply exactly: "Not in document." Always provide citations in the form 
[chunk-id] for any information you use from the context."""

PROMPT_TEMPLATE = """{system_instruction}

CONTEXT:
{context}

QUESTION: {question}

Provide a concise answer with citations in the form [chunk-id]. If the 
answer is not in the context, respond with: "Not in document."

ANSWER:"""
```

Context includes:
- Chunk ID
- Source filename
- Page number (for PDFs)
- Character offset range

Example context format:
```
[chunk_5] (Source: terms.pdf, Page: 3, Chars: 2000-3000)
Refunds are available within 30 days of purchase...
```

## Project Structure

```
tosinspector/
├── tosinspector/          # Main package
│   ├── __init__.py
│   ├── cli.py            # CLI commands (Typer)
│   ├── config.py         # Configuration (Pydantic)
│   ├── embeddings.py     # Ollama embeddings adapter
│   ├── ingestion.py      # Document loading & chunking
│   ├── indexer.py        # Vector index (ChromaDB)
│   ├── qa.py             # QA engine & prompt engineering
│   └── utils.py          # Logging utilities
├── tests/                # Unit tests
│   ├── __init__.py
│   ├── test_embeddings.py
│   ├── test_ingestion.py
│   └── test_qa.py
├── indexes/              # Index manifests (created at runtime)
├── .env.example          # Example environment variables
├── .gitignore
├── pyproject.toml        # Project metadata & dependencies
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Design Decisions

### In-Memory Vector Store

The vector store uses ChromaDB's in-memory client for speed. While the vectors aren't persisted, a manifest file is saved with metadata (source file, model, chunk config). This design allows:

1. **Fast retrieval** during a session
2. **Easy reconstruction** from source files
3. **Future persistence** by swapping to persistent ChromaDB

To add persistence in the future, change in [tosinspector/indexer.py](tosinspector/indexer.py):

```python
# Current (in-memory)
self.chroma_client = chromadb.Client(...)

# Future (persistent)
self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
```

### Modular Architecture

The project separates concerns:
- **CLI layer** (`cli.py`) - User interface
- **API layer** (`indexer.py`, `qa.py`) - Core logic
- **Adapters** (`embeddings.py`, `ingestion.py`) - External services

This makes it easy to:
- Add a web UI (reuse API layer)
- Swap Ollama for OpenAI (replace embeddings adapter)
- Add new document formats (extend ingestion module)

### Error Handling

All modules include comprehensive error handling:
- Network failures (timeouts, connection errors)
- Invalid inputs (empty documents, malformed files)
- API errors (Ollama unavailable, rate limiting)

### Type Safety

Uses Python 3.11+ features:
- Type hints throughout
- Pydantic for configuration validation
- Dataclasses for structured data

## Troubleshooting

### "Connection refused" error

Ensure your Ollama server is running and accessible. Test with:

```bash
curl https://your-ollama-server.com/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "test"
}'
```

### "Not in document" responses

The model couldn't find relevant information. Try:
- Rephrasing your question
- Increasing `--top-k` to retrieve more chunks
- Checking if the information exists with `list-sources`

### Slow indexing

Embedding generation can be slow for large documents. To speed up:
- Reduce `CHUNK_SIZE` (fewer, larger chunks)
- Use a faster embedding model
- Ensure good network connection to Ollama server

## Future Enhancements

Potential additions mentioned in the design:

- [ ] Persistent vector store (save embeddings to disk)
- [ ] Export command (export index to JSON/CSV)
- [ ] Web UI (FastAPI + React)
- [ ] Support for more file formats (Word, Markdown)
- [ ] Multi-document indexing (combine multiple ToS)
- [ ] Query history and analytics
- [ ] Custom embedding models (local transformers)

## License

MIT

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Format code (`black`, `ruff`)
6. Submit a pull request

## Support

For issues or questions, please open an issue on GitHub.
