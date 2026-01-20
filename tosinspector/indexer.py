"""Indexing module using ChromaDB for vector storage and LlamaIndex for orchestration."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from tosinspector.config import settings
from tosinspector.utils import logger
from tosinspector.ingestion import DocumentChunk, DocumentIngester
from tosinspector.embeddings import OllamaEmbeddings


class IndexManifest:
    """
    Represents metadata about an index for persistence and reloading.

    The manifest is saved as JSON and allows the in-memory collection
    to be recreated or persisted externally in the future.
    """

    def __init__(
        self,
        index_name: str,
        source_file: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        num_chunks: int,
        created_at: Optional[str] = None
    ):
        self.index_name = index_name
        self.source_file = source_file
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_chunks = num_chunks
        self.created_at = created_at or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "index_name": self.index_name,
            "source_file": self.source_file,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "num_chunks": self.num_chunks,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexManifest":
        """Create manifest from dictionary."""
        return cls(**data)

    def save(self, directory: str) -> str:
        """
        Save manifest to JSON file.

        Args:
            directory: Directory to save the manifest

        Returns:
            str: Path to the saved manifest file
        """
        os.makedirs(directory, exist_ok=True)
        manifest_path = os.path.join(directory, "manifest.json")

        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved manifest to {manifest_path}")
        return manifest_path

    @classmethod
    def load(cls, directory: str) -> "IndexManifest":
        """
        Load manifest from JSON file.

        Args:
            directory: Directory containing the manifest

        Returns:
            IndexManifest: Loaded manifest

        Raises:
            FileNotFoundError: If manifest file doesn't exist
        """
        manifest_path = os.path.join(directory, "manifest.json")

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        with open(manifest_path, "r") as f:
            data = json.load(f)

        logger.info(f"Loaded manifest from {manifest_path}")
        return cls.from_dict(data)


class VectorIndex:
    """
    Vector index using ChromaDB for in-memory storage.

    This class manages:
    - In-memory ChromaDB collection
    - Document embedding and storage
    - Similarity search with metadata
    - Index persistence via manifest files

    Design note: The vector store is kept in memory for speed, but the
    manifest design allows for future persistence to disk if needed.
    """

    def __init__(
        self,
        index_name: str,
        embeddings: Optional[OllamaEmbeddings] = None
    ):
        """
        Initialize vector index.

        Args:
            index_name: Name of the index (used as collection name)
            embeddings: OllamaEmbeddings instance (creates new if not provided)
        """
        self.index_name = index_name
        self.embeddings = embeddings or OllamaEmbeddings()

        # Initialize ChromaDB in-memory client
        self.chroma_client = chromadb.Client(
            ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=index_name,
            metadata={"created_at": datetime.now().isoformat()}
        )

        logger.info(f"Initialized VectorIndex '{index_name}'")

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the index.

        Args:
            chunks: List of DocumentChunk objects to add
        """
        if not chunks:
            logger.warning("No chunks to add to index")
            return

        logger.info(f"Adding {len(chunks)} chunks to index '{self.index_name}'")

        # Extract texts and metadata
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.id for chunk in chunks]

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embeddings.embed(texts)

        # Add to collection
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )

        logger.info(f"Successfully added {len(chunks)} chunks to index")

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query the index for similar chunks.

        Args:
            query_text: Query string
            top_k: Number of top results to return (defaults to settings.top_k)

        Returns:
            Dict containing:
                - ids: List of chunk IDs
                - documents: List of chunk texts
                - metadatas: List of metadata dicts
                - distances: List of similarity distances
        """
        k = top_k or settings.top_k

        logger.info(f"Querying index '{self.index_name}' with top_k={k}")

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query_text)

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        # ChromaDB returns results as lists of lists (for multiple queries)
        # Extract the first query's results
        output = {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

        logger.info(f"Retrieved {len(output['ids'])} results")

        return output

    def get_all_chunks(self) -> Dict[str, Any]:
        """
        Get all chunks from the index.

        Returns:
            Dict containing ids, documents, and metadatas
        """
        results = self.collection.get()

        return {
            "ids": results["ids"],
            "documents": results["documents"],
            "metadatas": results["metadatas"]
        }

    def count(self) -> int:
        """
        Get the number of chunks in the index.

        Returns:
            int: Number of chunks
        """
        return self.collection.count()


class Indexer:
    """
    High-level indexer that orchestrates ingestion, embedding, and storage.

    This is the main API for creating and managing indexes.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize indexer.

        Args:
            chunk_size: Chunk size for document splitting
            chunk_overlap: Overlap between chunks
        """
        self.ingester = DocumentIngester(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embeddings = OllamaEmbeddings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def create_index(
        self,
        file_path: str,
        index_name: str,
        save_manifest: bool = True
    ) -> tuple[VectorIndex, IndexManifest]:
        """
        Create an index from a document file.

        Args:
            file_path: Path to the document file
            index_name: Name for the index
            save_manifest: Whether to save manifest to disk

        Returns:
            tuple: (VectorIndex, IndexManifest)
        """
        logger.info(f"Creating index '{index_name}' from {file_path}")

        # Ingest document
        chunks = self.ingester.ingest(file_path)

        if not chunks:
            raise ValueError(f"No chunks created from {file_path}")

        # Create vector index
        index = VectorIndex(index_name, embeddings=self.embeddings)

        # Add chunks to index
        index.add_chunks(chunks)

        # Create manifest
        manifest = IndexManifest(
            index_name=index_name,
            source_file=file_path,
            embedding_model=settings.ollama_embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            num_chunks=len(chunks)
        )

        # Save manifest if requested
        if save_manifest:
            index_dir = os.path.join(settings.index_dir, index_name)
            manifest.save(index_dir)

        logger.info(
            f"Successfully created index '{index_name}' with {len(chunks)} chunks"
        )

        return index, manifest

    @staticmethod
    def load_index(index_name: str) -> tuple[VectorIndex, IndexManifest]:
        """
        Load an existing index from disk.

        Note: Since we use in-memory ChromaDB, this requires re-indexing
        the source document. In a future version with persistent storage,
        this could load the collection directly.

        Args:
            index_name: Name of the index to load

        Returns:
            tuple: (VectorIndex, IndexManifest)

        Raises:
            FileNotFoundError: If manifest doesn't exist
        """
        index_dir = os.path.join(settings.index_dir, index_name)

        # Load manifest
        manifest = IndexManifest.load(index_dir)

        logger.info(f"Loading index '{index_name}' from manifest")

        # Re-create index from source file
        # Note: This re-processes the document. For persistence, we'd load
        # the vectors directly instead.
        indexer = Indexer(
            chunk_size=manifest.chunk_size,
            chunk_overlap=manifest.chunk_overlap
        )

        # Check if source file exists
        if not os.path.exists(manifest.source_file):
            raise FileNotFoundError(
                f"Source file {manifest.source_file} not found. "
                f"Cannot reload index."
            )

        index, _ = indexer.create_index(
            file_path=manifest.source_file,
            index_name=index_name,
            save_manifest=False  # Don't overwrite existing manifest
        )

        return index, manifest
