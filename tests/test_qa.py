"""Unit tests for retrieval and QA logic."""

from unittest.mock import Mock, MagicMock, patch
import numpy as np

from tosinspector.qa import QAEngine, OllamaGenerator
from tosinspector.indexer import VectorIndex
from tosinspector.embeddings import OllamaEmbeddings


class TestQAEngine:
    """Test cases for QAEngine."""

    @patch('tosinspector.indexer.chromadb.Client')
    @patch.object(OllamaEmbeddings, 'embed')
    @patch.object(OllamaEmbeddings, 'embed_query')
    @patch.object(OllamaGenerator, 'generate')
    def test_query_success(
        self,
        mock_generate: Mock,
        mock_embed_query: Mock,
        mock_embed: Mock,
        mock_chromadb: Mock
    ) -> None:
        """Test successful query execution."""
        # Setup mocks
        mock_embed_query.return_value = np.array([0.1, 0.2, 0.3])
        mock_generate.return_value = "The refund policy allows returns within 30 days [chunk_0]."

        # Mock ChromaDB collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["chunk_0", "chunk_1"]],
            "documents": [["Refunds are allowed within 30 days.", "No refunds after 60 days."]],
            "metadatas": [[
                {"source": "tos.pdf", "page": 1, "start_char": 0, "end_char": 100},
                {"source": "tos.pdf", "page": 2, "start_char": 100, "end_char": 200}
            ]],
            "distances": [[0.1, 0.3]]
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_chroma_client

        # Create index and QA engine
        index = VectorIndex("test_index")
        qa_engine = QAEngine(index, top_k=2)

        # Test query
        result = qa_engine.query("What is the refund policy?")

        # Assertions
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) == 2
        assert result["chunks_used"] >= 0
        assert isinstance(result["truncated"], bool)

    @patch('tosinspector.indexer.chromadb.Client')
    @patch.object(OllamaEmbeddings, 'embed_query')
    def test_query_no_results(
        self,
        mock_embed_query: Mock,
        mock_chromadb: Mock
    ) -> None:
        """Test query with no retrieved results."""
        # Setup mocks
        mock_embed_query.return_value = np.array([0.1, 0.2, 0.3])

        # Mock ChromaDB collection with no results
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_chroma_client

        # Create index and QA engine
        index = VectorIndex("test_index")
        qa_engine = QAEngine(index)

        # Test query
        result = qa_engine.query("What is the refund policy?")

        # Assertions
        assert result["answer"] == "Not in document."
        assert len(result["sources"]) == 0

    @patch('tosinspector.indexer.chromadb.Client')
    @patch.object(OllamaEmbeddings, 'embed_query')
    @patch.object(OllamaGenerator, 'generate')
    def test_context_formatting(
        self,
        mock_generate: Mock,
        mock_embed_query: Mock,
        mock_chromadb: Mock
    ) -> None:
        """Test that context is properly formatted with metadata."""
        # Setup mocks
        mock_embed_query.return_value = np.array([0.1, 0.2, 0.3])
        mock_generate.return_value = "Test answer"

        # Mock ChromaDB collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["chunk_0"]],
            "documents": [["Test document content"]],
            "metadatas": [[
                {"source": "test.pdf", "page": 1, "start_char": 0, "end_char": 100}
            ]],
            "distances": [[0.1]]
        }

        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_chroma_client

        # Create index and QA engine
        index = VectorIndex("test_index")
        qa_engine = QAEngine(index)

        # Test query
        _ = qa_engine.query("Test question")

        # Verify generate was called
        assert mock_generate.called

        # Check that the prompt contains context markers
        call_args = mock_generate.call_args[0][0]
        assert "[chunk_0]" in call_args
        assert "Source: test.pdf" in call_args
        assert "Page: 1" in call_args
