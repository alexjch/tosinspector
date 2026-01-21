"""Unit tests for Ollama embeddings adapter."""

import pytest
from typing import Any
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from tosinspector.embeddings import OllamaEmbeddings, OllamaEmbeddingsError


class TestOllamaEmbeddings:
    """Test cases for OllamaEmbeddings."""

    @patch('tosinspector.embeddings.requests.Session')
    def test_embed_query_success(self, mock_session_class: Mock) -> None:
        """Test successful single query embedding."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        # Test
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434")
        result = embeddings.embed_query("test query")

        # Assertions
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert np.allclose(result, [0.1, 0.2, 0.3])

    @patch('tosinspector.embeddings.requests.Session')
    def test_embed_query_empty_text(self, mock_session_class: Mock) -> None:
        """Test that empty text raises ValueError."""
        embeddings = OllamaEmbeddings()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            embeddings.embed_query("")

    @patch('tosinspector.embeddings.requests.Session')
    def test_embed_batch_success(self, mock_session_class: Mock) -> None:
        """Test successful batch embedding."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock responses for each text
        def mock_post(*args: Any, **kwargs: Any) -> Mock:
            """
            Mock function for HTTP POST requests to simulate embedding API responses.

            Args:
                *args: Variable length argument list (unused, required for mock signature).
                **kwargs: Arbitrary keyword arguments (unused, required for mock signature).

            Returns:
                Mock: A mock response object with a json() method that returns a dictionary
                      containing an 'embedding' key with a list of float values [0.1, 0.2, 0.3],
                      and a raise_for_status() method for HTTP error handling.
            """
            mock_response = Mock()
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            mock_response.raise_for_status = Mock()
            return mock_response

        mock_session.post.side_effect = mock_post

        # Test
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434")
        texts = ["text1", "text2", "text3"]
        result = embeddings.embed(texts)

        # Assertions
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)  # 3 texts, 3-dim embeddings

    @patch('tosinspector.embeddings.requests.Session')
    def test_embed_empty_list(self, mock_session_class: Mock) -> None:
        """Test that empty list raises ValueError."""
        embeddings = OllamaEmbeddings()

        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            embeddings.embed([])

    @patch('tosinspector.embeddings.requests.Session')
    def test_request_timeout(self, mock_session_class: Mock) -> None:
        """Test handling of request timeout."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        import requests
        mock_session.post.side_effect = requests.exceptions.Timeout()

        # Test
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434")

        with pytest.raises(OllamaEmbeddingsError, match="timed out"):
            embeddings.embed_query("test")

    @patch('tosinspector.embeddings.requests.Session')
    def test_connection_error(self, mock_session_class: Mock) -> None:
        """Test handling of connection error."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        import requests
        mock_session.post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        # Test
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434")

        with pytest.raises(OllamaEmbeddingsError, match="Failed to connect"):
            embeddings.embed_query("test")

    @patch('tosinspector.embeddings.requests.Session')
    def test_http_error(self, mock_session_class: Mock) -> None:
        """Test handling of HTTP error."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        import requests
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_session.post.return_value = mock_response

        # Test
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434")

        with pytest.raises(OllamaEmbeddingsError, match="HTTP error"):
            embeddings.embed_query("test")

    @patch('tosinspector.embeddings.requests.Session')
    def test_missing_embedding_field(self, mock_session_class: Mock) -> None:
        """Test handling of response without embedding field."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.json.return_value = {"error": "something went wrong"}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        # Test
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434")

        with pytest.raises(OllamaEmbeddingsError, match="missing 'embedding' field"):
            embeddings.embed_query("test")
