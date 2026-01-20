"""Ollama embeddings adapter with batch processing and error handling."""

import time
from typing import List, Optional, Dict, Any
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tosinspector.config import settings
from tosinspector.utils import logger


class OllamaEmbeddingsError(Exception):
    """Custom exception for Ollama embeddings errors."""
    pass


class OllamaEmbeddings:
    """
    Adapter for generating embeddings via remote Ollama server.

    This class handles:
    - Single and batch embedding generation
    - Error handling and retries with exponential backoff
    - Rate limiting
    - SSL verification and timeout configuration

    To change the Ollama endpoint or model, update the settings in config.py or .env file.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
        verify_ssl: Optional[bool] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Ollama embeddings adapter.

        Args:
            base_url: Ollama server base URL (defaults to settings.ollama_base_url)
            model_name: Model name for embeddings (defaults to settings.ollama_embedding_model)
            timeout: Request timeout in seconds (defaults to settings.ollama_timeout)
            verify_ssl: Whether to verify SSL certificates (defaults to settings.ollama_verify_ssl)
            api_key: Optional API key for authentication (defaults to settings.ollama_api_key)
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model_name = model_name or settings.ollama_embedding_model
        self.timeout = timeout or settings.ollama_timeout
        self.verify_ssl = verify_ssl if verify_ssl is not None else settings.ollama_verify_ssl
        self.api_key = api_key or settings.ollama_api_key

        # Ensure base_url doesn't end with /
        self.base_url = self.base_url.rstrip("/")

        # Setup session with retries
        self.session = self._create_session()

        logger.info(
            f"Initialized OllamaEmbeddings with base_url={self.base_url}, "
            f"model={self.model_name}"
        )

    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry logic.

        Returns:
            requests.Session: Configured session with retry strategy
        """
        session = requests.Session()

        # Configure retry strategy for transient errors
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to Ollama API with error handling.

        Args:
            endpoint: API endpoint (e.g., '/api/embed')
            payload: Request payload

        Returns:
            Dict[str, Any]: Response JSON

        Raises:
            OllamaEmbeddingsError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            error_msg = f"Request to {url} timed out after {self.timeout}s"
            logger.error(error_msg)
            raise OllamaEmbeddingsError(error_msg)

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Failed to connect to {url}: {str(e)}"
            logger.error(error_msg)
            raise OllamaEmbeddingsError(error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error from {url}: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise OllamaEmbeddingsError(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error calling {url}: {str(e)}"
            logger.error(error_msg)
            raise OllamaEmbeddingsError(error_msg)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query text.

        Args:
            text: Query text to embed

        Returns:
            np.ndarray: Embedding vector

        Raises:
            OllamaEmbeddingsError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        logger.debug(f"Generating embedding for query (length: {len(text)})")

        payload = {
            "model": self.model_name,
            "prompt": text
        }

        response = self._make_request("/api/embeddings", payload)

        if "embedding" not in response:
            raise OllamaEmbeddingsError("Response missing 'embedding' field")

        embedding = np.array(response["embedding"], dtype=np.float32)
        logger.debug(f"Generated embedding with shape {embedding.shape}")

        return embedding

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        This method processes texts one at a time to handle rate limiting.
        For production use, consider implementing batching if the Ollama API supports it.

        Args:
            texts: List of texts to embed

        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), embedding_dim)

        Raises:
            OllamaEmbeddingsError: If embedding generation fails
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        logger.info(f"Generating embeddings for {len(texts)} texts")

        embeddings = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                logger.warning(f"Skipping empty text at index {i}")
                # Use zero vector for empty texts
                if embeddings:
                    embeddings.append(np.zeros_like(embeddings[0]))
                else:
                    # If this is the first text and it's empty, we need a dimension
                    # This shouldn't normally happen, but handle gracefully
                    raise ValueError(f"First text at index {i} is empty")
                continue

            try:
                embedding = self.embed_query(text)
                embeddings.append(embedding)

                # Simple rate limiting: small delay between requests
                if i < len(texts) - 1:  # Don't sleep after the last request
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to embed text at index {i}: {str(e)}")
                raise

        embeddings_array = np.vstack(embeddings)
        logger.info(f"Generated embeddings with shape {embeddings_array.shape}")

        return embeddings_array

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.

        Returns:
            int: Embedding dimension
        """
        test_embedding = self.embed_query("test")
        return len(test_embedding)
