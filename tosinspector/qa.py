"""QA module for question answering with prompt engineering and citation support."""

from typing import Dict, Any, List, Optional
import requests

from tosinspector.config import settings
from tosinspector.utils import logger
from tosinspector.indexer import VectorIndex


# Prompt template for QA
# This is the exact template used to construct prompts for Ollama generation
SYSTEM_INSTRUCTION = """You are an assistant that answers questions using only the provided context. If the answer is not present in the context, reply exactly: "Not in document." Always provide citations in the form [chunk-id] for any information you use from the context."""

PROMPT_TEMPLATE = """{system_instruction}

CONTEXT:
{context}

QUESTION: {question}

Provide a concise answer with citations in the form [chunk-id]. If the answer is not in the context, respond with: "Not in document."

ANSWER:"""


class OllamaGenerator:
    """
    Generator for text generation using Ollama's /api/generate endpoint.

    Handles prompt construction, token limit management, and error handling.
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
        Initialize Ollama generator.

        Args:
            base_url: Ollama server base URL
            model_name: Model name for generation
            timeout: Request timeout
            verify_ssl: Whether to verify SSL
            api_key: Optional API key
        """
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model_name = model_name or settings.ollama_generation_model
        self.timeout = timeout or settings.ollama_timeout
        self.verify_ssl = verify_ssl if verify_ssl is not None else settings.ollama_verify_ssl
        self.api_key = api_key or settings.ollama_api_key

        logger.info(
            f"Initialized OllamaGenerator with base_url={self.base_url}, "
            f"model={self.model_name}"
        )

    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to send to the model

        Returns:
            str: Generated text

        Raises:
            Exception: If generation fails
        """
        url = f"{self.base_url}/api/generate"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            logger.debug(f"Sending generate request to {url}")

            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            response.raise_for_status()

            result = response.json()

            if "response" not in result:
                raise Exception("Response missing 'response' field")

            generated_text = result["response"]
            logger.debug(f"Generated text (length: {len(generated_text)})")

            return generated_text

        except requests.exceptions.Timeout:
            error_msg = f"Request to {url} timed out after {self.timeout}s"
            logger.error(error_msg)
            raise Exception(error_msg)

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Failed to connect to {url}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error from {url}: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)

        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            logger.error(error_msg)
            raise


class QAEngine:
    """
    Question-answering engine that retrieves context and generates answers.

    This class orchestrates:
    1. Retrieving relevant chunks from the vector index
    2. Constructing a prompt with context and citations
    3. Generating an answer using Ollama
    4. Returning the answer with source metadata
    """

    def __init__(
        self,
        index: VectorIndex,
        generator: Optional[OllamaGenerator] = None,
        top_k: Optional[int] = None,
        max_context_tokens: Optional[int] = None
    ):
        """
        Initialize QA engine.

        Args:
            index: VectorIndex to query
            generator: OllamaGenerator instance (creates new if not provided)
            top_k: Number of chunks to retrieve
            max_context_tokens: Maximum tokens in context (approximated by chars/4)
        """
        self.index = index
        self.generator = generator or OllamaGenerator()
        self.top_k = top_k or settings.top_k
        self.max_context_tokens = max_context_tokens or settings.max_context_tokens

        # Rough approximation: 1 token ≈ 4 characters
        self.max_context_chars = self.max_context_tokens * 4

        logger.info(
            f"Initialized QAEngine with top_k={self.top_k}, "
            f"max_context_tokens={self.max_context_tokens}"
        )

    def _format_context(
        self,
        chunks: List[str],
        chunk_ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> str:
        """
        Format retrieved chunks into context string with metadata.

        Args:
            chunks: List of chunk texts
            chunk_ids: List of chunk IDs
            metadatas: List of metadata dicts

        Returns:
            str: Formatted context string
        """
        context_parts = []
        total_chars = 0
        truncated = False

        for chunk_text, chunk_id, metadata in zip(chunks, chunk_ids, metadatas):
            # Format chunk with metadata
            source = metadata.get("source", "unknown")
            page = metadata.get("page")
            start_char = metadata.get("start_char")
            end_char = metadata.get("end_char")

            # Build metadata string
            meta_str = f"Source: {source}"
            if page is not None:
                meta_str += f", Page: {page}"
            if start_char is not None and end_char is not None:
                meta_str += f", Chars: {start_char}-{end_char}"

            chunk_context = f"[{chunk_id}] ({meta_str})\n{chunk_text}\n"

            # Check if adding this chunk would exceed the limit
            if total_chars + len(chunk_context) > self.max_context_chars:
                truncated = True
                logger.warning(
                    f"Context truncated: reached {self.max_context_chars} char limit "
                    f"after {len(context_parts)} chunks"
                )
                break

            context_parts.append(chunk_context)
            total_chars += len(chunk_context)

        context = "\n".join(context_parts)

        if truncated:
            context += "\n[Note: Additional context was truncated due to length limits]"

        return context

    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the indexed documents.

        Args:
            question: The question to answer

        Returns:
            Dict containing:
                - answer: Generated answer text
                - sources: List of source metadata dicts
                - chunks_used: Number of chunks used in context
                - truncated: Whether context was truncated
        """
        logger.info(f"Processing question: {question}")

        # Retrieve relevant chunks
        retrieval_results = self.index.query(question, top_k=self.top_k)

        chunks = retrieval_results["documents"]
        chunk_ids = retrieval_results["ids"]
        metadatas = retrieval_results["metadatas"]

        if not chunks:
            logger.warning("No chunks retrieved for question")
            return {
                "answer": "Not in document.",
                "sources": [],
                "chunks_used": 0,
                "truncated": False
            }

        # Format context
        context = self._format_context(chunks, chunk_ids, metadatas)

        # Count chunks actually used (before truncation)
        chunks_used = context.count("[chunk_")
        truncated = "truncated due to length limits" in context

        # Construct prompt
        prompt = PROMPT_TEMPLATE.format(
            system_instruction=SYSTEM_INSTRUCTION,
            context=context,
            question=question
        )

        logger.debug(f"Prompt length: {len(prompt)} chars")

        # Generate answer
        answer = self.generator.generate(prompt)

        # Prepare source information
        sources = []
        for chunk_id, metadata in zip(chunk_ids, metadatas):
            sources.append({
                "chunk_id": chunk_id,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "start_char": metadata.get("start_char"),
                "end_char": metadata.get("end_char")
            })

        return {
            "answer": answer.strip(),
            "sources": sources,
            "chunks_used": chunks_used,
            "truncated": truncated
        }

    def summarize(self) -> str:
        """
        Generate a summary of the indexed document.

        Uses a special prompt to create a concise summary based on
        a sample of chunks from the document.

        Returns:
            str: Summary text
        """
        logger.info("Generating summary of indexed document")

        # Get a sample of chunks (first few chunks give a good overview)
        all_chunks = self.index.get_all_chunks()

        # Take up to 10 chunks for summary
        num_chunks = min(10, len(all_chunks["documents"]))
        sample_chunks = all_chunks["documents"][:num_chunks]

        # Combine chunks for context
        context = "\n\n".join(sample_chunks)

        # Truncate if too long
        if len(context) > self.max_context_chars:
            context = context[:self.max_context_chars] + "..."

        # Create summary prompt
        summary_prompt = f"""Provide a concise summary of the following Terms of Service document. Focus on the key points, user rights, and important obligations.

DOCUMENT EXCERPT:
{context}

SUMMARY:"""

        # Generate summary
        summary = self.generator.generate(summary_prompt)

        return summary.strip()
