"""Document ingestion module for loading and chunking various file formats."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pypdf import PdfReader
from bs4 import BeautifulSoup

from tosinspector.config import settings
from tosinspector.utils import logger


@dataclass
class DocumentChunk:
    """
    Represents a chunk of a document with metadata.

    Attributes:
        id: Unique identifier for the chunk
        text: The text content of the chunk
        metadata: Dictionary containing source filename, page number, char offsets, etc.
    """
    id: str
    text: str
    metadata: Dict[str, Any]


class DocumentLoader:
    """
    Load documents from various formats (text, HTML, PDF).

    Supports:
    - Plain text files (.txt)
    - HTML files (.html, .htm)
    - PDF files (.pdf)
    """

    @staticmethod
    def load_text(file_path: str) -> str:
        """
        Load plain text file.

        Args:
            file_path: Path to the text file

        Returns:
            str: File contents
        """
        logger.info(f"Loading text file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"File {file_path} is empty")

            return content

        except UnicodeDecodeError:
            # Try with a different encoding
            logger.warning(f"UTF-8 decoding failed, trying latin-1 for {file_path}")
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    @staticmethod
    def load_html(file_path: str) -> str:
        """
        Load HTML file and extract text content.

        Args:
            file_path: Path to the HTML file

        Returns:
            str: Extracted text content
        """
        logger.info(f"Loading HTML file: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text

    @staticmethod
    def load_pdf(file_path: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Load PDF file and extract text with page information.

        Args:
            file_path: Path to the PDF file

        Returns:
            tuple: (full_text, page_metadata_list)
                - full_text: All text concatenated
                - page_metadata_list: List of dicts with page numbers and char offsets
        """
        logger.info(f"Loading PDF file: {file_path}")

        try:
            reader = PdfReader(file_path)

            if len(reader.pages) == 0:
                logger.warning(f"PDF {file_path} has no pages")
                return "", []

            full_text = ""
            page_metadata = []
            char_offset = 0

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()

                if page_text:
                    page_metadata.append({
                        "page": page_num,
                        "start_char": char_offset,
                        "end_char": char_offset + len(page_text)
                    })

                    full_text += page_text
                    char_offset += len(page_text)

            logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")

            return full_text, page_metadata

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    @classmethod
    def load(cls, file_path: str) -> tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Load document based on file extension.

        Args:
            file_path: Path to the document

        Returns:
            tuple: (text_content, optional_page_metadata)

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = Path(file_path).suffix.lower()

        if file_ext == ".pdf":
            return cls.load_pdf(file_path)
        elif file_ext in [".html", ".htm"]:
            return cls.load_html(file_path), None
        elif file_ext in [".txt", ""]:
            return cls.load_text(file_path), None
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")


class DocumentChunker:
    """
    Chunk documents into smaller pieces with metadata.

    Uses character-based chunking with overlap to ensure context preservation.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize document chunker.

        Args:
            chunk_size: Number of characters per chunk (defaults to settings.chunk_size)
            chunk_overlap: Number of overlapping characters (defaults to settings.chunk_overlap)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        logger.info(
            f"Initialized DocumentChunker with chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def chunk_text(
        self,
        text: str,
        source_file: str,
        page_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk text into smaller pieces with metadata.

        Args:
            text: Text to chunk
            source_file: Source filename for metadata
            page_metadata: Optional list of page metadata from PDF

        Returns:
            List[DocumentChunk]: List of document chunks with metadata
        """
        if not text or not text.strip():
            logger.warning("Received empty text for chunking")
            return []

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Don't create empty chunks
            if not chunk_text.strip():
                break

            # Determine page number if page_metadata is provided
            page_num = None
            if page_metadata:
                for page_info in page_metadata:
                    if page_info["start_char"] <= start < page_info["end_char"]:
                        page_num = page_info["page"]
                        break

            metadata = {
                "source": os.path.basename(source_file),
                "source_path": source_file,
                "start_char": start,
                "end_char": min(end, len(text)),
            }

            if page_num is not None:
                metadata["page"] = page_num

            chunk = DocumentChunk(
                id=f"chunk_{chunk_id}",
                text=chunk_text.strip(),
                metadata=metadata
            )

            chunks.append(chunk)
            chunk_id += 1

            # Move to next chunk with overlap
            start = end - self.chunk_overlap

            # Prevent infinite loop
            if start >= len(text):
                break

        logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")

        return chunks


class DocumentIngester:
    """
    High-level interface for loading and chunking documents.

    Combines DocumentLoader and DocumentChunker for easy use.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize document ingester.

        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of overlapping characters
        """
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest(self, file_path: str) -> List[DocumentChunk]:
        """
        Load and chunk a document file.

        Args:
            file_path: Path to the document file

        Returns:
            List[DocumentChunk]: List of document chunks

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        logger.info(f"Ingesting document: {file_path}")

        # Load document
        text, page_metadata = self.loader.load(file_path)

        if not text or not text.strip():
            raise ValueError(f"Document {file_path} is empty or has no extractable text")

        # Chunk document
        chunks = self.chunker.chunk_text(text, file_path, page_metadata)

        logger.info(f"Successfully ingested {file_path}: {len(chunks)} chunks")

        return chunks
