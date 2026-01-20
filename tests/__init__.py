"""Unit tests for document ingestion module."""

import pytest
from tosinspector.ingestion import DocumentChunker, DocumentChunk


class TestDocumentChunker:
    """Test cases for DocumentChunker."""

    def test_basic_chunking(self) -> None:
        """Test basic text chunking."""
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        text = "This is a test document with some text to chunk into smaller pieces."

        chunks = chunker.chunk_text(text, "test.txt")

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.metadata["source"] == "test.txt" for chunk in chunks)

    def test_chunk_overlap(self) -> None:
        """Test that chunks have proper overlap."""
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        text = "A" * 50

        chunks = chunker.chunk_text(text, "test.txt")

        # With chunk_size=20 and overlap=5, we should have multiple chunks
        assert len(chunks) > 1

        # Check that metadata reflects proper offsets
        for i in range(len(chunks) - 1):
            current_end = chunks[i].metadata["end_char"]
            next_start = chunks[i + 1].metadata["start_char"]
            # The overlap should be 5
            assert current_end - next_start == 5

    def test_empty_text(self) -> None:
        """Test handling of empty text."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

        chunks = chunker.chunk_text("", "test.txt")

        assert len(chunks) == 0

    def test_text_shorter_than_chunk_size(self) -> None:
        """Test text that's shorter than chunk size."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        text = "Short text."

        chunks = chunker.chunk_text(text, "test.txt")

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_chunk_metadata(self) -> None:
        """Test that chunk metadata is properly set."""
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        text = "This is a test document."

        chunks = chunker.chunk_text(text, "test.txt", page_metadata=None)

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "source_path" in chunk.metadata
            assert "start_char" in chunk.metadata
            assert "end_char" in chunk.metadata
            assert chunk.metadata["source"] == "test.txt"

    def test_chunk_with_page_metadata(self) -> None:
        """Test chunking with page metadata."""
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        text = "A" * 100

        page_metadata = [
            {"page": 1, "start_char": 0, "end_char": 50},
            {"page": 2, "start_char": 50, "end_char": 100}
        ]

        chunks = chunker.chunk_text(text, "test.pdf", page_metadata)

        # Check that some chunks have page numbers
        pages_found = [chunk.metadata.get("page") for chunk in chunks]
        assert any(page is not None for page in pages_found)

    def test_invalid_overlap(self) -> None:
        """Test that invalid overlap raises error."""
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=10, chunk_overlap=10)

        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=10, chunk_overlap=15)
