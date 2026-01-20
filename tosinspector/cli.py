"""CLI interface for ToS Inspector using Typer."""

import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from tosinspector.config import settings
from tosinspector.utils import logger
from tosinspector.indexer import Indexer, IndexManifest
from tosinspector.qa import QAEngine

app = typer.Typer(
    name="tosinspector",
    help="CLI tool to index and query Terms of Service documents using LLM",
    add_completion=False
)

console = Console()


@app.command()
def index(
    file: str = typer.Argument(..., help="Path to the ToS document file (text/HTML/PDF)"),
    name: str = typer.Option(..., "--name", "-n", help="Name for the index"),
    chunk_size: Optional[int] = typer.Option(
        None, "--chunk-size", help="Chunk size in characters"
    ),
    chunk_overlap: Optional[int] = typer.Option(
        None, "--chunk-overlap", help="Chunk overlap in characters"
    ),
) -> None:
    """
    Index a Terms of Service document.

    Creates an in-memory vector index from the document and saves a manifest
    to disk for later retrieval.

    Example:
        tosinspector index sample_tos.pdf --name my_tos
    """
    try:
        console.print(f"\n[bold blue]Indexing document:[/bold blue] {file}")
        console.print(f"[bold blue]Index name:[/bold blue] {name}\n")

        # Validate file exists
        if not os.path.exists(file):
            console.print(f"[bold red]Error:[/bold red] File not found: {file}")
            raise typer.Exit(1)

        # Create indexer
        indexer = Indexer(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Create index
        with console.status("[bold green]Processing document..."):
            vector_index, manifest = indexer.create_index(
                file_path=file,
                index_name=name,
                save_manifest=True
            )

        # Display summary
        console.print("[bold green]✓[/bold green] Index created successfully!\n")

        table = Table(title="Index Summary")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Index Name", manifest.index_name)
        table.add_row("Source File", manifest.source_file)
        table.add_row("Embedding Model", manifest.embedding_model)
        table.add_row("Chunk Size", str(manifest.chunk_size))
        table.add_row("Chunk Overlap", str(manifest.chunk_overlap))
        table.add_row("Number of Chunks", str(manifest.num_chunks))
        table.add_row("Created At", manifest.created_at)

        console.print(table)
        console.print(f"\n[dim]Manifest saved to: {settings.index_dir}/{name}/manifest.json[/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        logger.exception("Error during indexing")
        raise typer.Exit(1)


@app.command()
def query(
    index_name: str = typer.Argument(..., help="Name of the index to query"),
    question: str = typer.Option(..., "--question", "-q", help="Question to ask"),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", "-k", help="Number of chunks to retrieve"
    ),
    show_sources: bool = typer.Option(
        True, "--show-sources/--no-sources", help="Show source citations"
    ),
) -> None:
    """
    Query an indexed ToS document.

    Retrieves relevant chunks and generates an answer with citations.

    Example:
        tosinspector query my_tos --question "What is the refund policy?"
    """
    try:
        console.print(f"\n[bold blue]Querying index:[/bold blue] {index_name}")
        console.print(f"[bold blue]Question:[/bold blue] {question}\n")

        # Load index
        with console.status("[bold green]Loading index..."):
            vector_index, manifest = Indexer.load_index(index_name)

        # Create QA engine
        qa_engine = QAEngine(vector_index, top_k=top_k)

        # Query
        with console.status("[bold green]Generating answer..."):
            result = qa_engine.query(question)

        # Display answer
        answer_panel = Panel(
            result["answer"],
            title="[bold green]Answer[/bold green]",
            border_style="green"
        )
        console.print(answer_panel)

        # Display metadata
        console.print(f"\n[dim]Chunks used: {result['chunks_used']}/{len(result['sources'])}[/dim]")
        if result["truncated"]:
            console.print("[dim yellow]⚠ Context was truncated due to length limits[/dim yellow]")

        # Display sources
        if show_sources and result["sources"]:
            console.print("\n[bold cyan]Sources:[/bold cyan]\n")

            sources_table = Table(show_header=True)
            sources_table.add_column("Chunk ID", style="cyan")
            sources_table.add_column("Source", style="white")
            sources_table.add_column("Page", style="yellow")
            sources_table.add_column("Char Range", style="magenta")

            for source in result["sources"]:
                page = str(source["page"]) if source["page"] is not None else "-"
                char_range = (
                    f"{source['start_char']}-{source['end_char']}"
                    if source["start_char"] is not None
                    else "-"
                )

                sources_table.add_row(
                    source["chunk_id"],
                    source["source"] or "-",
                    page,
                    char_range
                )

            console.print(sources_table)

        console.print()

    except FileNotFoundError as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        console.print(f"[dim]Available indexes are stored in: {settings.index_dir}/[/dim]\n")
        raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        logger.exception("Error during query")
        raise typer.Exit(1)


@app.command()
def summarize(
    index_name: str = typer.Argument(..., help="Name of the index to summarize"),
) -> None:
    """
    Generate a summary of an indexed ToS document.

    Example:
        tosinspector summarize my_tos
    """
    try:
        console.print(f"\n[bold blue]Summarizing index:[/bold blue] {index_name}\n")

        # Load index
        with console.status("[bold green]Loading index..."):
            vector_index, manifest = Indexer.load_index(index_name)

        # Create QA engine
        qa_engine = QAEngine(vector_index)

        # Generate summary
        with console.status("[bold green]Generating summary..."):
            summary = qa_engine.summarize()

        # Display summary
        summary_panel = Panel(
            summary,
            title=f"[bold green]Summary of {manifest.source_file}[/bold green]",
            border_style="green"
        )
        console.print(summary_panel)
        console.print()

    except FileNotFoundError as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        logger.exception("Error during summarization")
        raise typer.Exit(1)


@app.command()
def list_sources(
    index_name: str = typer.Argument(..., help="Name of the index"),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Maximum number of chunks to display"
    ),
) -> None:
    """
    List all chunks and their metadata from an index.

    Example:
        tosinspector list-sources my_tos --limit 20
    """
    try:
        console.print(f"\n[bold blue]Listing sources for index:[/bold blue] {index_name}\n")

        # Load index
        with console.status("[bold green]Loading index..."):
            vector_index, manifest = Indexer.load_index(index_name)

        # Get all chunks
        all_chunks = vector_index.get_all_chunks()

        chunk_ids = all_chunks["ids"]
        documents = all_chunks["documents"]
        metadatas = all_chunks["metadatas"]

        total_chunks = len(chunk_ids)
        display_limit = limit if limit is not None else total_chunks

        console.print(f"[bold]Total chunks:[/bold] {total_chunks}")
        if limit and limit < total_chunks:
            console.print(f"[dim]Showing first {limit} chunks[/dim]\n")
        else:
            console.print()

        # Display chunks
        for i, (chunk_id, doc, metadata) in enumerate(
            zip(chunk_ids[:display_limit], documents[:display_limit], metadatas[:display_limit])
        ):
            # Format metadata
            source = metadata.get("source", "unknown")
            page = metadata.get("page")
            start_char = metadata.get("start_char")
            end_char = metadata.get("end_char")

            meta_str = f"Source: {source}"
            if page is not None:
                meta_str += f" | Page: {page}"
            if start_char is not None and end_char is not None:
                meta_str += f" | Chars: {start_char}-{end_char}"

            # Truncate document preview
            doc_preview = doc[:200] + "..." if len(doc) > 200 else doc

            console.print(f"[bold cyan]{chunk_id}[/bold cyan]")
            console.print(f"[dim]{meta_str}[/dim]")
            console.print(f"{doc_preview}\n")

        if limit and limit < total_chunks:
            console.print(f"[dim]... and {total_chunks - limit} more chunks[/dim]\n")

    except FileNotFoundError as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        logger.exception("Error listing sources")
        raise typer.Exit(1)


@app.command()
def list_indexes() -> None:
    """
    List all available indexes.

    Example:
        tosinspector list-indexes
    """
    try:
        console.print("\n[bold blue]Available indexes:[/bold blue]\n")

        index_dir = settings.index_dir

        if not os.path.exists(index_dir):
            console.print("[dim]No indexes found. Create one with 'tosinspector index'[/dim]\n")
            return

        # Find all manifest files
        indexes = []
        for item in os.listdir(index_dir):
            item_path = os.path.join(index_dir, item)
            manifest_path = os.path.join(item_path, "manifest.json")

            if os.path.isdir(item_path) and os.path.exists(manifest_path):
                try:
                    manifest = IndexManifest.load(item_path)
                    indexes.append(manifest)
                except Exception as e:
                    logger.warning(f"Failed to load manifest for {item}: {e}")

        if not indexes:
            console.print("[dim]No indexes found. Create one with 'tosinspector index'[/dim]\n")
            return

        # Display table
        table = Table(show_header=True)
        table.add_column("Index Name", style="cyan")
        table.add_column("Source File", style="white")
        table.add_column("Chunks", style="yellow")
        table.add_column("Model", style="magenta")
        table.add_column("Created", style="green")

        for manifest in indexes:
            table.add_row(
                manifest.index_name,
                manifest.source_file,
                str(manifest.num_chunks),
                manifest.embedding_model,
                manifest.created_at.split("T")[0]  # Just show date
            )

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        logger.exception("Error listing indexes")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
