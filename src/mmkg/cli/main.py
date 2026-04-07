from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from mmkg import __version__
from mmkg.schemas.core import GraphDocument
from mmkg.utils.config import load_config
from mmkg.utils.logging_utils import configure_logging

console = Console()


@click.group()
@click.option("--config", default=None)
@click.option("--log-level", default="INFO")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], log_level: str) -> None:
    """MMKG — Multimodal Incremental Knowledge Graph pipeline."""
    configure_logging(log_level.upper())
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Print project version and config summary."""
    cfg = ctx.obj["config"]
    console.rule("[bold blue]MMKG — Project Info[/bold blue]")
    console.print(f"[bold]Version:[/bold] {__version__}")
    console.print(f"[bold]Primary extractor:[/bold] {cfg['extraction']['primary']}")
    console.print(f"[bold]CLIP threshold:[/bold] {cfg['fusion']['clip_similarity_threshold']}")
    console.print(f"[bold]MKGformer enabled:[/bold] {cfg['mkgformer']['enabled']}")
    console.rule()
    console.print("[green]Config loaded successfully.[/green]")


@cli.command()
@click.option("--doc-id", required=True)
@click.option("--text", default=None)
@click.option("--text-file", default=None, type=click.Path(exists=True))
@click.option("--images", default=None, multiple=True)
@click.pass_context
def ingest(ctx: click.Context, doc_id: str, text: Optional[str],
           text_file: Optional[str], images: tuple) -> None:
    """Run the full pipeline on one document."""
    from mmkg.pipeline.ingest import IngestPipeline
    if text_file:
        text = Path(text_file).read_text()
    if not text:
        console.print("[red]Provide --text or --text-file[/red]")
        sys.exit(1)
    image_paths = [Path(p) for p in images] if images else None
    pipeline = IngestPipeline(ctx.obj["config"])
    result = pipeline.run(doc_id=doc_id, text=text, image_paths=image_paths)
    console.print(f"\n[green]Ingest complete:[/green] {result.summary()}")


@cli.command()
@click.option("--doc", required=True, type=click.Path(exists=True))
@click.pass_context
def validate(ctx: click.Context, doc: str) -> None:
    """Validate a GraphDocument JSON file."""
    raw = json.loads(Path(doc).read_text())
    try:
        gd = GraphDocument.model_validate(raw)
    except Exception as exc:
        console.print(f"[red]Validation failed:[/red] {exc}")
        sys.exit(1)
    table = Table(title=f"GraphDocument: {gd.doc_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("doc_id", gd.doc_id)
    table.add_row("stage", gd.pipeline_stage)
    table.add_row("entities", str(len(gd.entities)))
    table.add_row("relations", str(len(gd.relations)))
    table.add_row("evidences", str(len(gd.evidences)))
    console.print(table)
    console.print(f"[green]{gd.summary()}[/green]")
