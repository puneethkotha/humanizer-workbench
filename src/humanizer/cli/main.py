"""
CLI interface for humanizer-workbench.

Design principles:
  - Sensible defaults: running `humanizer input.txt` should produce useful output
    without requiring any flags.
  - Progressive disclosure: basic options first, advanced options available but
    not in the way.
  - Rich output for terminals, plain output when piped (respects NO_COLOR).
  - Errors are actionable: tell the user what to do, not just what went wrong.

The CLI is intentionally thin — it translates command-line arguments into
HumanizerEngine calls and formats the output. No business logic here.
"""

import difflib
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from humanizer.core.engine import HumanizerEngine
from humanizer.core.models import HumanizerResult, Intensity, StyleName
from humanizer.detectors.base import CompositeDetector
from humanizer.detectors.lexical import LexicalDetector
from humanizer.detectors.structural import StructuralDetector
from humanizer.scoring.scorer import AIScorer

console = Console()
err_console = Console(stderr=True)


def _build_diff(original: str, output: str) -> list[str]:
    """Generate a unified diff between original and output text."""
    original_lines = original.splitlines(keepends=True)
    output_lines = output.splitlines(keepends=True)
    return list(
        difflib.unified_diff(
            original_lines,
            output_lines,
            fromfile="original",
            tofile="humanized",
            lineterm="",
        )
    )


def _render_diff(diff_lines: list[str]) -> None:
    """Render a diff with color highlighting using Rich."""
    if not diff_lines:
        console.print("[dim]No changes detected.[/dim]")
        return

    console.print(Rule("Diff", style="dim"))
    for line in diff_lines:
        if line.startswith("---") or line.startswith("+++"):
            console.print(f"[dim]{line}[/dim]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{line}[/cyan]")
        elif line.startswith("+"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{line}[/red]")
        else:
            console.print(f"[dim]{line}[/dim]")


def _render_scores(result: HumanizerResult, verbose: bool = False) -> None:
    """Render before/after AI-likeness scores as a Rich table."""
    scorer = AIScorer()

    table = Table(
        title="AI-Likeness Scores",
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
    )
    table.add_column("", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Grade")

    before_grade = scorer.grade(result.before_score)
    after_grade = scorer.grade(result.after_score)
    improvement = result.improvement

    before_color = _score_color(result.before_score)
    after_color = _score_color(result.after_score)

    table.add_row(
        "Before",
        f"[{before_color}]{result.before_score:.0f}/100[/{before_color}]",
        f"[{before_color}]{before_grade}[/{before_color}]",
    )
    table.add_row(
        "After",
        f"[{after_color}]{result.after_score:.0f}/100[/{after_color}]",
        f"[{after_color}]{after_grade}[/{after_color}]",
    )

    improvement_sign = "+" if improvement < 0 else "-"
    improvement_color = "green" if improvement > 0 else ("red" if improvement < 0 else "dim")
    table.add_row(
        "Improvement",
        f"[{improvement_color}]{abs(improvement):.0f} pts[/{improvement_color}]",
        f"[{improvement_color}]{improvement_sign}{result.improvement_pct:.0f}%[/{improvement_color}]",
    )

    console.print(table)

    if verbose and result.changes_summary:
        console.print()
        console.print("[bold]Changes made:[/bold]")
        for change in result.changes_summary:
            console.print(f"  [dim]·[/dim] {change}")


def _score_color(score: float) -> str:
    if score >= 60:
        return "red"
    elif score >= 35:
        return "yellow"
    else:
        return "green"


def _render_result(
    result: HumanizerResult,
    *,
    show_diff: bool,
    show_scores: bool,
    explain: bool,
    output_file: Path | None,
) -> None:
    """Render the humanization result to terminal and/or file."""

    if output_file:
        output_file.write_text(result.output, encoding="utf-8")
        console.print(f"[dim]Output written to {output_file}[/dim]")
    else:
        console.print()
        console.print(Panel(result.output, title="Humanized", border_style="green"))

    if show_scores:
        console.print()
        _render_scores(result, verbose=explain)

    if show_diff:
        console.print()
        diff = _build_diff(result.original, result.output)
        _render_diff(diff)
    elif explain and not show_diff and result.changes_summary:
        console.print()
        console.print("[bold]Key changes:[/bold]")
        for change in result.changes_summary:
            console.print(f"  [dim]·[/dim] {change}")


@click.command(name="humanizer", no_args_is_help=True)
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--style",
    "-s",
    type=click.Choice([s.value for s in StyleName], case_sensitive=False),
    default=StyleName.PROFESSIONAL.value,
    show_default=True,
    help="Writing style preset to apply.",
)
@click.option(
    "--intensity",
    "-i",
    type=click.Choice([i.value for i in Intensity], case_sensitive=False),
    default=Intensity.MEDIUM.value,
    show_default=True,
    help="Transformation intensity: light preserves structure, aggressive rewrites freely.",
)
@click.option(
    "--diff",
    "show_diff",
    is_flag=True,
    default=False,
    help="Show a before/after diff.",
)
@click.option(
    "--explain",
    is_flag=True,
    default=False,
    help="Explain the key changes made.",
)
@click.option(
    "--score",
    "show_scores",
    is_flag=True,
    default=False,
    help="Show AI-likeness scores before and after.",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Write output to a file instead of stdout.",
)
@click.option(
    "--model",
    default=None,
    help="Claude model to use (default: claude-sonnet-4-6).",
    hidden=True,
)
@click.option(
    "--api-key",
    envvar="ANTHROPIC_API_KEY",
    default=None,
    help="Anthropic API key. Defaults to ANTHROPIC_API_KEY environment variable.",
)
@click.version_option(package_name="humanizer-workbench")
def cli(
    input_file: Path,
    style: str,
    intensity: str,
    show_diff: bool,
    explain: bool,
    show_scores: bool,
    output_file: Path | None,
    model: str | None,
    api_key: str | None,
) -> None:
    """Transform AI-generated text into natural, human-like writing.

    INPUT_FILE is the path to a plain text file. Reads from stdin if INPUT_FILE is '-'.

    Examples:

      humanizer input.txt

      humanizer input.txt --style founder --intensity aggressive

      humanizer input.txt --diff --score

      humanizer input.txt --style technical --output result.txt
    """
    # Read input
    try:
        if str(input_file) == "-":
            text = sys.stdin.read()
        else:
            text = input_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        err_console.print(f"[red]Error reading {input_file}: {e}[/red]")
        sys.exit(1)

    text = text.strip()
    if not text:
        err_console.print("[red]Input file is empty.[/red]")
        sys.exit(1)

    # Build engine
    engine_kwargs: dict[str, str] = {}
    if model:
        engine_kwargs["model"] = model
    if api_key:
        engine_kwargs["api_key"] = api_key

    try:
        engine = HumanizerEngine(**engine_kwargs)
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        err_console.print("[dim]Set ANTHROPIC_API_KEY or pass --api-key to use this tool.[/dim]")
        sys.exit(1)

    # Display run info
    style_name = StyleName(style)
    intensity_level = Intensity(intensity)

    console.print(f"[dim]Style: {style} · Intensity: {intensity} · Model: {engine.model}[/dim]")

    # Run humanization
    with console.status("[bold green]Humanizing...[/bold green]"):
        try:
            result = engine.humanize(
                text=text,
                style=style_name,
                intensity=intensity_level,
            )
        except Exception as e:
            err_console.print(f"[red]Humanization failed: {e}[/red]")
            sys.exit(1)

    # Render result
    _render_result(
        result,
        show_diff=show_diff,
        show_scores=show_scores,
        explain=explain,
        output_file=output_file,
    )


@click.command(name="detect")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
def detect_command(input_file: Path) -> None:
    """Analyze a file for AI-like patterns without transforming it.

    Useful for checking a piece of writing before deciding whether to humanize it.
    """
    try:
        text = input_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        err_console.print(f"[red]Error reading {input_file}: {e}[/red]")
        sys.exit(1)

    detector = CompositeDetector([LexicalDetector(), StructuralDetector()])
    scorer = AIScorer()

    detection = detector.detect(text)
    score = scorer.score(text, detection)
    grade = scorer.grade(score)
    components = scorer.describe_components(text, detection)

    score_color = _score_color(score)

    console.print()
    console.print(
        Panel(
            f"[{score_color}][bold]{score:.0f}/100[/bold] — {grade}[/{score_color}]",
            title=f"AI-likeness: {input_file.name}",
            border_style=score_color,
        )
    )

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Component")
    table.add_column("Score", justify="right")
    table.add_column("Max", justify="right", style="dim")

    component_max = {
        "vocabulary_density": 30,
        "filler_phrases": 25,
        "sentence_uniformity": 20,
        "structural_patterns": 15,
        "ai_openers": 10,
    }

    for name, comp_score in components.items():
        label = name.replace("_", " ").title()
        max_val = component_max.get(name, "?")
        comp_color = _score_color(comp_score * 2)  # normalize to 100 for color
        table.add_row(label, f"[{comp_color}]{comp_score:.1f}[/{comp_color}]", str(max_val))

    console.print(table)

    if detection.ai_vocabulary_hits:
        console.print()
        console.print("[bold]AI vocabulary found:[/bold]")
        vocab_text = Text()
        for i, word in enumerate(detection.ai_vocabulary_hits):
            if i > 0:
                vocab_text.append(", ", style="dim")
            vocab_text.append(word, style="yellow")
        console.print(f"  {vocab_text}")

    if detection.filler_phrase_hits:
        console.print()
        console.print("[bold]Filler phrases found:[/bold]")
        for phrase in detection.filler_phrase_hits:
            console.print(f'  [dim]·[/dim] [yellow]"{phrase}"[/yellow]')

    if detection.structural_flags:
        console.print()
        console.print("[bold]Structural patterns:[/bold]")
        for flag in detection.structural_flags:
            console.print(f"  [dim]·[/dim] {flag.replace('_', ' ')}")


# Entry points (see pyproject.toml):
#   humanizer        → cli (transform command)
#   humanizer-detect → detect_command
