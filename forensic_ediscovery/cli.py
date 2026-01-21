"""
Command-line interface for forensic e-discovery email processing.
Provides commands for export, analysis, and bundle generation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from .config import get_settings, reset_settings
from .gmail_adapter import GmailAdapter, create_gmail_adapters
from .email_parser import EmailParser
from .hasher import ManifestGenerator
from .gemini_extractor import GeminiExtractor, batch_extract
from .bundle_generator import BundleGenerator, create_evidence_bundle
from .models import EmailRecord, AIExtraction

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("--debug", is_flag=True, help="Debug output")
@click.pass_context
def cli(ctx, verbose: bool, debug: bool):
    """Forensic E-Discovery Email Processing System.

    Export, analyze, and bundle email evidence for investigations.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug
    setup_logging(verbose, debug)


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("evidence_output"),
    help="Output directory",
)
@click.option(
    "--credentials", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="OAuth credentials.json file",
)
@click.option(
    "--accounts",
    multiple=True,
    help="Email accounts to process (default: from config)",
)
@click.option(
    "--max-emails",
    type=int,
    default=None,
    help="Maximum emails per account",
)
@click.pass_context
def export(
    ctx,
    output: Path,
    credentials: Optional[Path],
    accounts: tuple,
    max_emails: Optional[int],
):
    """Export emails from Gmail accounts.

    Authenticates via OAuth and downloads matching emails as .eml files.
    """
    settings = get_settings()
    accounts_list = list(accounts) if accounts else settings.email_accounts

    console.print(f"[bold blue]Exporting from {len(accounts_list)} account(s)[/bold blue]")
    console.print(f"Output directory: {output}")

    output.mkdir(parents=True, exist_ok=True)

    all_records: List[EmailRecord] = []

    for account in accounts_list:
        console.print(f"\n[bold]Processing: {account}[/bold]")

        try:
            adapter = GmailAdapter(
                account=account,
                credentials_file=credentials,
                token_dir=output / "tokens",
            )

            # Authenticate
            with console.status(f"Authenticating {account}..."):
                adapter.authenticate()
            console.print(f"[green]✓[/green] Authenticated")

            # Search for messages
            with console.status("Searching for matching messages..."):
                message_ids = adapter.search_messages(max_results=max_emails)
            console.print(f"[green]✓[/green] Found {len(message_ids)} matching messages")

            if not message_ids:
                continue

            # Export messages
            email_dir = output / "emails"
            parser = EmailParser(email_dir)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Exporting {account}", total=len(message_ids))

                for msg_id, raw_bytes, metadata in adapter.export_messages(email_dir):
                    record = parser.parse_raw_email(
                        raw_bytes=raw_bytes,
                        gmail_id=metadata["id"],
                        thread_id=metadata["threadId"],
                        internal_date=metadata["internalDate"],
                        source_account=account,
                    )
                    all_records.append(record)
                    progress.advance(task)

            console.print(f"[green]✓[/green] Exported {len(message_ids)} emails from {account}")

        except Exception as e:
            console.print(f"[red]✗[/red] Error processing {account}: {e}")
            if ctx.obj.get("debug"):
                raise

    # Save records summary
    summary_file = output / "export_summary.json"
    import json
    with open(summary_file, "w") as f:
        json.dump({
            "total_emails": len(all_records),
            "accounts": accounts_list,
            "email_ids": [r.gmail_id for r in all_records],
        }, f, indent=2)

    console.print(f"\n[bold green]Export complete![/bold green]")
    console.print(f"Total emails: {len(all_records)}")
    console.print(f"Summary: {summary_file}")


@cli.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input directory with exported emails",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for extractions",
)
@click.option(
    "--model",
    type=click.Choice(["flash", "pro"]),
    default="flash",
    help="Gemini model to use",
)
@click.option(
    "--api-key",
    envvar="GEMINI_API_KEY",
    help="Gemini API key",
)
@click.pass_context
def analyze(
    ctx,
    input: Path,
    output: Optional[Path],
    model: str,
    api_key: Optional[str],
):
    """Analyze emails with Gemini AI.

    Extracts threats, financial references, and key claims.
    """
    output = output or input / "ai_extractions"
    output.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Analyzing emails with Gemini AI[/bold blue]")

    # Load email records
    email_records: List[EmailRecord] = []
    emails_dir = input / "emails"

    if not emails_dir.exists():
        console.print(f"[red]Error: No emails directory found at {emails_dir}[/red]")
        return

    # Find all metadata JSON files
    metadata_files = list(emails_dir.rglob("*_metadata.json"))
    console.print(f"Found {len(metadata_files)} email records")

    for meta_file in metadata_files:
        try:
            import json
            data = json.loads(meta_file.read_text())
            record = EmailRecord.model_validate(data)
            email_records.append(record)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {meta_file}: {e}[/yellow]")

    if not email_records:
        console.print("[red]No valid email records found[/red]")
        return

    # Initialize extractor
    use_pro = model == "pro"
    try:
        extractor = GeminiExtractor(
            api_key=api_key,
            use_reasoning_model=use_pro,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    console.print(f"Using model: {extractor.model_name}")

    # Process emails
    extractions: List[AIExtraction] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing emails", total=len(email_records))

        for record in email_records:
            try:
                extraction = extractor.extract_from_email(record)
                extractor.save_extraction(extraction, output)
                extractions.append(extraction)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to analyze {record.gmail_id}: {e}[/yellow]")

            progress.advance(task)

    # Summary
    threats_found = sum(len(e.threat_indicators) for e in extractions)
    financial_refs = sum(len(e.financial_references) for e in extractions)
    needs_review = sum(1 for e in extractions if e.requires_human_review)

    console.print(f"\n[bold green]Analysis complete![/bold green]")
    console.print(f"Emails analyzed: {len(extractions)}")
    console.print(f"Threat indicators: {threats_found}")
    console.print(f"Financial references: {financial_refs}")
    console.print(f"Requires human review: {needs_review}")


@cli.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input directory with emails and extractions",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for bundle",
)
@click.option(
    "--name",
    default="evidence_bundle",
    help="Bundle name",
)
@click.pass_context
def bundle(
    ctx,
    input: Path,
    output: Optional[Path],
    name: str,
):
    """Create evidence bundle with all artifacts.

    Generates manifest, timeline, index, and ZIP archive.
    """
    output = output or input
    output.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Creating evidence bundle[/bold blue]")

    # Load email records
    email_records: List[EmailRecord] = []
    emails_dir = input / "emails"

    if emails_dir.exists():
        for meta_file in emails_dir.rglob("*_metadata.json"):
            try:
                import json
                data = json.loads(meta_file.read_text())
                record = EmailRecord.model_validate(data)
                email_records.append(record)
            except Exception as e:
                logger.warning(f"Could not load {meta_file}: {e}")

    console.print(f"Loaded {len(email_records)} email records")

    # Load AI extractions
    ai_extractions: List[AIExtraction] = []
    extractions_dir = input / "ai_extractions"

    if extractions_dir.exists():
        for ext_file in extractions_dir.glob("*_extraction.json"):
            try:
                import json
                data = json.loads(ext_file.read_text())
                extraction = AIExtraction.model_validate(data)
                ai_extractions.append(extraction)
            except Exception as e:
                logger.warning(f"Could not load {ext_file}: {e}")

    console.print(f"Loaded {len(ai_extractions)} AI extractions")

    # Create bundle
    with console.status("Generating bundle..."):
        manifest, zip_path = create_evidence_bundle(
            email_records=email_records,
            ai_extractions=ai_extractions,
            output_dir=output,
            bundle_name=name,
        )

    console.print(f"\n[bold green]Bundle created![/bold green]")
    console.print(f"Bundle ID: {manifest.bundle_id}")
    console.print(f"Total emails: {manifest.total_emails}")
    console.print(f"Total files: {manifest.total_files}")
    console.print(f"ZIP archive: {zip_path}")


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("evidence_output"),
    help="Output directory",
)
@click.option(
    "--credentials", "-c",
    type=click.Path(exists=True, path_type=Path),
    help="OAuth credentials.json file",
)
@click.option(
    "--max-emails",
    type=int,
    default=None,
    help="Maximum emails per account",
)
@click.option(
    "--api-key",
    envvar="GEMINI_API_KEY",
    help="Gemini API key",
)
@click.option(
    "--skip-export",
    is_flag=True,
    help="Skip export, use existing emails",
)
@click.option(
    "--skip-analyze",
    is_flag=True,
    help="Skip AI analysis",
)
@click.pass_context
def full(
    ctx,
    output: Path,
    credentials: Optional[Path],
    max_emails: Optional[int],
    api_key: Optional[str],
    skip_export: bool,
    skip_analyze: bool,
):
    """Run full pipeline: export → analyze → bundle.

    Complete end-to-end processing of email evidence.
    """
    console.print("[bold blue]Running full evidence pipeline[/bold blue]")
    console.print(f"Output: {output}")

    settings = get_settings()

    # Step 1: Export
    if not skip_export:
        console.print("\n[bold]Step 1: Exporting emails[/bold]")
        ctx.invoke(export, output=output, credentials=credentials, max_emails=max_emails)
    else:
        console.print("\n[bold]Step 1: Skipping export[/bold]")

    # Step 2: Analyze
    if not skip_analyze:
        console.print("\n[bold]Step 2: Analyzing with Gemini[/bold]")
        ctx.invoke(analyze, input=output, api_key=api_key)
    else:
        console.print("\n[bold]Step 2: Skipping analysis[/bold]")

    # Step 3: Bundle
    console.print("\n[bold]Step 3: Creating evidence bundle[/bold]")
    ctx.invoke(bundle, input=output)

    console.print("\n[bold green]Pipeline complete![/bold green]")


@cli.command()
def config():
    """Show current configuration."""
    settings = get_settings()

    table = Table(title="Forensic E-Discovery Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Email Accounts", "\n".join(settings.email_accounts))
    table.add_row("Target Emails", "\n".join(settings.get_all_target_emails()))
    table.add_row("Target Domains", "\n".join(settings.get_all_target_domains()))
    table.add_row("Incident Date", settings.incident_date)
    table.add_row("Output Directory", str(settings.output_dir))
    table.add_row("Gemini Model", settings.gemini.default_model)
    table.add_row("Gmail Query", settings.build_gmail_query()[:100] + "...")

    console.print(table)


def main():
    """Entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
