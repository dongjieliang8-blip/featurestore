"""FeatureStore CLI -- multi-agent ML feature engineering pipeline."""

import json
import sys
import click
from rich.console import Console
from rich.panel import Panel

from src.pipeline import Pipeline

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="featurestore")
def cli():
    """FeatureStore -- AI-powered multi-agent ML feature engineering pipeline.

    Four specialized agents collaborate in sequence:
    Feature Extractor -> Feature Validator -> Feature Registry -> Feature Serving
    """


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to raw data JSON file")
@click.option("--task", "-t", "task_description", default="", help="Description of the ML task")
@click.option("--scenario", default="training", help="Serving scenario: training|inference|both")
@click.option("--output", "-o", default="featurestore_report.json", help="Save full report to JSON file")
def extract(input_path, task_description, scenario, output):
    """Run the full FeatureStore pipeline on INPUT data."""
    console.print(Panel.fit(
        "[bold]FeatureStore Pipeline[/]\n"
        f"Input: {input_path}\n"
        f"Task: {task_description or 'general'}\n"
        f"Scenario: {scenario}",
        border_style="blue"
    ))

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]File not found:[/] {input_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON:[/] {e}")
        sys.exit(1)

    try:
        pipeline = Pipeline()
        result = pipeline.run(data, task_description, scenario)

        if result.success:
            pipeline.save_report(output)
        elif result.errors:
            console.print("[red]Pipeline errors:[/]")
            for err in result.errors:
                console.print(f"  - {err}")
            sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Configuration Error:[/] {e}")
        console.print("[dim]Make sure DEEPSEEK_API_KEY is set in your .env file[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected Error:[/] {e}")
        sys.exit(1)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to raw data JSON file")
@click.option("--task", "-t", "task_description", default="", help="Description of the ML task")
def validate(input_path, task_description):
    """Run Feature Extractor + Validator only (first 2 stages)."""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]File not found:[/] {input_path}")
        sys.exit(1)

    try:
        pipeline = Pipeline()
        result = pipeline.run(data, task_description)
        if result.success:
            pipeline.save_report("featurestore_partial.json")
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to raw data JSON file")
@click.option("--task", "-t", "task_description", default="", help="Description of the ML task")
def register(input_path, task_description):
    """Run up to Feature Registry (first 3 stages)."""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]File not found:[/] {input_path}")
        sys.exit(1)

    try:
        pipeline = Pipeline()
        from src.agents.feature_extractor import FeatureExtractorAgent
        from src.agents.feature_validator import FeatureValidatorAgent
        from src.agents.feature_registry import FeatureRegistryAgent

        extractor = FeatureExtractorAgent()
        features = extractor.extract(data, task_description)

        validator = FeatureValidatorAgent()
        validation = validator.validate(features, data)

        registry = FeatureRegistryAgent()
        reg_result = registry.register(features, validation)

        console.print(json.dumps(reg_result, ensure_ascii=False, indent=2))
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to pipeline JSON report")
@click.option("--scenario", default="training", help="Serving scenario: training|inference|both")
def serve(input_path, scenario):
    """Design serving plan from a saved report."""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        from src.agents.feature_serving import FeatureServingAgent
        serving_agent = FeatureServingAgent()
        result = serving_agent.serve(
            report_data.get("features", {}),
            report_data.get("registry", {}),
            scenario,
        )

        views = result.get("feature_views", [])
        strategies = result.get("serving_strategies", [])
        console.print(f"[bold]Feature Views:[/] {len(views)}")
        for view in views:
            console.print(f"\n[bold cyan]{view.get('name', 'Unnamed')}[/bold cyan]")
            console.print(f"  Entities: {view.get('entities', [])}")
            console.print(f"  Features: {view.get('features', [])}")
            console.print(f"  TTL: {view.get('ttl', 'N/A')}")
        console.print(f"\n[bold]Serving Strategies:[/] {len(strategies)}")
        for strat in strategies:
            console.print(f"  - {strat.get('scenario', '?')}: {strat.get('strategy', '')} (target: {strat.get('latency_target', '?')})")
    except FileNotFoundError:
        console.print(f"[red]File not found:[/] {input_path}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


@cli.command()
def config():
    """Show current configuration."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    console.print(f"API Base: {os.getenv('DEEPSEEK_BASE_URL', 'NOT SET')}")
    console.print(f"Model: {os.getenv('DEEPSEEK_MODEL', 'NOT SET')}")
    console.print(f"API Key: {'***' + api_key[-4:] if api_key else 'NOT SET'}")


if __name__ == "__main__":
    cli()
