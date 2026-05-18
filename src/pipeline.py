import json
import time
from dataclasses import dataclass, field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agents.feature_extractor import FeatureExtractorAgent
from src.agents.feature_validator import FeatureValidatorAgent
from src.agents.feature_registry import FeatureRegistryAgent
from src.agents.feature_serving import FeatureServingAgent

console = Console()


@dataclass
class PipelineResult:
    features: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    registry: dict = field(default_factory=dict)
    serving: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class Pipeline:
    """FeatureStore pipeline: Feature Extractor -> Feature Validator -> Feature Registry -> Feature Serving."""

    def __init__(self):
        self.result = PipelineResult()

    def run(self, data: dict, task_description: str = "", scenario: str = "training") -> PipelineResult:
        t0 = time.time()

        # Stage 1: Feature Extractor
        console.print(Panel.fit("[bold blue]STAGE 1/4: Feature Extractor Agent[/] -- extracting features", border_style="blue"))
        try:
            extractor = FeatureExtractorAgent()
            self.result.features = extractor.extract(data, task_description)
            self._print_extractor_summary()
        except Exception as e:
            self.result.errors.append(f"Feature Extractor failed: {e}")
            console.print(f"[red]  X Feature Extractor error: {e}[/red]")
            return self.result

        # Stage 2: Feature Validator
        console.print(Panel.fit("[bold yellow]STAGE 2/4: Feature Validator Agent[/] -- validating features", border_style="yellow"))
        try:
            validator = FeatureValidatorAgent()
            self.result.validation = validator.validate(self.result.features, data)
            self._print_validator_summary()
        except Exception as e:
            self.result.errors.append(f"Feature Validator failed: {e}")
            console.print(f"[red]  X Feature Validator error: {e}[/red]")
            return self.result

        # Stage 3: Feature Registry
        console.print(Panel.fit("[bold green]STAGE 3/4: Feature Registry Agent[/] -- registering features", border_style="green"))
        try:
            registry = FeatureRegistryAgent()
            self.result.registry = registry.register(self.result.features, self.result.validation)
            self._print_registry_summary()
        except Exception as e:
            self.result.errors.append(f"Feature Registry failed: {e}")
            console.print(f"[red]  X Feature Registry error: {e}[/red]")
            return self.result

        # Stage 4: Feature Serving
        console.print(Panel.fit("[bold red]STAGE 4/4: Feature Serving Agent[/] -- designing serving plan", border_style="red"))
        try:
            serving = FeatureServingAgent()
            self.result.serving = serving.serve(self.result.features, self.result.registry, scenario)
            self._print_serving_summary()
        except Exception as e:
            self.result.errors.append(f"Feature Serving failed: {e}")
            console.print(f"[red]  X Feature Serving error: {e}[/red]")
            return self.result

        self.result.elapsed_seconds = time.time() - t0
        self._print_final_summary()
        return self.result

    def _print_extractor_summary(self):
        r = self.result.features
        features = r.get("features", [])
        groups = r.get("feature_groups", [])
        stats = r.get("raw_stats", {})
        table = Table(title="Feature Extraction")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Features Found", str(len(features)))
        table.add_row("Feature Groups", str(len(groups)))
        table.add_row("Row Count", str(stats.get("row_count", "?")))
        table.add_row("Column Count", str(stats.get("column_count", "?")))
        if stats.get("missing_fields"):
            table.add_row("Missing Fields", ", ".join(stats["missing_fields"][:5]))
        console.print(table)

    def _print_validator_summary(self):
        v = self.result.validation
        summary = v.get("quality_summary", {})
        recommendations = v.get("recommendations", [])
        table = Table(title="Feature Validation")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Total Features", str(summary.get("total_features", "?")))
        table.add_row("Passed", str(summary.get("passed", "?")))
        table.add_row("Warnings", str(summary.get("warnings", "?")))
        table.add_row("Failed", str(summary.get("failed", "?")))
        table.add_row("Quality Score", str(v.get("overall_quality_score", "?")))
        console.print(table)
        for rec in recommendations[:3]:
            console.print(f"  - [yellow]{rec.get('feature', '?')}[/]: {rec.get('action', '')} ({rec.get('reason', '')})")

    def _print_registry_summary(self):
        reg = self.result.registry
        metadata = reg.get("metadata", {})
        table = Table(title="Feature Registry")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Total Features", str(metadata.get("total_features", "?")))
        table.add_row("Active", str(metadata.get("active", "?")))
        table.add_row("Deprecated", str(metadata.get("deprecated", "?")))
        table.add_row("Draft", str(metadata.get("draft", "?")))
        console.print(table)

    def _print_serving_summary(self):
        s = self.result.serving
        views = s.get("feature_views", [])
        strategies = s.get("serving_strategies", [])
        console.print(f"[red]Feature Views:[/] {len(views)}")
        console.print(f"[red]Serving Strategies:[/] {len(strategies)}")
        for strat in strategies[:3]:
            console.print(f"  - [bold]{strat.get('scenario', '?')}[/]: {strat.get('strategy', '')} (target: {strat.get('latency_target', '?')})")

    def _print_final_summary(self):
        console.print()
        console.print(Panel.fit(
            f"[bold]Pipeline Complete[/]\n"
            f"Time: {self.result.elapsed_seconds:.1f}s\n"
            f"Features Extracted: {len(self.result.features.get('features', []))}\n"
            f"Errors: {len(self.result.errors)}",
            border_style="green" if self.result.success else "red"
        ))

    def save_report(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "features": self.result.features,
                "validation": self.result.validation,
                "registry": self.result.registry,
                "serving": self.result.serving,
                "elapsed_seconds": self.result.elapsed_seconds,
                "errors": self.result.errors,
            }, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Report saved to {path}[/]")
