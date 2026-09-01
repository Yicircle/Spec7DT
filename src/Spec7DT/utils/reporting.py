from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import warnings

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


_ACTIVE_REPORTER = ContextVar("spec7dt_pipeline_reporter", default=None)


@contextmanager
def reporter_context(reporter):
    """Expose a pipeline reporter to nested reduction helpers."""
    token = _ACTIVE_REPORTER.set(reporter)
    try:
        yield reporter
    finally:
        _ACTIVE_REPORTER.reset(token)


def emit_detail(message):
    """Emit a verbose pipeline detail, preserving direct-call text output."""
    reporter = _ACTIVE_REPORTER.get()
    if reporter is None:
        print(message)
    else:
        reporter.detail(str(message))


def emit_alert(
    message,
    *,
    level="warning",
    context=None,
    dedupe_key=None,
    category=RuntimeWarning,
):
    """Report an operational alert without corrupting active progress output."""
    reporter = _ACTIVE_REPORTER.get()
    if reporter is None:
        rendered = str(message)
        if context and str(context) not in rendered:
            rendered = f"{context}: {rendered}"
        if level == "warning":
            warnings.warn(rendered, category, stacklevel=2)
        else:
            print(rendered)
        return
    reporter.alert(
        str(message),
        level=level,
        context=None if context is None else str(context),
        dedupe_key=dedupe_key,
        category=category,
    )


class RichPipelineReporter:
    """Render nested pipeline progress and a compact timing summary."""

    def __init__(self, *, progress=True, verbose=False, show_alerts=True, console=None):
        self.enabled = bool(progress)
        self.verbose = bool(verbose)
        self.show_alerts = bool(show_alerts)
        self.console = console or Console()
        self._progress = None
        self._overall_task = None
        self._step_task = None
        self._step_name = None
        self._step_completed = 0
        self._alerts = {}

    @property
    def step_completed(self):
        return self._step_completed

    def start(self, total_steps):
        self._alerts = {}
        if not self.enabled:
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=2,
            transient=False,
        )
        self._progress.start()
        self._overall_task = self._progress.add_task(
            "Pipeline steps",
            total=max(0, int(total_steps)),
        )

    def start_step(self, name, total_targets):
        self._step_name = str(name)
        self._step_completed = 0
        if not self.enabled:
            return
        if self._step_task is not None:
            self._progress.remove_task(self._step_task)
        self._step_task = self._progress.add_task(
            str(name),
            total=max(0, int(total_targets)),
        )

    def set_target(self, step_name, target=None):
        if target is None:
            target = step_name
            step_name = self._step_name or "Step"
        if self.enabled and self._step_task is not None:
            self._progress.update(
                self._step_task,
                description=f"{step_name}: {target}",
            )

    def advance_target(self, target=None, *, step_name=None, advance=1):
        if target is not None:
            self.set_target(step_name or self._step_name or "Step", target)
        amount = max(0, int(advance))
        self._step_completed += amount
        if self.enabled and self._step_task is not None:
            self._progress.advance(self._step_task, amount)

    def finish_step(self, name):
        if not self.enabled:
            return
        if self._step_task is not None:
            self._progress.update(self._step_task, description=f"{name} [green]done[/green]")
        if self._overall_task is not None:
            self._progress.advance(self._overall_task, 1)

    def detail(self, message):
        if not self.verbose:
            return
        if self.enabled and self._progress is not None:
            self._progress.console.log(message)
        else:
            self.console.print(message)

    def alert(
        self,
        message,
        *,
        level="warning",
        context=None,
        dedupe_key=None,
        category=RuntimeWarning,
    ):
        """Display the first alert and retain grouped counts for the summary."""
        normalized_level = str(level).lower()
        key = (
            normalized_level,
            str(dedupe_key) if dedupe_key is not None else str(message),
        )
        record = self._alerts.setdefault(
            key,
            {
                "level": normalized_level,
                "category": getattr(category, "__name__", str(category)),
                "message": str(message),
                "count": 0,
                "contexts": [],
            },
        )
        record["count"] += 1
        if context and context not in record["contexts"] and len(record["contexts"]) < 3:
            record["contexts"].append(context)

        if record["count"] != 1:
            return
        if not self.show_alerts:
            return
        rendered = str(message)
        if context and str(context) not in rendered:
            rendered = f"{context}: {rendered}"
        style = "yellow" if normalized_level == "warning" else "red"
        label = normalized_level.capitalize()
        console = self._progress.console if self._progress is not None else self.console
        console.print(f"[bold {style}]{label}:[/bold {style}] {escape(rendered)}")

    def failure(self, step_name, target, exc):
        label = f"{step_name} failed"
        if target:
            label += f" at {target}"
        self.console.print(f"[bold red]{label}:[/bold red] {exc}")

    def stop(self):
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def summary(self, rows, *, pdf_path=None):
        table = Table(title="Pipeline summary")
        table.add_column("Step")
        table.add_column("Targets", justify="right")
        table.add_column("Elapsed", justify="right")
        table.add_column("Per target", justify="right")
        for row in rows:
            count = int(row["targets"])
            elapsed = float(row["elapsed"])
            per_target = elapsed / count if count else 0.0
            table.add_row(
                str(row["step"]),
                str(count),
                f"{elapsed:.3f}s",
                f"{per_target:.3f}s",
            )
        self.console.print(table)
        if self.show_alerts and self._alerts:
            alerts = Table(title="Pipeline alerts")
            alerts.add_column("Level")
            alerts.add_column("Count", justify="right")
            alerts.add_column("Alert")
            alerts.add_column("Example targets")
            for record in self._alerts.values():
                alerts.add_row(
                    record["level"],
                    str(record["count"]),
                    record["message"],
                    ", ".join(record["contexts"]) or "—",
                )
            self.console.print(alerts)
        if pdf_path is not None:
            self.console.print(f"[green]Plot PDF:[/green] {pdf_path}")
