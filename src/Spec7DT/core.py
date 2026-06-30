from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass
class CatalogFrame:
    """SED-fitting input catalog plus the metadata needed to interpret it."""

    data: pd.DataFrame
    catalog_type: str
    filter_metadata: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class FitResultSet:
    """Standard container for fitting-tool outputs restored by analysis code."""

    results: Mapping[str, pd.DataFrame]
    totals: pd.DataFrame | None = None
    units: Mapping[str, Any] = field(default_factory=dict)
    source_file: Path | None = None
    tool: str = "cigale"
    image_size: int | None = None
    date_tag: str | None = None


@dataclass
class AnalysisProduct:
    """Named analysis output such as a map, profile, plot, or audit table."""

    kind: str
    data: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
