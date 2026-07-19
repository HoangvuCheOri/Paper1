#!/usr/bin/env python3
"""Audit data, then generate all registered paper figures and tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from circle_sop import build_circle_sop_assets, configured as circle_sop_configured
from paper_audit import audit, render
from paper_common import load_registry
from paper_figures import FIGURES
from paper_style import apply_style
from paper_tables import build as build_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(Path(__file__).with_name("datasets.yaml")))
    parser.add_argument("--output-dir", default="paper_exports")
    args = parser.parse_args()
    registry = load_registry(args.registry)
    output = Path(args.output_dir)
    errors, warnings = audit(registry)
    report = render(errors, warnings)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_audit.md").write_text(report, encoding="utf-8")
    print(report, end="")
    if errors:
        raise SystemExit("blocking audit errors; assets were not generated")
    apply_style()
    for name, function in FIGURES.items():
        function(registry, output / "figures")
        print(f"generated figure: {name}")
    build_tables(registry, output / "tables")
    if circle_sop_configured(registry):
        paths = build_circle_sop_assets(registry, output)
        print(f"generated Circle SOP assets: {len(paths)} files")


if __name__ == "__main__":
    main()
