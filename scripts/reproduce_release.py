#!/usr/bin/env python3
"""Validate aggregate results and regenerate all released tables and figures."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    commands = [
        [
            sys.executable,
            str(root / "scripts" / "validate_reported_results.py"),
            "--data-dir",
            str(root / "data" / "aggregate"),
        ],
        [
            sys.executable,
            str(root / "scripts" / "generate_summary_tables.py"),
            "--data-dir",
            str(root / "data" / "aggregate"),
            "--output-dir",
            str(root / "tables"),
        ],
        [
            sys.executable,
            str(root / "scripts" / "generate_final_figures.py"),
            "--data-dir",
            str(root / "data" / "aggregate"),
            "--output-dir",
            str(root / "figures"),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
