#!/usr/bin/env python3
"""Write Jupyter notebooks using deterministic JSON serialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_notebook(path: Path) -> bool:
    """Normalize one notebook and return whether its file changed."""
    original = path.read_text(encoding="utf-8")
    notebook = json.loads(original)

    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"{path} is not a valid notebook document")

    # Repair fields required by the nbformat v4 output schema.
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type in {"display_data", "execute_result"}:
                output.setdefault("metadata", {})
            elif output_type == "stream":
                output.setdefault("name", "stdout")

    canonical = (
        json.dumps(notebook, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    if canonical == original:
        return False

    path.write_text(canonical, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize notebook JSON to prevent serialization-only diffs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Notebook paths; defaults to *.ipynb in the current directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.paths or sorted(Path.cwd().glob("*.ipynb"))

    for path in paths:
        if path.suffix != ".ipynb":
            raise ValueError(f"Expected an .ipynb file: {path}")
        changed = normalize_notebook(path)
        print(f"{'Normalized' if changed else 'Already canonical'}: {path}")


if __name__ == "__main__":
    main()
