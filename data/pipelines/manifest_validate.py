"""CLI tool to validate a DatasetManifest JSON file against the schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema


def main() -> None:
    """Validate a manifest JSON file against the DatasetManifest schema."""
    parser = argparse.ArgumentParser(description="Validate a DatasetManifest JSON file.")
    parser.add_argument("manifest", help="Path to manifest JSON file to validate.")
    parser.add_argument(
        "--schema",
        default="data/schemas/dataset_manifest_schema.json",
        help="Path to JSON Schema file (default: data/schemas/dataset_manifest_schema.json).",
    )
    args = parser.parse_args()

    schema_path = Path(args.schema)
    manifest_path = Path(args.manifest)

    with schema_path.open(encoding="utf-8") as f:
        schema = json.load(f)
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    try:
        jsonschema.validate(instance=manifest, schema=schema)
    except jsonschema.ValidationError as exc:
        print(f"✗ {manifest_path} is invalid: {exc.message}", file=sys.stderr)
        sys.exit(1)

    print(f"✓ {manifest_path} is valid.")


if __name__ == "__main__":
    main()
