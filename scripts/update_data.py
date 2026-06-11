#!/usr/bin/env python3
"""Scrape the PyTorch wheel index and regenerate data/torch_matrix.json."""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchcompat.wheel_index import INDEX_URL, aggregate, fetch_index, parse_index_html

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "data" / "torch_matrix.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=INDEX_URL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    html = fetch_index(args.url)
    wheels = parse_index_html(html)
    if not wheels:
        print("ERROR: parsed 0 wheels — index format may have changed", file=sys.stderr)
        return 1

    combos = aggregate(wheels)
    print(f"Parsed {len(wheels)} wheel links into {len(combos)} combos")

    matrix = {
        "source": args.url,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "combos": combos,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Keep the diff (and CI commit decision) driven by actual data changes,
    # not by the generation timestamp.
    if args.output.exists():
        old = json.loads(args.output.read_text())
        if old.get("combos") == combos and old.get("source") == matrix["source"]:
            print("No changes in compatibility data; keeping existing file")
            return 0

    args.output.write_text(json.dumps(matrix, indent=1) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
