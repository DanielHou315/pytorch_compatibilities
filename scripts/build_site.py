#!/usr/bin/env python3
"""Render the static site into _site/ from data/ + templates/ + static/."""

import argparse
import json
import shutil
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DOMAIN = "torch-compat.danielhou.me"


def _load(name: str) -> dict:
    return json.loads((REPO_ROOT / "data" / name).read_text())


def _embed_json(payload) -> str:
    """JSON safe for inlining in a <script> block."""
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def build(output_dir: Path) -> Path:
    matrix = _load("torch_matrix.json")
    dataset = {
        "generated_utc": matrix["generated_utc"],
        "source": matrix["source"],
        "combos": matrix["combos"],
        "cuda_drivers": _load("cuda_drivers.json"),
        "cuda_sm_support": _load("cuda_sm_support.json")["support"],
        "rocm_support": _load("rocm_support.json")["support"],
        "gpus": _load("gpus.json")["gpus"],
    }

    env = Environment(
        loader=FileSystemLoader(REPO_ROOT / "templates"),
        autoescape=select_autoescape(["html"]),
    )
    html = env.get_template("index.html.j2").render(
        dataset_json=_embed_json(dataset),
        generated_utc=matrix["generated_utc"],
        n_combos=len(matrix["combos"]),
        domain=SITE_DOMAIN,
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "index.html").write_text(html)
    (output_dir / "CNAME").write_text(SITE_DOMAIN + "\n")
    (output_dir / ".nojekyll").write_text("")
    for asset in (REPO_ROOT / "static").iterdir():
        shutil.copy(asset, output_dir / asset.name)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "_site")
    args = parser.parse_args()
    out = build(args.output)
    print(f"Site built at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
