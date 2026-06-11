"""Smoke tests: the built site contains the dataset and UI scaffolding."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_site import SITE_DOMAIN, build


def test_build_outputs(tmp_path):
    out = build(tmp_path / "_site")

    index = (out / "index.html").read_text()
    assert 'id="dataset"' in index
    assert 'id="f-torch"' in index
    assert "elenacliu" in index  # acknowledgement must stay

    assert (out / "CNAME").read_text().strip() == SITE_DOMAIN
    assert (out / ".nojekyll").exists()
    assert (out / "app.js").exists()
    assert (out / "style.css").exists()


def test_embedded_dataset_is_valid_json(tmp_path):
    out = build(tmp_path / "_site")
    index = (out / "index.html").read_text()
    start = index.index('id="dataset" type="application/json">') + len(
        'id="dataset" type="application/json">'
    )
    end = index.index("</script>", start)
    dataset = json.loads(index[start:end].replace("<\\/", "</"))

    assert dataset["combos"], "dataset must contain combos"
    accels = {c["accel"] for c in dataset["combos"]}
    assert {"cuda", "rocm", "xpu", "cpu"} <= accels
    assert dataset["cuda_drivers"]["linux"]["12.1"] == "530.30.02"
    assert "gfx942" in dataset["rocm_support"]["7.0"]
