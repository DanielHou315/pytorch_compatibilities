"""Sanity checks on the committed data files."""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _load(name):
    return json.loads((DATA / name).read_text())


class TestTorchMatrix:
    def test_full_version_history_present(self):
        # Regression: the matrix must span the full wheel history, not just
        # recent releases (user-reported: site appeared to cut off at 2.4.x).
        versions = {c["torch"] for c in _load("torch_matrix.json")["combos"]}
        for old in ("0.4.0", "1.10.0", "1.13.1", "2.0.0"):
            assert old in versions, f"torch {old} missing from matrix"

    def test_every_cuda_version_has_driver_and_sm_entries(self):
        combos = _load("torch_matrix.json")["combos"]
        cuda_vers = {c["accel_ver"] for c in combos if c["accel"] == "cuda"}
        drivers = _load("cuda_drivers.json")["linux"]
        sms = _load("cuda_sm_support.json")["support"]
        missing_driver = cuda_vers - drivers.keys()
        missing_sm = cuda_vers - sms.keys()
        assert not missing_driver, f"cuda_drivers.json missing: {missing_driver}"
        assert not missing_sm, f"cuda_sm_support.json missing: {missing_sm}"

    def test_every_rocm_version_resolves(self):
        combos = _load("torch_matrix.json")["combos"]
        rocm = _load("rocm_support.json")["support"]
        for ver in {c["accel_ver"] for c in combos if c["accel"] == "rocm"}:
            minor = ".".join(ver.split(".")[:2])
            assert ver in rocm or minor in rocm, f"rocm_support.json missing {ver}"


class TestGpus:
    def test_schema(self):
        gpus = _load("gpus.json")["gpus"]
        for g in gpus:
            assert g["vendor"] in {"nvidia", "amd", "intel", "apple"}, g
            assert g["name"] and g["arch"], g
            assert g.get("only_os") in {None, "linux", "windows", "macos"}, g
            assert g.get("only_arch") in {None, "x86_64", "aarch64", "arm64"}, g

    def test_all_accelerator_vendors_covered(self):
        vendors = {g["vendor"] for g in _load("gpus.json")["gpus"]}
        assert {"nvidia", "amd", "intel", "apple"} <= vendors

    def test_jetson_wheel_availability(self):
        # Pre-Thor Jetson must match zero rows (the SBSA aarch64 wheels don't
        # run on it, and there are no Windows wheels for it either) and explain
        # why. Jetson Thor is the exception: JetPack 7 adopts the unified SBSA
        # aarch64 CUDA stack, so standard wheels apply.
        jetsons = [g for g in _load("gpus.json")["gpus"] if "Jetson" in g["name"]]
        assert len(jetsons) >= 3
        for g in jetsons:
            assert g.get("note") and g.get("note_url"), f"{g['name']} needs a JetPack note"
            assert g.get("only_os") == "linux" and g.get("only_arch") == "aarch64", g
            expected = None if "Thor" in g["name"] else True
            assert g.get("no_official_wheels") is expected, g

    def test_gpu_list_newest_first_within_vendor(self):
        nvidia = [g["arch"] for g in _load("gpus.json")["gpus"] if g["vendor"] == "nvidia"]
        as_floats = [float(a) for a in nvidia]
        assert as_floats == sorted(as_floats, reverse=True), nvidia

    def test_dgx_spark_and_thor_present(self):
        names = " ".join(g["name"] for g in _load("gpus.json")["gpus"])
        assert "DGX Spark" in names and "Jetson Thor" in names
