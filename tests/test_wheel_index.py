"""Tests for parsing the PyTorch wheel index."""

from torchcompat.wheel_index import (
    aggregate,
    cu_tag_to_version,
    parse_index_html,
    parse_wheel,
)


class TestCuTagToVersion:
    def test_two_digit(self):
        assert cu_tag_to_version("cu75") == "7.5"
        assert cu_tag_to_version("cu92") == "9.2"

    def test_three_digit(self):
        assert cu_tag_to_version("cu100") == "10.0"
        assert cu_tag_to_version("cu118") == "11.8"
        assert cu_tag_to_version("cu130") == "13.0"
        assert cu_tag_to_version("cu132") == "13.2"


class TestParseWheel:
    def test_cuda_local_tag(self):
        w = parse_wheel("torch-2.3.1+cu118-cp310-cp310-linux_x86_64.whl", "whl/cu118")
        assert w == {
            "torch": "2.3.1",
            "accel": "cuda",
            "accel_ver": "11.8",
            "python": "3.10",
            "os": "linux",
            "arch": "x86_64",
        }

    def test_rocm_multi_dot_tag(self):
        w = parse_wheel(
            "torch-2.5.1+rocm6.2.4-cp312-cp312-linux_x86_64.whl", "whl/rocm6.2.4"
        )
        assert w["accel"] == "rocm"
        assert w["accel_ver"] == "6.2.4"

    def test_xpu_tag(self):
        w = parse_wheel("torch-2.6.0+xpu-cp311-cp311-win_amd64.whl", "whl/xpu")
        assert w["accel"] == "xpu"
        assert w["accel_ver"] == ""
        assert w["os"] == "windows"
        assert w["arch"] == "x86_64"

    def test_cpu_cxx11_abi_tag(self):
        w = parse_wheel(
            "torch-2.0.0+cpu.cxx11.abi-cp38-cp38-linux_x86_64.whl", "whl/cpu"
        )
        assert w["accel"] == "cpu"

    def test_untagged_macos_arm64(self):
        w = parse_wheel("torch-2.3.1-cp312-none-macosx_11_0_arm64.whl", "whl/cpu")
        assert w["accel"] == "cpu"
        assert w["os"] == "macos"
        assert w["arch"] == "arm64"

    def test_untagged_cuda_from_dir(self):
        w = parse_wheel("torch-1.0.0-cp37-cp37m-linux_x86_64.whl", "whl/cu90")
        assert w["accel"] == "cuda"
        assert w["accel_ver"] == "9.0"

    def test_python2_mu_abi(self):
        w = parse_wheel("torch-0.3.1-cp27-cp27mu-linux_x86_64.whl", "whl/cu75")
        assert w["python"] == "2.7"

    def test_post_release_version(self):
        w = parse_wheel("torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl", "whl/cpu")
        assert w["torch"] == "0.3.0.post4"

    def test_aarch64_manylinux(self):
        w = parse_wheel(
            "torch-2.4.0-cp310-cp310-manylinux_2_28_aarch64.whl", "whl/cpu"
        )
        assert w["os"] == "linux"
        assert w["arch"] == "aarch64"

    def test_not_a_wheel(self):
        assert parse_wheel("torch-2.3.1.tar.gz", "whl/cpu") is None

    def test_malformed_filename(self):
        assert parse_wheel("torch.whl", "whl/cpu") is None


class TestParseIndexHtml:
    HTML = """<!DOCTYPE html><html><body><h1>Links for torch</h1>
    <a href="https://download-r2.pytorch.org/whl/cpu/torch-2.3.1-cp312-none-macosx_11_0_arm64.whl#sha256=abc">torch-2.3.1-cp312-none-macosx_11_0_arm64.whl</a><br/>
    <a href="https://download-r2.pytorch.org/whl/cu118/torch-2.3.1%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=def">torch-2.3.1+cu118-cp310-cp310-linux_x86_64.whl</a><br/>
    </body></html>"""

    def test_parses_anchors(self):
        wheels = parse_index_html(self.HTML)
        assert len(wheels) == 2
        assert wheels[0]["accel"] == "cpu"
        assert wheels[1]["accel"] == "cuda"
        assert wheels[1]["accel_ver"] == "11.8"

    def test_urlencoded_plus_is_decoded(self):
        wheels = parse_index_html(self.HTML)
        assert wheels[1]["torch"] == "2.3.1"


class TestAggregate:
    def test_groups_pythons(self):
        wheels = [
            {"torch": "2.3.1", "accel": "cuda", "accel_ver": "12.1",
             "python": "3.10", "os": "linux", "arch": "x86_64"},
            {"torch": "2.3.1", "accel": "cuda", "accel_ver": "12.1",
             "python": "3.8", "os": "linux", "arch": "x86_64"},
            # duplicate must collapse
            {"torch": "2.3.1", "accel": "cuda", "accel_ver": "12.1",
             "python": "3.8", "os": "linux", "arch": "x86_64"},
        ]
        combos = aggregate(wheels)
        assert len(combos) == 1
        assert combos[0]["python"] == ["3.8", "3.10"]  # natural sort, not lexical

    def test_distinct_os_kept_separate(self):
        wheels = [
            {"torch": "2.3.1", "accel": "cpu", "accel_ver": "",
             "python": "3.10", "os": "linux", "arch": "x86_64"},
            {"torch": "2.3.1", "accel": "cpu", "accel_ver": "",
             "python": "3.10", "os": "windows", "arch": "x86_64"},
        ]
        assert len(aggregate(wheels)) == 2

    def test_sorted_newest_torch_first(self):
        wheels = [
            {"torch": "1.13.1", "accel": "cpu", "accel_ver": "",
             "python": "3.10", "os": "linux", "arch": "x86_64"},
            {"torch": "2.3.1", "accel": "cpu", "accel_ver": "",
             "python": "3.10", "os": "linux", "arch": "x86_64"},
        ]
        combos = aggregate(wheels)
        assert combos[0]["torch"] == "2.3.1"
