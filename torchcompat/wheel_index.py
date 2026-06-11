"""Fetch and parse the official PyTorch wheel index.

Every wheel filename on https://download.pytorch.org/whl/torch/ encodes a
full compatibility record::

    torch-<version>[+<variant>]-<pytag>-<abitag>-<platform>.whl

Old wheels (pre-0.4.1) carry no ``+variant`` local tag; for those the
accelerator is recovered from the variant directory in the link URL
(e.g. ``/whl/cu90/torch-1.0.0-...``).
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

INDEX_URL = "https://download.pytorch.org/whl/torch/"

_WHEEL_RE = re.compile(
    r"^torch-(?P<version>[0-9][^-+]*)"
    r"(?:\+(?P<variant>[^-]+))?"
    r"-(?P<pytag>[a-z0-9]+)-(?P<abitag>[a-z0-9_.]+)-(?P<platform>[A-Za-z0-9_.]+)\.whl$"
)

_PY_TAG_RE = re.compile(r"^cp(?P<major>\d)(?P<minor>\d+)$")

_VARIANT_DIR_RE = re.compile(r"/whl/(?P<variant>[^/]+)/[^/]+$")


def cu_tag_to_version(tag: str) -> str:
    """``cu118`` -> ``11.8``; ``cu92`` -> ``9.2``; ``cu130`` -> ``13.0``."""
    digits = tag[2:]
    return f"{digits[:-1]}.{digits[-1]}"


def _classify_variant(variant: str) -> tuple[str, str] | None:
    """Map a local-version tag or variant directory to (accel, accel_ver)."""
    if re.fullmatch(r"cu\d{2,3}", variant):
        return "cuda", cu_tag_to_version(variant)
    m = re.fullmatch(r"rocm(\d+(?:\.\d+)*)", variant)
    if m:
        return "rocm", m.group(1)
    if variant == "xpu" or variant.startswith("xpu."):
        return "xpu", ""
    if variant == "cpu" or variant.startswith("cpu."):
        return "cpu", ""
    return None


def _parse_platform(platform_tag: str) -> tuple[str, str] | None:
    """Map a wheel platform tag to (os, arch)."""
    # Compressed tag sets are joined with '.'; the first one is enough to
    # identify os/arch since torch never mixes platforms in one wheel.
    tag = platform_tag.split(".")[0]
    if tag.startswith(("manylinux", "linux")):
        os_name = "linux"
    elif tag.startswith("win"):
        os_name = "windows"
    elif tag.startswith("macosx"):
        os_name = "macos"
    else:
        return None
    if tag.endswith(("x86_64", "amd64")):
        arch = "x86_64"
    elif tag.endswith("aarch64"):
        arch = "aarch64"
    elif tag.endswith("arm64"):
        arch = "arm64"
    elif tag.endswith(("i686", "win32")):
        arch = "x86"
    else:
        arch = tag.rsplit("_", 1)[-1]
    return os_name, arch


def parse_wheel(filename: str, variant_dir: str) -> dict | None:
    """Parse one wheel filename into a compatibility record, or None.

    ``variant_dir`` is the URL path the wheel is served from (e.g.
    ``whl/cu90``), used when the filename has no local version tag.
    """
    m = _WHEEL_RE.match(filename)
    if not m:
        return None

    py = _PY_TAG_RE.match(m.group("pytag"))
    if not py:
        return None

    platform = _parse_platform(m.group("platform"))
    if not platform:
        return None

    variant = m.group("variant") or variant_dir.rstrip("/").rsplit("/", 1)[-1]
    classified = _classify_variant(variant)
    if not classified:
        return None
    accel, accel_ver = classified

    return {
        "torch": m.group("version"),
        "accel": accel,
        "accel_ver": accel_ver,
        "python": f"{py.group('major')}.{py.group('minor')}",
        "os": platform[0],
        "arch": platform[1],
    }


class _AnchorParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.hrefs.append(value)


def parse_index_html(html: str) -> list[dict]:
    """Parse the wheel-index HTML into a list of compatibility records."""
    parser = _AnchorParser()
    parser.feed(html)

    wheels = []
    for href in parser.hrefs:
        path = urllib.parse.unquote(urllib.parse.urlparse(href).path)
        filename = path.rsplit("/", 1)[-1]
        dir_match = _VARIANT_DIR_RE.search(path)
        variant_dir = dir_match.group("variant") if dir_match else ""
        wheel = parse_wheel(filename, variant_dir)
        if wheel:
            wheels.append(wheel)
    return wheels


def _version_key(version: str) -> tuple:
    """Natural-sort key for dotted versions ('2.10' > '2.9')."""
    return tuple(
        int(part) if part.isdigit() else -1
        for part in re.split(r"[.+]|post", version)
        if part != ""
    )


def aggregate(wheels: list[dict]) -> list[dict]:
    """Group records by everything except python; collect python versions.

    Returns combos sorted newest torch first, then accelerator.
    """
    groups: dict[tuple, set] = {}
    for w in wheels:
        key = (w["torch"], w["accel"], w["accel_ver"], w["os"], w["arch"])
        groups.setdefault(key, set()).add(w["python"])

    combos = [
        {
            "torch": torch,
            "accel": accel,
            "accel_ver": accel_ver,
            "os": os_name,
            "arch": arch,
            "python": sorted(pythons, key=_version_key),
        }
        for (torch, accel, accel_ver, os_name, arch), pythons in groups.items()
    ]
    combos.sort(
        key=lambda c: (
            _version_key(c["torch"]),
            c["accel"],
            _version_key(c["accel_ver"] or "0"),
            c["os"],
            c["arch"],
        ),
        reverse=True,
    )
    return combos


def fetch_index(url: str = INDEX_URL) -> str:
    """Download the wheel index page; raises on HTTP errors."""
    req = urllib.request.Request(url, headers={"User-Agent": "torch-compat-bot"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Index fetch failed: HTTP {resp.status}")
        return resp.read().decode("utf-8")
