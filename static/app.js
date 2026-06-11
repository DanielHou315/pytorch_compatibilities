"use strict";

const DATA = JSON.parse(document.getElementById("dataset").textContent);

const PAGE_SIZE = 200;

const ACCEL_LABELS = { cuda: "CUDA", rocm: "ROCm", xpu: "Intel XPU", cpu: "CPU", mps: "Apple Silicon (MPS)" };
// Which GPU vendors are relevant when an accelerator type is selected.
const ACCEL_VENDORS = { cuda: ["nvidia"], rocm: ["amd"], xpu: ["intel"], cpu: [], mps: ["apple"] };

// "Apple Silicon (MPS)" is a virtual accelerator: MPS ships inside the macOS
// arm64 CPU wheels, so it selects those rows rather than a wheel variant.
function matchesAccel(c, accel) {
  if (!accel) return true;
  if (accel === "mps") return c.accel === "cpu" && c.os === "macos" && c.arch === "arm64";
  return c.accel === accel;
}
const OS_LABELS = { linux: "Linux", windows: "Windows", macos: "macOS" };

const els = {
  torch: document.getElementById("f-torch"),
  accel: document.getElementById("f-accel"),
  accelVer: document.getElementById("f-accel-ver"),
  python: document.getElementById("f-python"),
  os: document.getElementById("f-os"),
  gpu: document.getElementById("f-gpu"),
  reset: document.getElementById("f-reset"),
  count: document.getElementById("count"),
  gpuNote: document.getElementById("gpu-note"),
  tbody: document.querySelector("#results tbody"),
  pager: document.getElementById("pager"),
};

let currentPage = 1;

function versionKey(v) {
  return v.split(/[.+-]|post/).map((p) => (/^\d+$/.test(p) ? +p : -1));
}

function cmpVersions(a, b) {
  const ka = versionKey(a), kb = versionKey(b);
  for (let i = 0; i < Math.max(ka.length, kb.length); i++) {
    const d = (ka[i] || 0) - (kb[i] || 0);
    if (d) return d;
  }
  return 0;
}

function uniqueSortedDesc(values) {
  return [...new Set(values)].sort(cmpVersions).reverse();
}

function fillSelect(select, options, { anyLabel = "any" } = {}) {
  const prev = select.value;
  select.replaceChildren();
  const any = new Option(anyLabel, "");
  select.add(any);
  for (const [value, label] of options) select.add(new Option(label, value));
  if ([...select.options].some((o) => o.value === prev)) select.value = prev;
}

function initFilters() {
  fillSelect(els.torch, uniqueSortedDesc(DATA.combos.map((c) => c.torch)).map((v) => [v, v]));
  fillSelect(els.accel, Object.entries(ACCEL_LABELS));
  fillAccelVersions();
  fillSelect(els.python, uniqueSortedDesc(DATA.combos.flatMap((c) => c.python)).map((v) => [v, v]));
  const oses = [...new Set(DATA.combos.map((c) => `${c.os}/${c.arch}`))].sort();
  fillSelect(els.os, oses.map((v) => {
    const [os, arch] = v.split("/");
    return [v, `${OS_LABELS[os] || os} ${arch}`];
  }));
  fillGpuOptions();
}

function fillGpuOptions() {
  const vendors = ACCEL_VENDORS[els.accel.value] || null;
  const options = DATA.gpus
    .map((g, i) => [String(i), g.name, g.vendor])
    .filter(([, , vendor]) => !vendors || vendors.includes(vendor))
    .map(([value, label]) => [value, label]);
  fillSelect(els.gpu, options, { anyLabel: "— select to check your GPU —" });
}

function fillAccelVersions() {
  const accel = els.accel.value;
  // Version options only make sense once an accelerator type is chosen.
  const vers = !accel || accel === "cpu" || accel === "mps"
    ? []
    : uniqueSortedDesc(
        DATA.combos.filter((c) => c.accel === accel && c.accel_ver).map((c) => c.accel_ver)
      );
  fillSelect(els.accelVer, vers.map((v) => [v, v]));
}

function rocmEntry(ver) {
  const s = DATA.rocm_support;
  if (s[ver]) return s[ver];
  const minor = ver.split(".").slice(0, 2).join(".");
  return s[minor] || null;
}

function selectedGpu() {
  if (!els.gpu.value) return null;
  return DATA.gpus[Number(els.gpu.value)] || null;
}

function gpuSupportsRow(gpu, c) {
  // No official wheels exist for this device (e.g. Jetson) — match nothing;
  // the GPU note explains where to get wheels instead.
  if (gpu.no_official_wheels) return false;

  // Hard-coded platform constraints (e.g. ROCm GPUs are Linux only).
  if (gpu.only_os && c.os !== gpu.only_os) return false;
  if (gpu.only_arch && c.arch !== gpu.only_arch) return false;

  if (gpu.vendor === "nvidia") {
    if (c.accel !== "cuda") return c.accel === "cpu";
    const sms = DATA.cuda_sm_support[c.accel_ver];
    return !!sms && sms.includes(gpu.arch);
  }
  if (gpu.vendor === "amd") {
    if (c.accel !== "rocm") return c.accel === "cpu";
    const gfx = rocmEntry(c.accel_ver);
    return !!gfx && gfx.includes(gpu.arch);
  }
  if (gpu.vendor === "intel") {
    return c.accel === "xpu" || c.accel === "cpu";
  }
  if (gpu.vendor === "apple") {
    return c.accel === "cpu";
  }
  return true;
}

function matches(c) {
  if (els.torch.value && c.torch !== els.torch.value) return false;
  if (!matchesAccel(c, els.accel.value)) return false;
  if (els.accelVer.value && c.accel_ver !== els.accelVer.value) return false;
  if (els.python.value && !c.python.includes(els.python.value)) return false;
  if (els.os.value && `${c.os}/${c.arch}` !== els.os.value) return false;
  const gpu = selectedGpu();
  if (gpu && !gpuSupportsRow(gpu, c)) return false;
  return true;
}

function variantTag(c) {
  if (c.accel === "cuda") return "cu" + c.accel_ver.replace(/\./g, "");
  if (c.accel === "rocm") return "rocm" + c.accel_ver;
  if (c.accel === "xpu") return "xpu";
  return "cpu";
}

function installCommand(c) {
  const base = `pip install torch==${c.torch}`;
  if (c.os === "macos") return base;
  return `${base} --index-url https://download.pytorch.org/whl/${variantTag(c)}`;
}

function requirements(c) {
  if (c.accel === "cuda") {
    const table = c.os === "windows" ? DATA.cuda_drivers.windows : DATA.cuda_drivers.linux;
    const min = table[c.accel_ver];
    return min ? `NVIDIA driver ≥ ${min}` : "NVIDIA driver: see CUDA release notes";
  }
  if (c.accel === "rocm") {
    const gfx = rocmEntry(c.accel_ver);
    return gfx ? `Officially: ${gfx.join(", ")}` : "See ROCm docs";
  }
  if (c.accel === "xpu") return "Intel GPU driver + oneAPI runtime";
  if (c.os === "macos") return "MPS acceleration on Apple Silicon (torch ≥ 1.12)";
  return "—";
}

function accelCell(c) {
  const label = ACCEL_LABELS[c.accel] || c.accel;
  return c.accel_ver ? `${label} ${c.accel_ver}` : label;
}

function render() {
  const rows = DATA.combos.filter(matches);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  currentPage = Math.min(Math.max(1, currentPage), pages);
  const start = (currentPage - 1) * PAGE_SIZE;
  const pageRows = rows.slice(start, start + PAGE_SIZE);

  els.tbody.replaceChildren();
  for (const c of pageRows) {
    const tr = document.createElement("tr");

    const cells = [
      c.torch,
      accelCell(c),
      c.python.join(", "),
      `${OS_LABELS[c.os] || c.os} ${c.arch}`,
      requirements(c),
    ];
    for (const text of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }

    const tdCmd = document.createElement("td");
    tdCmd.className = "cmd";
    const code = document.createElement("code");
    code.textContent = installCommand(c);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy";
    btn.textContent = "copy";
    btn.addEventListener("click", () => {
      navigator.clipboard.writeText(code.textContent).then(() => {
        btn.textContent = "copied!";
        setTimeout(() => (btn.textContent = "copy"), 1200);
      });
    });
    tdCmd.append(code, btn);
    tr.appendChild(tdCmd);

    els.tbody.appendChild(tr);
  }

  els.count.textContent =
    rows.length > PAGE_SIZE
      ? `Showing ${start + 1}–${start + pageRows.length} of ${rows.length} matching combinations.`
      : `${rows.length} matching combination${rows.length === 1 ? "" : "s"}.`;
  renderPager(pages);

  const gpu = selectedGpu();
  els.gpuNote.hidden = !gpu;
  if (gpu) renderGpuNote(gpu);
}

function renderPager(pages) {
  els.pager.replaceChildren();
  if (pages <= 1) return;

  const addButton = (label, page, { disabled = false, current = false } = {}) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.disabled = disabled || current;
    if (current) btn.className = "current";
    btn.addEventListener("click", () => {
      currentPage = page;
      render();
      document.getElementById("results").scrollIntoView({ block: "start" });
    });
    els.pager.appendChild(btn);
  };

  addButton("‹ Prev", currentPage - 1, { disabled: currentPage === 1 });
  for (let p = 1; p <= pages; p++) addButton(String(p), p, { current: p === currentPage });
  addButton("Next ›", currentPage + 1, { disabled: currentPage === pages });
}

function renderGpuNote(gpu) {
  const vendorText = {
    nvidia: `Showing CUDA builds whose toolkit supports compute capability ${gpu.arch} (plus CPU builds). Official wheels may target fewer architectures — verify with torch.cuda.is_available().`,
    amd: `Showing ROCm builds that officially support ${gpu.arch} (plus CPU builds). Unlisted GPUs often still work, e.g. via HSA_OVERRIDE_GFX_VERSION.`,
    intel: "Showing Intel XPU builds (plus CPU builds). Intel GPU drivers are required.",
    apple: "Showing macOS arm64 builds.",
  };
  els.gpuNote.replaceChildren();
  // When no official wheels exist (Jetson), the vendor blurb about shown
  // builds would contradict the empty table — show only the explanation.
  const parts = gpu.no_official_wheels
    ? [gpu.note || "No official wheels exist for this device."]
    : [vendorText[gpu.vendor] || "", gpu.note || ""].filter(Boolean);
  els.gpuNote.append(parts.join(" "));
  if (gpu.note_url) {
    const link = document.createElement("a");
    link.href = gpu.note_url;
    link.textContent = "Official install guide.";
    els.gpuNote.append(" ", link);
  }
}

function onFilterChange() {
  currentPage = 1;
  render();
}

for (const el of [els.torch, els.accelVer, els.python, els.os, els.gpu]) {
  el.addEventListener("change", onFilterChange);
}
els.accel.addEventListener("change", () => {
  fillAccelVersions();
  fillGpuOptions();
  onFilterChange();
});
els.reset.addEventListener("click", () => {
  for (const el of [els.torch, els.accel, els.accelVer, els.python, els.os, els.gpu]) {
    el.value = "";
  }
  fillAccelVersions();
  fillGpuOptions();
  onFilterChange();
});

initFilters();
render();

// Browsers may restore previous <select> state after scripts run (reload /
// back-forward); re-sync the dependent options and table once settled.
window.addEventListener("load", () => {
  fillAccelVersions();
  fillGpuOptions();
  render();
});
