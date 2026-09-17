const PY_FILES = [
  "ctf_lpbf/__init__.py",
  "ctf_lpbf/ctf_io.py",
  "ctf_lpbf/orientation.py",
  "ctf_lpbf/cleaning.py",
  "ctf_lpbf/analyze.py",
  "ctf_lpbf/visualize.py",
  "ctf_lpbf/pipeline.py",
  "tests/__init__.py",
  "tests/generate_synthetic_ctf.py",
  "web/pyodide_app.py",
];

const statusEl = document.getElementById("status");
const runBtn = document.getElementById("runBtn");
const resultsEl = document.getElementById("results");
const errorEl = document.getElementById("error");
const fileInput = document.getElementById("ctfFiles");
const fileLabel = document.getElementById("fileLabel");

let pyodidePromise = null;
let mode = "demo";

function log(message) {
  statusEl.textContent = message;
}

function setMode(next) {
  mode = next;
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.mode === next);
  });
  document.getElementById("demoControls").classList.toggle("hidden", next === "upload");
  document.getElementById("uploadControls").classList.toggle("hidden", next !== "upload");
  runBtn.textContent =
    next === "smoke" ? "스모크 테스트 실행" : next === "upload" ? "업로드 파일 분석" : "합성 데이터로 실행";
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

function bindSlider(id) {
  const input = document.getElementById(id);
  const value = document.getElementById(`${id}Value`);
  const update = () => {
    value.textContent = input.value;
  };
  input.addEventListener("input", update);
  update();
  return input;
}

const sliders = {
  grid: bindSlider("grid"),
  redWeight: bindSlider("redWeight"),
  greenWeight: bindSlider("greenWeight"),
  noise: bindSlider("noise"),
  mad: bindSlider("mad"),
  isolation: bindSlider("isolation"),
};

fileInput.addEventListener("change", () => {
  const n = fileInput.files.length;
  fileLabel.textContent = n ? `${n}개 CTF 선택됨` : "CTF 파일 선택";
});

function pageBase() {
  return new URL("./", window.location.href);
}

async function fetchText(rel) {
  const url = new URL(rel, pageBase());
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${rel} 로드 실패 (${res.status})`);
  }
  return res.text();
}

function formatErr(err) {
  if (!err) return "unknown error";
  if (typeof err === "string") return err;
  if (err.message) return err.message;
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

function mkdirp(pyodide, path) {
  const parts = path.split("/").filter(Boolean);
  let cur = "";
  for (const part of parts) {
    cur += `/${part}`;
    if (!pyodide.FS.analyzePath(cur).exists) {
      pyodide.FS.mkdir(cur);
    }
  }
}

async function ensurePyodide() {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      log("Pyodide 로딩 중… 처음이면 30초 정도 걸릴 수 있습니다.");
      const pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.4/full/" });
      pyodide.setStdout({ batched: (text) => log(statusEl.textContent + "\n" + text) });
      log("NumPy / pandas / matplotlib 설치 중…");
      await pyodide.loadPackage(["numpy", "pandas", "matplotlib"]);
      mkdirp(pyodide, "/pkg/ctf_lpbf");
      mkdirp(pyodide, "/pkg/tests");
      mkdirp(pyodide, "/pkg/web");
      for (const rel of PY_FILES) {
        log(`소스 적재: ${rel}`);
        const text = await fetchText(rel);
        pyodide.FS.writeFile(`/pkg/${rel}`, text);
      }
      await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/pkg")
import importlib.util
spec = importlib.util.spec_from_file_location("pyodide_app", "/pkg/web/pyodide_app.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
globals()["pyodide_app"] = mod
`);
      log("준비 완료. 테스트를 실행할 수 있습니다.");
      return pyodide;
    })();
  }
  return pyodidePromise;
}

function params() {
  const red = Number(sliders.redWeight.value);
  const green = Number(sliders.greenWeight.value);
  return {
    grid: Number(sliders.grid.value),
    red_weight: red,
    green_weight: green,
    noise_fraction: Number(sliders.noise.value),
    mad_threshold: Number(sliders.mad.value),
    isolation_threshold_deg: Number(sliders.isolation.value),
  };
}

function renderSummary(rows) {
  if (!rows || !rows.length) return "";
  const keys = Object.keys(rows[0]);
  const head = keys.map((k) => `<th>${k}</th>`).join("");
  const body = rows
    .map((row) => `<tr>${keys.map((k) => `<td>${row[k] ?? ""}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderImages(images) {
  if (!images || !images.length) return "";
  return `<div class="gallery">${images
    .map(
      (img) => `<figure>
        <img alt="${img.title}" src="data:image/png;base64,${img.b64}">
        <figcaption>${img.title}</figcaption>
      </figure>`
    )
    .join("")}</div>`;
}

function renderAssertions(items) {
  if (!items || !items.length) return "";
  return `<ul class="checks">${items
    .map(
      (item) => `<li>
        <span>${item.name}</span>
        <strong class="${item.passed ? "pass" : "fail"}">${item.passed ? "통과" : "실패"} · ${item.detail}</strong>
      </li>`
    )
    .join("")}</ul>`;
}

function showResult(result) {
  errorEl.classList.add("hidden");
  errorEl.textContent = "";
  if (!result.ok) {
    errorEl.classList.remove("hidden");
    errorEl.textContent = result.error || "알 수 없는 오류";
  }
  resultsEl.innerHTML = `
    ${renderAssertions(result.assertions)}
    ${renderSummary(result.summary)}
    ${renderImages(result.images)}
  `;
  if (result.log) {
    log(result.log.trim());
  }
}

async function writeUploads(pyodide) {
  const files = [...fileInput.files];
  if (!files.length) {
    throw new Error("CTF 파일을 하나 이상 선택하세요.");
  }
  await pyodide.runPythonAsync(`
from pathlib import Path
import shutil
p = Path("/tmp/ctf_upload/data")
if p.exists():
    shutil.rmtree(p)
p.mkdir(parents=True)
`);
  mkdirp(pyodide, "/tmp/ctf_upload/data/uploaded");
  for (const file of files) {
    const buf = new Uint8Array(await file.arrayBuffer());
    pyodide.FS.writeFile(`/tmp/ctf_upload/data/uploaded/${file.name}`, buf);
  }
}

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  errorEl.classList.add("hidden");
  try {
    const pyodide = await ensurePyodide();
    if (mode === "upload") {
      await writeUploads(pyodide);
    }
    log("파이프라인 실행 중… 브라우저 WASM이라 1~2분 걸릴 수 있습니다.");
    pyodide.FS.writeFile("/tmp/params.json", JSON.stringify(params()));
    const py =
      mode === "smoke"
        ? "pyodide_app.run_smoke()"
        : mode === "upload"
          ? "pyodide_app.run_uploads(open('/tmp/params.json', encoding='utf-8').read())"
          : "pyodide_app.run_demo(open('/tmp/params.json', encoding='utf-8').read())";
    const raw = await pyodide.runPythonAsync(py);
    showResult(JSON.parse(raw));
  } catch (err) {
    errorEl.classList.remove("hidden");
    errorEl.textContent = formatErr(err);
    log(formatErr(err));
  } finally {
    runBtn.disabled = false;
  }
});

setMode("demo");
log("실행을 누르면 Cursor/GitHub Pages에서 파이프라인을 브라우저로 검증합니다.");
