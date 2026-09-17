"""Pyodide에서 호출되는 브라우저 테스트 러너."""
from __future__ import annotations

import base64
import io
import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path("/pkg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.generate_synthetic_ctf import generate_synthetic_ctf  # noqa: E402
from ctf_lpbf.pipeline import run  # noqa: E402


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _collect_images(out_dir: Path) -> list[dict]:
    images = []
    for path in sorted(out_dir.rglob("*.png")):
        images.append(
            {
                "title": str(path.relative_to(out_dir)).replace("\\", "/"),
                "b64": _b64(path),
            }
        )
    return images


def _summary_rows(summary_path: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(summary_path)
    return json.loads(df.to_json(orient="records"))


def _reset_dir(path: Path) -> Path:
    import shutil

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_demo(params_json: str) -> str:
    params = json.loads(params_json or "{}")
    n = int(params.get("grid", 48))
    red = float(params.get("red_weight", 0.6))
    green = float(params.get("green_weight", 0.25))
    noise = float(params.get("noise_fraction", 0.02))
    mad = float(params.get("mad_threshold", 1.0))
    isolation = float(params.get("isolation_threshold_deg", 10.0))
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            data = _reset_dir(Path("/tmp/ctf_demo/data"))
            out = _reset_dir(Path("/tmp/ctf_demo/out"))
            generate_synthetic_ctf(
                data / "67deg" / "sample1.ctf",
                n_x=n,
                n_y=n,
                red_weight=red,
                green_weight=green,
                random_weight=max(0.0, 1.0 - red - green),
                isolated_noise_fraction=noise,
                seed=1,
            )
            generate_synthetic_ctf(
                data / "90deg" / "sample1.ctf",
                n_x=n,
                n_y=n,
                red_weight=max(0.05, 1.0 - red - 0.15),
                green_weight=red,
                random_weight=0.15,
                isolated_noise_fraction=noise,
                seed=3,
            )
            summary_path = run(
                data,
                out,
                mad_threshold=mad,
                isolation_threshold_deg=isolation,
                neighbor_coherence_deg=5.0,
            )
            rows = _summary_rows(summary_path)
        return json.dumps(
            {
                "ok": True,
                "mode": "demo",
                "log": buf.getvalue(),
                "summary": rows,
                "images": _collect_images(out),
            }
        )
    except Exception:
        return json.dumps({"ok": False, "error": traceback.format_exc(), "log": buf.getvalue()})


def run_smoke() -> str:
    buf = io.StringIO()
    assertions: list[dict] = []
    try:
        with redirect_stdout(buf):
            data = _reset_dir(Path("/tmp/ctf_smoke/data"))
            out = _reset_dir(Path("/tmp/ctf_smoke/out"))
            generate_synthetic_ctf(
                data / "67deg" / "sample1.ctf",
                n_x=48,
                n_y=48,
                red_weight=0.6,
                green_weight=0.25,
                random_weight=0.15,
                seed=1,
            )
            generate_synthetic_ctf(
                data / "67deg" / "sample2.ctf",
                n_x=48,
                n_y=48,
                red_weight=0.55,
                green_weight=0.3,
                random_weight=0.15,
                seed=2,
            )
            generate_synthetic_ctf(
                data / "90deg" / "sample1.ctf",
                n_x=48,
                n_y=48,
                red_weight=0.25,
                green_weight=0.55,
                random_weight=0.2,
                seed=3,
            )
            summary_path = run(data, out, isolation_threshold_deg=10.0, neighbor_coherence_deg=5.0)

            import pandas as pd

            df = pd.read_csv(summary_path)

            def check(name: str, passed: bool, detail: str) -> None:
                assertions.append({"name": name, "passed": bool(passed), "detail": detail})
                if not passed:
                    raise AssertionError(f"{name}: {detail}")

            check("파일 3개 처리", len(df) == 3, f"got {len(df)}")
            rgb_sum = (
                df["red_area_fraction"].fillna(0)
                + df["green_area_fraction"].fillna(0)
                + df["blue_area_fraction"].fillna(0)
            )
            check("R+G+B 면적분율 <= 1", bool((rgb_sum <= 1.0001).all()), rgb_sum.to_string())
            check(
                "artifact_fraction > 0",
                bool((df["artifact_fraction"] > 0).all()),
                df["artifact_fraction"].to_string(),
            )
            red_67 = float(df.loc[df["condition"] == "67deg", "red_area_fraction"].mean())
            red_90 = float(df.loc[df["condition"] == "90deg", "red_area_fraction"].mean())
            check(
                "67deg red > 90deg red",
                red_67 > red_90,
                f"67deg={red_67:.3f}, 90deg={red_90:.3f}",
            )
            pngs = list((out / "67deg").glob("*_IPF-Z.png"))
            check("IPF-Z PNG 생성", len(pngs) > 0 and pngs[0].stat().st_size > 0, str(len(pngs)))
            legends = list(out.glob("legend_*.png"))
            check("범례 PNG 생성", len(legends) > 0, str(legends))

        return json.dumps(
            {
                "ok": True,
                "mode": "smoke",
                "log": buf.getvalue(),
                "assertions": assertions,
                "summary": _summary_rows(summary_path),
                "images": _collect_images(out),
            }
        )
    except Exception:
        return json.dumps(
            {
                "ok": False,
                "mode": "smoke",
                "error": traceback.format_exc(),
                "assertions": assertions,
                "log": buf.getvalue(),
            }
        )


def run_uploads(params_json: str) -> str:
    params = json.loads(params_json or "{}")
    mad = float(params.get("mad_threshold", 1.0))
    isolation = float(params.get("isolation_threshold_deg", 10.0))
    buf = io.StringIO()
    try:
        data = Path("/tmp/ctf_upload/data")
        if not any(data.rglob("*.ctf")):
            raise FileNotFoundError("업로드된 .ctf 파일이 없습니다.")
        out = _reset_dir(Path("/tmp/ctf_upload/out"))
        with redirect_stdout(buf):
            summary_path = run(
                data,
                out,
                mad_threshold=mad,
                isolation_threshold_deg=isolation,
                neighbor_coherence_deg=5.0,
            )
        return json.dumps(
            {
                "ok": True,
                "mode": "upload",
                "log": buf.getvalue(),
                "summary": _summary_rows(summary_path),
                "images": _collect_images(out),
            }
        )
    except Exception:
        return json.dumps({"ok": False, "error": traceback.format_exc(), "log": buf.getvalue()})
