"""
visualize.py
------------
IPF-Z 맵 PNG 렌더링(+ 노이즈/아티팩트 오버레이) 및 표준 IPF 삼각형 범례 생성.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# CLI/CI/브라우저(Pyodide) 모두 화면 없이 PNG를 만들 수 있도록 Agg를 기본으로 둔다.
try:
    matplotlib.use("Agg")
except Exception:
    pass

import matplotlib.pyplot as plt

# 한글(조건명/라벨 등)이 그래프에 깨지지 않고 표시되도록 CJK 폰트를 지정한다.
for _font in ("Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC", "NanumGothic", "Malgun Gothic", "AppleGothic"):
    if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _font
        break
matplotlib.rcParams["axes.unicode_minus"] = False

from .cleaning import build_grid_index
from .orientation import (
    _reduce_cubic, _reduce_hexagonal, _triangle_rgb,
    CUBIC_CORNERS, HEXAGONAL_CORNERS,
)

ARTIFACT_COLOR = (0.55, 0.0, 0.55)   # 보라색: scratch/wild-spike 등 아티팩트로 판정된 픽셀
NON_INDEXED_COLOR = (0.05, 0.05, 0.05)  # 검정에 가까운 회색: 비인덱싱 픽셀


def render_ipf_map(
    df: pd.DataFrame,
    x_step: float,
    y_step: float,
    out_path: str | Path,
    title: str = "",
    show_artifacts: bool = True,
) -> Path:
    """포인트별 R,G,B(0~1)와 플래그(is_artifact, flag_non_indexed) 컬럼이 있는 df -> IPF-Z 맵 PNG."""
    grid, n_rows, n_cols = build_grid_index(df, x_step, y_step)
    img = np.ones((n_rows, n_cols, 3))
    valid = grid >= 0
    idx = grid[valid]

    rgb = df[["R", "G", "B"]].to_numpy()
    rgb = np.nan_to_num(rgb, nan=1.0)
    colors = rgb[idx]

    if "flag_non_indexed" in df.columns:
        non_idx_mask = df["flag_non_indexed"].to_numpy()[idx]
        colors[non_idx_mask] = NON_INDEXED_COLOR
    if show_artifacts and "is_artifact" in df.columns:
        artifact_mask = df["is_artifact"].to_numpy()[idx]
        if "flag_non_indexed" in df.columns:
            artifact_mask = artifact_mask & ~df["flag_non_indexed"].to_numpy()[idx]
        colors[artifact_mask] = ARTIFACT_COLOR

    img[valid] = colors

    fig, ax = plt.subplots(figsize=(n_cols / 100 + 2, n_rows / 100 + 2), dpi=150)
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_ipf_triangle_legend(
    crystal_system: str,
    out_path: str | Path,
    n_samples: int = 60000,
    corner_labels: tuple[str, str, str] = ("R: [001]", "G: [101]-type", "B: [111]"),
) -> Path:
    """표준 IPF(Z) 기본영역 삼각형 범례를 산점도 근사로 렌더링."""
    rng = np.random.default_rng(0)
    # 상반구 균일 샘플 (theta: polar from z, phi: azimuth)
    u = rng.uniform(0, 1, n_samples)
    v = rng.uniform(0, 1, n_samples)
    theta = np.arccos(u)  # 0~90deg 사이, cos 균일 -> 구면상 균일
    phi = v * 2 * np.pi
    d = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)

    if crystal_system == "cubic":
        reduced = _reduce_cubic(d)
        corners = CUBIC_CORNERS
    elif crystal_system == "hexagonal":
        # _reduce_hexagonal expects (N,3); 함수 시그니처 재사용
        reduced = _reduce_hexagonal(d)
        corners = HEXAGONAL_CORNERS
    else:
        raise ValueError(f"지원하지 않는 결정계: {crystal_system}")

    rgb = _triangle_rgb(reduced, corners)

    c_up = reduced[:, 2] if crystal_system == "cubic" else reduced[:, 2]
    x2d = reduced[:, 0] / (1 + c_up)
    y2d = reduced[:, 1] / (1 + c_up)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.scatter(x2d, y2d, c=rgb, s=2, marker="s", linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")

    def proj(v3):
        c = v3[2]
        return v3[0] / (1 + c), v3[1] / (1 + c)

    for pt, label, ha, va, dx, dy in zip(
        corners, corner_labels,
        ["right", "left", "left"], ["top", "bottom", "top"],
        [-0.01, 0.01, 0.01], [-0.01, 0.01, 0.01],
    ):
        px, py = proj(pt)
        ax.plot(px, py, "k.", markersize=3)
        ax.annotate(label, (px, py), xytext=(px + dx, py + dy), fontsize=8, ha=ha, va=va)

    ax.set_title(f"IPF-Z legend ({crystal_system})", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
