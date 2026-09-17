"""
analyze.py
----------
정제된 CTF 포인트 데이터에 IPF-Z 색상을 계산해 붙이고, 조건(예: 67도)별
Red/Green/Blue 경향을 정량화한 요약 테이블(추후 ML 피처 테이블의 기초)을 만든다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ctf_io import CTFData
from .orientation import ipf_z_colors
from .cleaning import clean_dataframe


def add_ipf_colors(df: pd.DataFrame, phases: dict) -> pd.DataFrame:
    """상(phase)별로 IPF-Z RGB를 계산해 R,G,B(0~1), dominant(0=R/1=G/2=B) 컬럼을 추가."""
    out = df.copy()
    out["R"] = np.nan
    out["G"] = np.nan
    out["B"] = np.nan
    for phase_idx, phase in phases.items():
        mask = out["Phase"] == phase_idx
        if not mask.any():
            continue
        rgb = ipf_z_colors(
            out.loc[mask, "Euler1"].to_numpy(),
            out.loc[mask, "Euler2"].to_numpy(),
            out.loc[mask, "Euler3"].to_numpy(),
            phase.laue_group,
        )
        out.loc[mask, ["R", "G", "B"]] = rgb
    # 비인덱싱 등으로 R,G,B가 모두 NaN인 행은 idxmax 대상에서 제외 (그런 행은 어차피
    # flag_non_indexed/is_artifact로 이후 분석에서 걸러진다).
    out["dominant"] = pd.Series(pd.NA, index=out.index, dtype="object")
    has_color = out[["R", "G", "B"]].notna().any(axis=1)
    out.loc[has_color, "dominant"] = out.loc[has_color, ["R", "G", "B"]].idxmax(axis=1)
    return out


def process_ctf(
    ctf: CTFData,
    condition: str,
    mad_threshold: float = 1.0,
    bc_threshold: float | None = None,
    isolation_threshold_deg: float = 10.0,
    neighbor_coherence_deg: float = 5.0,
    detect_spikes: bool = True,
) -> pd.DataFrame:
    """단일 CTFData -> 정제 플래그 + IPF 색상까지 붙인 전체 포인트 DataFrame."""
    laue_lookup = {p.index: p.laue_group for p in ctf.phases.values()}
    main_phase = ctf.df.loc[ctf.df["Phase"] != 0, "Phase"].mode()
    main_laue = laue_lookup.get(int(main_phase.iloc[0])) if not main_phase.empty else 11

    cleaned = clean_dataframe(
        ctf.df,
        x_step=ctf.x_step or 1.0,
        y_step=ctf.y_step or 1.0,
        laue_group=main_laue,
        mad_threshold=mad_threshold,
        bc_threshold=bc_threshold,
        isolation_threshold_deg=isolation_threshold_deg,
        neighbor_coherence_deg=neighbor_coherence_deg,
        detect_spikes=detect_spikes,
    )
    colored = add_ipf_colors(cleaned, ctf.phases)
    colored["condition"] = condition
    colored["source_file"] = ctf.path.name
    return colored


def summarize_condition(df: pd.DataFrame) -> dict:
    """조건 하나(=한 CTF 또는 한 조건으로 묶인 여러 CTF)의 요약 통계 (ML 피처 후보)."""
    total = len(df)
    indexed = df.loc[~df["flag_non_indexed"]]
    clean = df.loc[~df["is_artifact"]]
    n_indexed = len(indexed)
    n_clean = len(clean)

    def frac(mask_len):
        return mask_len / total if total else np.nan

    dom_counts_clean = clean["dominant"].value_counts()
    row = {
        "condition": df["condition"].iloc[0] if "condition" in df and len(df) else None,
        "n_points_total": total,
        "n_indexed": n_indexed,
        "indexed_fraction": frac(n_indexed),
        "n_artifact": int(df["is_artifact"].sum()),
        "artifact_fraction": frac(int(df["is_artifact"].sum())),
        "n_clean": n_clean,
        "clean_fraction": frac(n_clean),
        # clean(비오염) 포인트 중 dominant color 면적분율 -> 핵심 트렌드 지표
        "red_area_fraction": dom_counts_clean.get("R", 0) / n_clean if n_clean else np.nan,
        "green_area_fraction": dom_counts_clean.get("G", 0) / n_clean if n_clean else np.nan,
        "blue_area_fraction": dom_counts_clean.get("B", 0) / n_clean if n_clean else np.nan,
        # 연속형 평균 채널 강도 (0~1) - 이산 분류보다 부드러운 텍스처 지표
        "mean_R": clean["R"].mean() if n_clean else np.nan,
        "mean_G": clean["G"].mean() if n_clean else np.nan,
        "mean_B": clean["B"].mean() if n_clean else np.nan,
        "mean_MAD": indexed["MAD"].mean() if "MAD" in indexed and n_indexed else np.nan,
        "mean_BC": indexed["BC"].mean() if "BC" in indexed and n_indexed else np.nan,
    }
    return row


def build_summary_table(processed: list[pd.DataFrame]) -> pd.DataFrame:
    """process_ctf() 결과 리스트 -> 조건별 요약 테이블 (ML 피처 테이블의 토대).

    이후 실제 공정 파라미터(레이저 파워, 스캔속도, 해치간격, 스캔각도 등)를
    'condition' 기준으로 조인해 X(피처)로, 이 요약 지표들을 y(타깃) 또는
    추가 피처로 사용하는 ML 모델링 단계로 확장하면 된다.
    """
    rows = [summarize_condition(df) for df in processed]
    return pd.DataFrame(rows)


def save_outputs(processed: list[pd.DataFrame], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary_table(processed)
    summary_path = out_dir / "condition_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path
