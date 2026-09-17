"""
cleaning.py
-----------
Scratch(긁힘)로 인한 grain 내부 고립 오배향 픽셀("Blue 오염")을 포함한
저품질/비인덱싱 포인트를 식별·마스킹하는 모듈.

전략 (여러 플래그를 독립적으로 계산 -> 사용자가 조합/튜닝):
  1. non_indexed   : Phase == 0 (해당 픽셀에서 인덱싱 실패)
  2. low_mad       : MAD(Mean Angular Deviation)가 임계값 초과 (패턴 품질 낮음)
  3. low_bc        : Band Contrast가 임계값 미만 (패턴 대비 낮음, scratch/표면손상 등)
  4. wild_spike    : 격자상 이웃(상하좌우)과의 결정학적 오배향이 모두 크면서
                     그 이웃들끼리는 서로 잘 맞는 경우 -> 해당 픽셀 하나만 튀는
                     "고립점" (전형적인 scratch/미세 손상에 의한 단일픽셀 오염)

실제 임계값(mad_threshold, bc 등)은 장비/시편마다 다르므로, 실제 CTF 파일을
받으면 히스토그램을 보고 튜닝하는 것을 권장한다 (analyze.plot_quality_histograms 참고).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .orientation import euler_to_matrix, disorientation_angle_deg


def flag_non_indexed(df: pd.DataFrame) -> pd.Series:
    return df["Phase"] == 0


def flag_low_mad(df: pd.DataFrame, mad_threshold: float = 1.0) -> pd.Series:
    if "MAD" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["MAD"] > mad_threshold


def flag_low_band_contrast(df: pd.DataFrame, bc_threshold: float | None = None) -> pd.Series:
    """bc_threshold=None이면 (평균 - 2*표준편차)를 자동 임계값으로 사용."""
    if "BC" not in df.columns:
        return pd.Series(False, index=df.index)
    bc = df["BC"]
    if bc_threshold is None:
        bc_threshold = max(0.0, bc.mean() - 2 * bc.std())
    return bc < bc_threshold


def build_grid_index(df: pd.DataFrame, x_step: float, y_step: float) -> tuple[np.ndarray, int, int]:
    """X,Y 물리 좌표 -> (row,col) 격자 인덱스로 변환, df.index를 담은 2D 배열 반환 (없으면 -1)."""
    x0, y0 = df["X"].min(), df["Y"].min()
    col = np.rint((df["X"].to_numpy() - x0) / x_step).astype(int)
    row = np.rint((df["Y"].to_numpy() - y0) / y_step).astype(int)
    n_rows, n_cols = row.max() + 1, col.max() + 1
    grid = np.full((n_rows, n_cols), -1, dtype=np.int64)
    grid[row, col] = np.arange(len(df))
    return grid, n_rows, n_cols


def flag_wild_spikes(
    df: pd.DataFrame,
    x_step: float,
    y_step: float,
    laue_group: int,
    isolation_threshold_deg: float = 10.0,
    neighbor_coherence_deg: float = 5.0,
) -> pd.Series:
    """격자 4-이웃(상하좌우) 기반 고립 오배향 픽셀(wild spike) 탐지.

    조건: (i) 중심 픽셀이 유효한 모든 이웃과 isolation_threshold_deg 초과로
    어긋나고, (ii) 그 이웃들끼리는 서로 neighbor_coherence_deg 이내로 일치
    (=주변은 하나의 결정으로 코히어런트) -> 중심 픽셀만 튀는 경우로 판단.
    """
    g = euler_to_matrix(
        df["Euler1"].to_numpy(), df["Euler2"].to_numpy(), df["Euler3"].to_numpy()
    )
    grid, n_rows, n_cols = build_grid_index(df, x_step, y_step)

    def shifted(dr, dc):
        out = np.full((n_rows, n_cols), -1, dtype=np.int64)
        src_r0, src_r1 = max(0, -dr), n_rows - max(0, dr)
        dst_r0, dst_r1 = max(0, dr), n_rows - max(0, -dr)
        src_c0, src_c1 = max(0, -dc), n_cols - max(0, dc)
        dst_c0, dst_c1 = max(0, dc), n_cols - max(0, -dc)
        out[dst_r0:dst_r1, dst_c0:dst_c1] = grid[src_r0:src_r1, src_c0:src_c1]
        return out

    neighbor_grids = [shifted(-1, 0), shifted(1, 0), shifted(0, -1), shifted(0, 1)]  # up,down,left,right

    n = len(df)
    center_idx = grid.reshape(-1)
    valid_center = center_idx >= 0

    neighbor_angles = []  # 중심-이웃 오배향, shape (4, n_grid_cells)
    neighbor_idx_flat = []
    for ng in neighbor_grids:
        ng_flat = ng.reshape(-1)
        neighbor_idx_flat.append(ng_flat)
        valid_pair = valid_center & (ng_flat >= 0)
        ang = np.full(n_rows * n_cols, np.nan)
        ci = center_idx[valid_pair]
        ni = ng_flat[valid_pair]
        ang[valid_pair] = disorientation_angle_deg(g[ci], g[ni], laue_group)
        neighbor_angles.append(ang)
    neighbor_angles = np.stack(neighbor_angles, axis=0)  # (4, n_grid_cells)

    # 이웃들끼리의 상호 코히어런스: 유효한 이웃 쌍들의 평균 오배향 (완전 벡터화,
    # 셀 단위 파이썬 루프 없이 6개 이웃쌍 조합에 대해 한 번에 계산 - 큰 맵에서도 빠름)
    pair_combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    n_cells = n_rows * n_cols
    pair_angles = np.full((len(pair_combinations), n_cells), np.nan)
    for p, (a, b) in enumerate(pair_combinations):
        idx_a = neighbor_idx_flat[a]
        idx_b = neighbor_idx_flat[b]
        valid_pair = (idx_a >= 0) & (idx_b >= 0)
        if not valid_pair.any():
            continue
        pair_angles[p, valid_pair] = disorientation_angle_deg(
            g[idx_a[valid_pair]], g[idx_b[valid_pair]], laue_group
        )
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        coherence = np.nanmean(pair_angles, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        min_neighbor_angle = np.nanmin(neighbor_angles, axis=0)
    n_valid_neighbors = np.sum(~np.isnan(neighbor_angles), axis=0)

    is_spike_flat = (
        (n_valid_neighbors >= 2)
        & (min_neighbor_angle > isolation_threshold_deg)
        & (coherence < neighbor_coherence_deg)
    )
    is_spike_flat = np.where(valid_center, is_spike_flat, False)

    # flat(grid) 인덱스 -> df 위치 인덱스로 명시적 매핑 (정렬 불일치 방지)
    flat_to_df = -np.ones(n_rows * n_cols, dtype=np.int64)
    flat_to_df[center_idx[valid_center]] = np.nonzero(valid_center)[0]
    out = pd.Series(False, index=df.index)
    spike_cells = np.nonzero(is_spike_flat)[0]
    df_positions = flat_to_df[spike_cells]
    df_positions = df_positions[df_positions >= 0]
    out.iloc[df_positions] = True
    return out


def clean_dataframe(
    df: pd.DataFrame,
    x_step: float,
    y_step: float,
    laue_group: int,
    mad_threshold: float = 1.0,
    bc_threshold: float | None = None,
    isolation_threshold_deg: float = 10.0,
    neighbor_coherence_deg: float = 5.0,
    detect_spikes: bool = True,
) -> pd.DataFrame:
    """각 플래그 컬럼을 추가하고 종합 'is_artifact' 컬럼을 붙인 DataFrame을 반환 (원본은 변경하지 않음)."""
    out = df.copy()
    out["flag_non_indexed"] = flag_non_indexed(out)
    out["flag_low_mad"] = flag_low_mad(out, mad_threshold)
    out["flag_low_bc"] = flag_low_band_contrast(out, bc_threshold)
    if detect_spikes:
        out["flag_wild_spike"] = flag_wild_spikes(
            out, x_step, y_step, laue_group, isolation_threshold_deg, neighbor_coherence_deg
        )
    else:
        out["flag_wild_spike"] = False
    out["is_artifact"] = (
        out["flag_non_indexed"] | out["flag_low_mad"] | out["flag_low_bc"] | out["flag_wild_spike"]
    )
    return out
