"""
orientation.py
---------------
Bunge Euler각(phi1, Phi, phi2; deg) -> 결정 좌표계 기준 시편 Z축 방향
-> 결정 대칭 적용(기본 영역으로 축소) -> 표준 IPF(Inverse Pole Figure) 삼각형
색상(R,G,B) 매핑.

규약
----
* Bunge(ZXZ) 오일러각, HKL/Oxford CTF의 표준 규약을 따름:
  g(phi1,Phi,phi2)는 시편좌표 -> 결정좌표 변환(passive) 회전행렬이며,
  시편 Z축을 결정좌표계로 표현한 방향은 g의 3번째 열
      d = (sin(phi2)sin(Phi), cos(phi2)sin(Phi), cos(Phi))
  로 축약된다 (전체 회전행렬을 계산하지 않아도 IPF-Z에는 이 항만 필요).

* 색상 매핑은 널리 쓰이는 관례를 따름: cubic(m-3m) 기준 R=[001], G=[101]류,
  B=[111]. 결정계는 CTF의 LaueGroup 코드로 식별한다.

* CUBIC(LaueGroup 10, 11)은 |x|,|y|,|z|를 취해 오름차순 정렬하는 것만으로
  m-3m 전체 대칭(48개 원소, 항등원+반전 포함)을 기본영역으로 축소하는 것과
  수학적으로 동일하므로 이 방법을 사용한다 (엄밀해).

* HEXAGONAL(LaueGroup 8, 9)은 622 진성회전군(12개)을 명시적으로 생성해
  각 방향 벡터에 적용한 뒤 기본영역(방위각 0~30°, 상반구) 조건을 만족하는
  대표를 선택한다. 결정 a축의 절대 방위 기준(CTF/장비 소프트웨어의 결정학적
  좌표계 정의)에 따라 방위각 원점이 실제 소프트웨어와 다를 수 있으므로,
  실제 시료로 검증 시 기준 소프트웨어(Aztec Crystal, MTEX 등) 출력과
  나란히 비교할 것을 권장한다.

* 그 외 LaueGroup은 대칭 축소 없이(항등원+반전만 적용) 근사 처리하며,
  결과에 경고를 남긴다 — 필요 시 해당 결정계 전용 대칭군을 추가하면 된다.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np


# ----------------------------------------------------------------------
# IPF-Z 방향 벡터 (결정좌표계)
# ----------------------------------------------------------------------
def euler_to_ipf_direction(phi1_deg: np.ndarray, Phi_deg: np.ndarray, phi2_deg: np.ndarray) -> np.ndarray:
    """Bunge Euler각(deg) 배열 -> 결정좌표계로 표현한 시편 Z축 방향 (N,3), 단위벡터."""
    phi1 = np.radians(phi1_deg)
    Phi = np.radians(Phi_deg)
    phi2 = np.radians(phi2_deg)
    x = np.sin(phi2) * np.sin(Phi)
    y = np.cos(phi2) * np.sin(Phi)
    z = np.cos(Phi)
    d = np.stack([x, y, z], axis=-1)
    norm = np.linalg.norm(d, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return d / norm


# ----------------------------------------------------------------------
# 대칭군 생성
# ----------------------------------------------------------------------
def _cubic_proper_rotations() -> np.ndarray:
    """m-3m 점군의 진성회전 부분군(432), 24개 signed-permutation 행렬(det=+1)."""
    mats = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[i, j] = 1
        for signs in itertools.product([1, -1], repeat=3):
            S = P * np.array(signs)[:, None]
            if abs(np.linalg.det(S) - 1) < 1e-6:
                mats.append(S)
    return np.array(mats)  # (24,3,3)


def _hexagonal_proper_rotations() -> np.ndarray:
    """622 점군(hexagonal high, order 12): z축 6회 회전 x {E, C2_x}."""
    mats = []
    c2x = np.diag([1.0, -1.0, -1.0])
    for n in range(6):
        theta = np.radians(60.0 * n)
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        mats.append(Rz)
        mats.append(Rz @ c2x)
    return np.array(mats)  # (12,3,3)


_SYMMETRY_CACHE: dict[str, np.ndarray] = {}


def symmetry_operators(laue_group: int) -> tuple[np.ndarray, str]:
    """LaueGroup 코드 -> (진성회전 행렬 배열, 결정계 이름). 미지원 코드는 항등원만 반환."""
    if laue_group in (10, 11):
        key = "cubic"
        if key not in _SYMMETRY_CACHE:
            _SYMMETRY_CACHE[key] = _cubic_proper_rotations()
        return _SYMMETRY_CACHE[key], "cubic"
    if laue_group in (8, 9):
        key = "hexagonal"
        if key not in _SYMMETRY_CACHE:
            _SYMMETRY_CACHE[key] = _hexagonal_proper_rotations()
        return _SYMMETRY_CACHE[key], "hexagonal"
    warnings.warn(
        f"LaueGroup {laue_group}에 대한 전용 대칭군이 구현되어 있지 않습니다. "
        "항등원+반전만 적용한 근사 결과입니다 (IPF 색상이 부정확할 수 있음).",
        stacklevel=2,
    )
    return np.eye(3)[None, :, :], "unknown"


# ----------------------------------------------------------------------
# 기본영역(fundamental sector) 축소 + 삼각형 RGB 매핑
# ----------------------------------------------------------------------
def _reduce_cubic(d: np.ndarray) -> np.ndarray:
    """|x|,|y|,|z| 취한 뒤 오름차순 정렬 -> (a<=b<=c), corners: 001,011,111."""
    r = np.sort(np.abs(d), axis=-1)  # ascending: a,b,c
    return r  # already unit length (abs doesn't change norm)


def _reduce_hexagonal(d: np.ndarray) -> np.ndarray:
    """622 대칭(12) x 반전(2) = 24개 후보 중 기본영역(상반구, 방위각 0~30°)에 드는 대표 선택."""
    mats, _ = symmetry_operators(9)
    n = d.shape[0]
    best = np.zeros_like(d)
    candidates = np.einsum("kij,nj->nki", mats, d)  # (n,12,3)
    candidates = np.concatenate([candidates, -candidates], axis=1)  # (n,24,3) incl. inversion
    az = np.degrees(np.arctan2(candidates[..., 1], candidates[..., 0])) % 360.0
    z = candidates[..., 2]
    # 방위각을 0~30 구간으로 접어 넣기 위한 mod-60 후 30 초과분 반사
    az60 = az % 60.0
    az_folded = np.where(az60 <= 30.0, az60, 60.0 - az60)
    in_upper = z >= -1e-9
    score = np.where(in_upper, -az_folded, -1e9)  # 방위각 최소(0에 가까움) & 상반구 우선
    idx = np.argmax(score, axis=1)
    best = candidates[np.arange(n), idx]
    best[..., 2] = np.abs(best[..., 2])
    return best


def _triangle_rgb(reduced: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """구면 대삼각형 무게중심 근사(반대편 대원까지의 부호거리)로 RGB 계산.

    corners: (3,3) 배열, 각 행이 R,G,B에 대응하는 기본영역 꼭짓점(단위벡터).
    """
    A, B, C = corners
    nA = np.cross(B, C); nA /= np.linalg.norm(nA)
    if np.dot(nA, A) < 0:
        nA = -nA
    nB = np.cross(C, A); nB /= np.linalg.norm(nB)
    if np.dot(nB, B) < 0:
        nB = -nB
    nC = np.cross(A, B); nC /= np.linalg.norm(nC)
    if np.dot(nC, C) < 0:
        nC = -nC

    wA = np.clip(reduced @ nA, 0, None)
    wB = np.clip(reduced @ nB, 0, None)
    wC = np.clip(reduced @ nC, 0, None)
    w = np.stack([wA, wB, wC], axis=-1)
    m = w.max(axis=-1, keepdims=True)
    m[m == 0] = 1.0
    rgb = w / m  # 꼭짓점에서 순색(1,0,0) 등이 되도록 정규화
    return np.clip(rgb, 0, 1)


CUBIC_CORNERS = np.array([
    [0.0, 0.0, 1.0],                                   # R: [001]
    [0.0, 1 / np.sqrt(2), 1 / np.sqrt(2)],              # G: [011]/[101]류
    [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],   # B: [111]
])

HEXAGONAL_CORNERS = np.array([
    [0.0, 0.0, 1.0],                                    # R: [0001] (c축, pole)
    [1.0, 0.0, 0.0],                                    # G: 방위각 0° (basal, a축 기준)
    [np.cos(np.radians(30)), np.sin(np.radians(30)), 0.0],  # B: 방위각 30°
])


def euler_to_matrix(phi1_deg: np.ndarray, Phi_deg: np.ndarray, phi2_deg: np.ndarray) -> np.ndarray:
    """Bunge(ZXZ) Euler각(deg) 배열 -> (N,3,3) 회전행렬(시편->결정, passive)."""
    p1 = np.radians(np.asarray(phi1_deg))
    P = np.radians(np.asarray(Phi_deg))
    p2 = np.radians(np.asarray(phi2_deg))
    c1, s1 = np.cos(p1), np.sin(p1)
    c, s = np.cos(P), np.sin(P)
    c2, s2 = np.cos(p2), np.sin(p2)
    g = np.empty(p1.shape + (3, 3))
    g[..., 0, 0] = c1 * c2 - s1 * s2 * c
    g[..., 0, 1] = s1 * c2 + c1 * s2 * c
    g[..., 0, 2] = s2 * s
    g[..., 1, 0] = -c1 * s2 - s1 * c2 * c
    g[..., 1, 1] = -s1 * s2 + c1 * c2 * c
    g[..., 1, 2] = c2 * s
    g[..., 2, 0] = s1 * s
    g[..., 2, 1] = -c1 * s
    g[..., 2, 2] = c
    return g


def disorientation_angle_deg(g1: np.ndarray, g2: np.ndarray, laue_group: int) -> np.ndarray:
    """같은 상(phase) 두 방위(회전행렬 배열, (...,3,3)) 사이의 결정학적 오배향각(deg).

    단측(one-sided) 대칭 축소: delta = g1 @ g2^T, 대칭군 S에 대해 S@delta 중
    최소 회전각을 취한다. 두 방위가 완전히 대칭적으로 동일하면 0°를 반환하도록
    보장하는 것이 목적이며(고립 노이즈 픽셀 오탐 방지), 결정학적으로 엄밀한
    fundamental-zone 최소값(양측 대칭 N^2 조합)은 아니지만 실용적으로 충분하다.
    """
    sym, _ = symmetry_operators(laue_group)
    delta = np.einsum("...ij,...kj->...ik", g1, g2)  # g1 @ g2^T
    cand = np.einsum("sij,...jk->...sik", sym, delta)  # (...,S,3,3)
    tr = np.trace(cand, axis1=-2, axis2=-1)
    cos_ang = np.clip((tr - 1) / 2, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_ang))
    return ang.min(axis=-1)


def ipf_z_colors(phi1_deg: np.ndarray, Phi_deg: np.ndarray, phi2_deg: np.ndarray, laue_group: int) -> np.ndarray:
    """Euler각 배열 + LaueGroup -> (N,3) RGB(0~1) 배열."""
    d = euler_to_ipf_direction(phi1_deg, Phi_deg, phi2_deg)
    if laue_group in (10, 11):
        reduced = _reduce_cubic(d)
        return _triangle_rgb(reduced, CUBIC_CORNERS)
    if laue_group in (8, 9):
        reduced = _reduce_hexagonal(d)
        return _triangle_rgb(reduced, HEXAGONAL_CORNERS)
    # 미지원 결정계: 상반구로만 접어 대략적인 색상(정확도 낮음, 경고는 symmetry_operators에서 발생)
    symmetry_operators(laue_group)
    d2 = d.copy()
    d2[..., 2] = np.abs(d2[..., 2])
    return _triangle_rgb(np.abs(np.sort(d2, axis=-1)), CUBIC_CORNERS)
