"""
합성(synthetic) CTF 파일 생성기 — 파이프라인 검증용.

실제 L-PBF 상면(Z면) EBSD에서 흔히 보이는 상황을 모사:
  - <001> 방향(빨강) 위주의 컬럼형 결정립 다수
  - <101>류 방향(초록) 결정립 일부
  - 무작위 방위의 소수 결정립(파랑 근처로 갈 수 있음)
  - 격자 내 고립된 단일 픽셀 노이즈("scratch"로 인한 오배향/저품질 포인트)
  - 가장자리 비인덱싱(Phase=0) 테두리
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


def _direction_to_euler(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """목표 결정방향(시편 Z가 결정좌표계에서 향하는 방향) d -> Bunge Euler(deg). phi1은 자유(랜덤)."""
    d = d / np.linalg.norm(d)
    Phi = np.degrees(np.arccos(np.clip(d[2], -1, 1)))
    if np.sin(np.radians(Phi)) < 1e-6:
        phi2 = 0.0
    else:
        phi2 = np.degrees(np.arctan2(d[0], d[1]))
    phi1 = rng.uniform(0, 360)
    return phi1 % 360, Phi, phi2 % 360


def _perturb(direction: np.ndarray, sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    """direction 주변으로 sigma_deg 정도 무작위로 흔든 단위벡터 반환 (근사)."""
    noise = rng.normal(0, np.radians(sigma_deg), size=3)
    v = direction + noise
    return v / np.linalg.norm(v)


def generate_synthetic_ctf(
    out_path: str | Path,
    n_x: int = 80,
    n_y: int = 80,
    step: float = 0.5,
    n_grains: int = 40,
    red_weight: float = 0.5,
    green_weight: float = 0.3,
    random_weight: float = 0.2,
    isolated_noise_fraction: float = 0.02,
    edge_border_px: int = 2,
    seed: int = 42,
) -> Path:
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)

    # --- 1. Voronoi 스타일 결정립 맵 ---
    seed_xy = rng.integers(0, [n_x, n_y], size=(n_grains, 2))
    xs, ys = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="xy")
    dist = ((xs[..., None] - seed_xy[:, 0]) ** 2 + (ys[..., None] - seed_xy[:, 1]) ** 2)
    grain_id = np.argmin(dist, axis=-1)  # (n_y, n_x)

    type_choice = rng.choice(
        ["red", "green", "random"], size=n_grains,
        p=[red_weight, green_weight, random_weight],
    )
    base_dirs = []
    for t in type_choice:
        if t == "red":
            base_dirs.append(_perturb(np.array([0.0, 0.0, 1.0]), 8.0, rng))
        elif t == "green":
            base_dirs.append(_perturb(np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 8.0, rng))
        else:
            v = rng.normal(size=3)
            base_dirs.append(v / np.linalg.norm(v))
    base_dirs = np.array(base_dirs)

    euler = np.zeros((n_y, n_x, 3))
    for gy in range(n_y):
        for gx in range(n_x):
            gid = grain_id[gy, gx]
            d = _perturb(base_dirs[gid], 2.0, rng)  # 결정립 내부 서브그레인 지터
            euler[gy, gx] = _direction_to_euler(d, rng)

    mad = rng.normal(0.5, 0.12, size=(n_y, n_x)).clip(0.05, None)
    bc = rng.normal(150, 20, size=(n_y, n_x)).clip(10, 255)
    bands = rng.integers(6, 10, size=(n_y, n_x))
    error = np.zeros((n_y, n_x), dtype=int)
    phase = np.ones((n_y, n_x), dtype=int)

    # --- 2. 고립 노이즈 픽셀 (scratch) : 이웃과 무관한 완전 랜덤 방위 + 고MAD/저BC ---
    n_total = n_x * n_y
    n_noise = int(n_total * isolated_noise_fraction)
    flat_positions = rng.choice(n_total, size=n_noise, replace=False)
    noise_mask = np.zeros((n_y, n_x), dtype=bool)
    for pos in flat_positions:
        gy, gx = divmod(pos, n_x)
        # 이웃과 겹치지 않도록(고립성 유지) 이미 노이즈인 이웃이 있으면 skip
        neigh = noise_mask[max(0, gy - 1):gy + 2, max(0, gx - 1):gx + 2]
        if neigh.any():
            continue
        noise_mask[gy, gx] = True
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        euler[gy, gx] = _direction_to_euler(v, rng)
        mad[gy, gx] = rng.uniform(1.3, 2.5)
        bc[gy, gx] = rng.uniform(5, 40)

    # --- 3. 가장자리 비인덱싱 테두리 ---
    if edge_border_px > 0:
        phase[:edge_border_px, :] = 0
        phase[-edge_border_px:, :] = 0
        phase[:, :edge_border_px] = 0
        phase[:, -edge_border_px:] = 0

    # --- 4. CTF 텍스트 작성 ---
    lines = []
    lines.append("Channel Text File")
    lines.append("Prj\tsynthetic_test.cpr")
    lines.append("Author\tctf_lpbf synthetic generator")
    lines.append("JobMode\tGrid")
    lines.append(f"XCells\t{n_x}")
    lines.append(f"YCells\t{n_y}")
    lines.append(f"XStep\t{step}")
    lines.append(f"YStep\t{step}")
    lines.append("AcqE1\t0")
    lines.append("AcqE2\t0")
    lines.append("AcqE3\t0")
    lines.append(
        "Euler angles refer to Sample Coordinate system (CS0)!\tMag\t1000\tCoverage\t100\t"
        "Device\t0\tKV\t20\tTiltAngle\t70\tTiltAxis\t0"
    )
    lines.append("Phases\t1")
    lines.append("3.5200;3.5200;3.5200\t90.000;90.000;90.000\tAustenite (synthetic FCC)\t11\t225\t\t\t")
    lines.append("Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS")

    for gy in range(n_y):
        for gx in range(n_x):
            X = gx * step
            Y = gy * step
            p = int(phase[gy, gx])
            if p == 0:
                lines.append(f"0\t{X:.4f}\t{Y:.4f}\t0\t0\t0.0000\t0.0000\t0.0000\t0.000\t0\t0")
            else:
                e1, e2, e3 = euler[gy, gx]
                lines.append(
                    f"{p}\t{X:.4f}\t{Y:.4f}\t{int(bands[gy,gx])}\t{int(error[gy,gx])}\t"
                    f"{e1:.4f}\t{e2:.4f}\t{e3:.4f}\t{mad[gy,gx]:.3f}\t{int(bc[gy,gx])}\t50"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return out_path


if __name__ == "__main__":
    p = generate_synthetic_ctf("/home/claude/ctf_lpbf/tests/data/67deg/synthetic_67deg_sample1.ctf")
    print(f"generated: {p}")
