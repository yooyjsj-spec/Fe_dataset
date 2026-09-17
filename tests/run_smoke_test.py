"""End-to-end 스모크 테스트: 합성 CTF 2개(조건 67deg/90deg) 생성 -> 파이프라인 실행 -> 검증."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.generate_synthetic_ctf import generate_synthetic_ctf
from ctf_lpbf.pipeline import run

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "out"

if __name__ == "__main__":
    # 조건별로 텍스처 비중을 다르게 하여(67deg: red 위주, 90deg: green 비중 증가)
    # summary 테이블에서 조건 간 차이가 실제로 드러나는지 확인
    generate_synthetic_ctf(
        DATA / "67deg" / "sample1.ctf", n_x=70, n_y=70,
        red_weight=0.6, green_weight=0.25, random_weight=0.15, seed=1,
    )
    generate_synthetic_ctf(
        DATA / "67deg" / "sample2.ctf", n_x=70, n_y=70,
        red_weight=0.55, green_weight=0.3, random_weight=0.15, seed=2,
    )
    generate_synthetic_ctf(
        DATA / "90deg" / "sample1.ctf", n_x=70, n_y=70,
        red_weight=0.25, green_weight=0.55, random_weight=0.2, seed=3,
    )

    summary_path = run(DATA, OUT, isolation_threshold_deg=10.0, neighbor_coherence_deg=5.0)

    import pandas as pd
    df = pd.read_csv(summary_path)
    print("\n=== 검증 ===")
    assert len(df) == 3, f"조건-파일 조합 3개가 나와야 함, got {len(df)}"
    assert (df["red_area_fraction"].fillna(0) + df["green_area_fraction"].fillna(0)
            + df["blue_area_fraction"].fillna(0) <= 1.0001).all()
    assert (df["artifact_fraction"] > 0).all(), "합성 데이터에 노이즈를 주입했으므로 artifact_fraction>0 이어야 함"
    # 67deg 조건들이 90deg보다 red_area_fraction이 더 높아야 함(의도한 텍스처 차이)
    red_67 = df.loc[df["condition"] == "67deg", "red_area_fraction"].mean()
    red_90 = df.loc[df["condition"] == "90deg", "red_area_fraction"].mean()
    print(f"mean red_area_fraction: 67deg={red_67:.3f}, 90deg={red_90:.3f}")
    assert red_67 > red_90, "의도한 대로 67deg 조건이 90deg보다 red 비중이 높아야 함"

    for p in (OUT / "67deg").glob("*_IPF-Z.png"):
        assert p.stat().st_size > 0
    legend = list(OUT.glob("legend_*.png"))
    assert legend, "범례 PNG가 생성되어야 함"

    print("\n모든 스모크 테스트 통과.")
