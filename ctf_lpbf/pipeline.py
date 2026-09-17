"""
pipeline.py
-----------
폴더 단위(조건별) CTF 일괄 처리 CLI.

사용 예:
    python -m ctf_lpbf.pipeline --data-dir ./data --out-dir ./results

기대하는 폴더 구조 (조건 = 하위 폴더명, 예: 스캔각도):
    data/
      67deg/
        sample1.ctf
        sample2.ctf
      90deg/
        sample1.ctf
      180deg/
        ...

또는 단일 폴더에 파일만 있고 조건을 파일명에서 유추하고 싶다면
--condition-regex 로 정규식 그룹을 지정한다 (기본: 폴더명을 조건으로 사용).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .ctf_io import parse_ctf, summarize
from .analyze import process_ctf, build_summary_table
from .visualize import render_ipf_map, plot_ipf_triangle_legend
from .orientation import symmetry_operators


def discover_conditions(data_dir: Path) -> dict[str, list[Path]]:
    """data_dir 바로 아래 하위폴더 = 조건. 하위폴더가 없으면 data_dir 전체를 단일 조건으로 취급."""
    subdirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    if subdirs:
        return {d.name: sorted(d.glob("*.ctf")) for d in subdirs}
    return {data_dir.name: sorted(data_dir.glob("*.ctf"))}


def run(
    data_dir: str | Path,
    out_dir: str | Path,
    mad_threshold: float = 1.0,
    bc_threshold: float | None = None,
    isolation_threshold_deg: float = 10.0,
    neighbor_coherence_deg: float = 5.0,
    detect_spikes: bool = True,
    render_maps: bool = True,
) -> Path:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = discover_conditions(data_dir)
    if not any(files for files in conditions.values()):
        raise FileNotFoundError(f"'{data_dir}' 아래에서 .ctf 파일을 찾지 못했습니다.")

    processed = []
    legends_done = set()
    for condition, files in conditions.items():
        cond_out = out_dir / condition
        cond_out.mkdir(parents=True, exist_ok=True)
        for f in files:
            print(f"[처리 중] condition={condition}  file={f.name}")
            ctf = parse_ctf(f)
            print(summarize(ctf))
            df = process_ctf(
                ctf, condition,
                mad_threshold=mad_threshold,
                bc_threshold=bc_threshold,
                isolation_threshold_deg=isolation_threshold_deg,
                neighbor_coherence_deg=neighbor_coherence_deg,
                detect_spikes=detect_spikes,
            )
            processed.append(df)

            if render_maps:
                png_path = cond_out / f"{f.stem}_IPF-Z.png"
                render_ipf_map(
                    df, ctf.x_step or 1.0, ctf.y_step or 1.0, png_path,
                    title=f"{condition} / {f.name}  (IPF-Z, magenta=artifact)",
                )
                print(f"  -> {png_path}")

                main_phase = ctf.df.loc[ctf.df["Phase"] != 0, "Phase"].mode()
                if not main_phase.empty:
                    phase = ctf.phases.get(int(main_phase.iloc[0]))
                    if phase is not None:
                        _, sym_family = symmetry_operators(phase.laue_group)
                        if sym_family not in legends_done:
                            try:
                                legend_path = out_dir / f"legend_{sym_family}.png"
                                plot_ipf_triangle_legend(sym_family, legend_path)
                                legends_done.add(sym_family)
                                print(f"  -> {legend_path} (범례)")
                            except ValueError:
                                pass

            df.drop(columns=["Bands", "Error"], errors="ignore").to_csv(
                cond_out / f"{f.stem}_points_cleaned.csv.gz", index=False, compression="gzip"
            )

    summary = build_summary_table(processed)
    # 같은 조건에 여러 파일이 있으면 파일별 행이 나오므로, 필요 시 condition으로 groupby하여
    # 평균/표준편차를 내는 것은 사용자가 이 CSV를 열어 추가로 수행하면 된다.
    summary_path = out_dir / "condition_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[요약 테이블] {summary_path}")
    print(summary.to_string(index=False))
    return summary_path


def main():
    ap = argparse.ArgumentParser(description="L-PBF CTF Z-plane IPF 경향 분석 파이프라인")
    ap.add_argument("--data-dir", required=True, help="조건별 하위폴더를 포함한 데이터 폴더")
    ap.add_argument("--out-dir", required=True, help="결과(PNG, CSV) 저장 폴더")
    ap.add_argument("--mad-threshold", type=float, default=1.0)
    ap.add_argument("--bc-threshold", type=float, default=None)
    ap.add_argument("--isolation-threshold-deg", type=float, default=10.0)
    ap.add_argument("--neighbor-coherence-deg", type=float, default=5.0)
    ap.add_argument("--no-spike-detect", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    run(
        args.data_dir, args.out_dir,
        mad_threshold=args.mad_threshold,
        bc_threshold=args.bc_threshold,
        isolation_threshold_deg=args.isolation_threshold_deg,
        neighbor_coherence_deg=args.neighbor_coherence_deg,
        detect_spikes=not args.no_spike_detect,
        render_maps=not args.no_render,
    )


if __name__ == "__main__":
    main()
