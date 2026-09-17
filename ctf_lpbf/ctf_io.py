"""
ctf_io.py
---------
HKL/Oxford Instruments "Channel Text File" (.ctf) 파서.

CTF 파일 구조 (표준):
    1행                       : "Channel Text File" (시그니처)
    이후 key\\tvalue 형태의 헤더 라인들
        Prj, Author, JobMode, XCells, YCells, XStep, YStep,
        AcqE1/E2/E3, Euler angles refer to ... (촬영 조건: Mag, Coverage,
        Device, KV, TiltAngle, TiltAxis 등)
    "Phases\\t<N>"            : 이후 N줄에 상(phase) 정의
        <a;b;c>\\t<alpha;beta;gamma>\\t<Phase name>\\t<LaueGroup>\\t<SpaceGroup>\\t...
    데이터 컬럼 헤더 라인      : "Phase\\tX\\tY\\tBands\\tError\\tEuler1\\tEuler2\\tEuler3\\tMAD\\tBC\\tBS[...]"
    이후 포인트별 데이터 (X,Y 순서로 정렬된 격자, XCells x YCells 개)

주의: 장비/소프트웨어(Aztec, Channel5, Tango 등) 버전에 따라 헤더 키가 조금씩
다르거나 데이터 컬럼이 추가(GB, EDS 관련 등)될 수 있어, 고정 스키마 대신
"Phase"로 시작하는 컬럼 헤더 라인을 탐색해 그 뒤부터 데이터로 처리한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Oxford/HKL CTF LaueGroup 코드 -> (결정계 라벨, 대칭점군)
# MTEX의 CTF 임포터가 쓰는 코드 테이블과 동일 (실무에서 가장 널리 쓰이는 매핑).
LAUE_GROUP_TABLE = {
    1: ("triclinic", "-1"),
    2: ("monoclinic", "2/m"),
    3: ("orthorhombic", "mmm"),
    4: ("tetragonal_low", "4/m"),
    5: ("tetragonal_high", "4/mmm"),
    6: ("trigonal_low", "-3"),
    7: ("trigonal_high", "-3m"),
    8: ("hexagonal_low", "6/m"),
    9: ("hexagonal_high", "6/mmm"),
    10: ("cubic_low", "m-3"),
    11: ("cubic_high", "m-3m"),
}


@dataclass
class CTFPhase:
    index: int
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    name: str
    laue_group: int
    space_group: Optional[int] = None

    @property
    def crystal_system(self) -> str:
        return LAUE_GROUP_TABLE.get(self.laue_group, ("unknown", "unknown"))[0]

    @property
    def point_group(self) -> str:
        return LAUE_GROUP_TABLE.get(self.laue_group, ("unknown", "unknown"))[1]


@dataclass
class CTFData:
    path: Path
    header: dict = field(default_factory=dict)
    phases: dict[int, CTFPhase] = field(default_factory=dict)
    x_cells: Optional[int] = None
    y_cells: Optional[int] = None
    x_step: Optional[float] = None
    y_step: Optional[float] = None
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:
        n = len(self.df)
        phases = ", ".join(f"{p.index}:{p.name}({p.crystal_system})" for p in self.phases.values())
        return (
            f"CTFData('{self.path.name}', points={n}, "
            f"grid={self.x_cells}x{self.y_cells}, step={self.x_step}, phases=[{phases}])"
        )


def _split_line(line: str) -> list[str]:
    # CTF는 탭 구분이 표준이지만, 일부 export는 다중 공백을 섞어 쓰기도 해서
    # 탭 우선 분리 후 빈 토큰을 정리한다.
    parts = line.rstrip("\n\r").split("\t")
    if len(parts) == 1:
        parts = re.split(r"\s{1,}", line.strip())
    return [p.strip() for p in parts]


def parse_ctf(path: str | Path) -> CTFData:
    """CTF 파일을 읽어 CTFData(헤더/phase/좌표별 DataFrame)로 반환."""
    path = Path(path)
    raw_lines = path.read_text(encoding="latin-1", errors="replace").splitlines()

    if not raw_lines or "Channel Text File" not in raw_lines[0]:
        raise ValueError(
            f"'{path.name}'가 표준 CTF 시그니처('Channel Text File')로 시작하지 않습니다. "
            "올바른 CTF 파일인지 확인하세요."
        )

    header: dict = {}
    phases: dict[int, CTFPhase] = {}
    data_header_idx = None
    n_phases_declared = 0
    i = 1
    while i < len(raw_lines):
        line = raw_lines[i]
        if not line.strip():
            i += 1
            continue
        tokens = _split_line(line)

        # 데이터 테이블의 컬럼 헤더 라인을 만나면 헤더 파싱 종료
        if tokens[0] == "Phase" and "X" in tokens and "Y" in tokens:
            data_header_idx = i
            data_columns = tokens
            break

        if tokens[0] == "Phases":
            n_phases_declared = int(tokens[1]) if len(tokens) > 1 else 0
            # 다음 n_phases_declared 줄이 phase 정의
            for k in range(1, n_phases_declared + 1):
                if i + k >= len(raw_lines):
                    break
                ptoks = _split_line(raw_lines[i + k])
                try:
                    a, b, c = (float(v) for v in ptoks[0].split(";"))
                    alpha, beta, gamma = (float(v) for v in ptoks[1].split(";"))
                    name = ptoks[2]
                    laue = int(ptoks[3])
                    sg = int(ptoks[4]) if len(ptoks) > 4 and ptoks[4].strip().isdigit() else None
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        f"{path.name}: {i + k + 1}번째 줄의 Phase 정의를 해석할 수 없습니다: "
                        f"{raw_lines[i + k]!r} ({exc})"
                    ) from exc
                phases[k] = CTFPhase(
                    index=k, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                    name=name, laue_group=laue, space_group=sg,
                )
            i += n_phases_declared + 1
            continue

        # 그 외 일반 key-value / key-multi-value 헤더 라인
        key = tokens[0]
        header[key] = tokens[1:] if len(tokens) > 2 else (tokens[1] if len(tokens) == 2 else "")
        i += 1

    if data_header_idx is None:
        raise ValueError(
            f"{path.name}: 데이터 컬럼 헤더('Phase\\tX\\tY\\t...')를 찾지 못했습니다. "
            "CTF 파일이 손상되었거나 알려지지 않은 변형 포맷일 수 있습니다."
        )

    # 데이터 본문 읽기
    from io import StringIO

    body_text = "\n".join(raw_lines[data_header_idx + 1:])
    df = pd.read_csv(
        StringIO(body_text),
        sep=r"\t|\s{2,}",
        engine="python",
        names=data_columns,
        header=None,
        na_values=["", "NaN"],
    )
    # 컬럼명을 표준화 (앞뒤 공백 제거)
    df.columns = [c.strip() for c in df.columns]
    numeric_cols = [c for c in df.columns if c != "Phase" or True]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    df["Phase"] = df["Phase"].astype(int)

    def _get_num(key, cast=float):
        v = header.get(key)
        if v is None:
            return None
        if isinstance(v, list):
            v = v[0]
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    ctf = CTFData(
        path=path,
        header=header,
        phases=phases,
        x_cells=_get_num("XCells", int),
        y_cells=_get_num("YCells", int),
        x_step=_get_num("XStep", float),
        y_step=_get_num("YStep", float),
        df=df,
    )
    return ctf


def summarize(ctf: CTFData) -> str:
    """빠른 확인용 요약 문자열."""
    lines = [repr(ctf), ""]
    lines.append("[Phases]")
    for p in ctf.phases.values():
        lines.append(
            f"  {p.index}: {p.name}  a,b,c=({p.a},{p.b},{p.c})  "
            f"laue_group={p.laue_group} -> {p.crystal_system} ({p.point_group})"
        )
    lines.append("")
    lines.append("[Phase point counts]")
    counts = ctf.df["Phase"].value_counts().sort_index()
    for phase_idx, cnt in counts.items():
        name = ctf.phases.get(int(phase_idx))
        name = name.name if name else ("Non-indexed" if phase_idx == 0 else f"phase{phase_idx}")
        lines.append(f"  Phase {phase_idx} ({name}): {cnt} pts ({cnt/len(ctf.df):.1%})")
    return "\n".join(lines)
