# ctf_lpbf — L-PBF 적층체 Z면 CTF(EBSD) 미세조직 경향 분석 파이프라인

CTF(HKL/Oxford Channel Text File) 파일을 파싱해 **IPF-Z 텍스처(Red=<001>, Green=<101>류,
Blue=<111>) 경향**을 조건별로 정량화하고, scratch로 인한 grain 내부 고립 오배향(Blue 오염)을
식별·제거한 뒤, 향후 공정조건 → 미세조직 ML 예측 모델링의 입력 피처 테이블을 만드는 파이프라인입니다.

## 왜 이렇게 만들었나 (설계 근거)

- **Z축 & 67° 조건부터 시작**: 응고가 끝난 최상단면이라 텍스처가 가장 뚜렷하고, 67° 스캔
  회전각이 가장 열분배가 안정적이라는 실험적 판단에 따라 이 조건을 기준 파이프라인으로 먼저
  검증하고, 이후 90°/180°로 조건 폴더만 추가하면 되도록 설계했습니다(폴더 = 조건).
- **Red/Green 발달 = <001>/<101>류 파이버 텍스처**: L-PBF는 melt pool을 따라 최대 열류
  방향(대략 build 방향)으로 dendrite가 epitaxial하게 성장하면서 <001> 우선방위(빨강)가
  발달하는 것이 전형적입니다. 조건별로 Red/Green 면적분율이 어떻게 바뀌는지가 핵심 지표입니다.
- **Blue = scratch 오염**이라는 가정은 코드에 다음과 같이 반영했습니다: 실제 결정 방위가
  아니라 "측정 오류"라는 전제이므로, 색상 자체(파란 픽셀)를 지우는 게 아니라 **원인(저품질
  패턴, 고립된 단일픽셀 오배향)을 판별**해 마스킹합니다. 색만 보고 필터링하면 진짜 <111>
  방위 결정립까지 지워버리는 오류를 범하기 때문입니다.

## 폴더 구조

```
ctf_lpbf/
  ctf_io.py       CTF 파서 (헤더 + Phase 테이블 + 포인트 데이터 -> DataFrame)
  orientation.py  Euler각 -> IPF-Z 색상 (결정 대칭: cubic m-3m, hexagonal 6/mmm)
  cleaning.py     비인덱싱/저품질(MAD,BC)/고립오배향(wild-spike) 필터링
  analyze.py      조건별 R/G/B 면적분율 등 요약 테이블(ML 피처 테이블의 토대)
  visualize.py    IPF-Z 맵 PNG + 표준 삼각형 범례
  pipeline.py     폴더(조건별) 일괄 처리 CLI
tests/
  generate_synthetic_ctf.py  합성 CTF 생성기 (실제 파일 없이도 파이프라인 검증 가능)
  run_smoke_test.py          end-to-end 검증 스크립트 (통과 확인됨)
```

## 웹에서 테스트 (Cursor / GitHub)

로컬에 Python이 없어도 브라우저에서 합성 CTF를 만들어 파이프라인을 검증할 수 있습니다.

### Cursor

1. 터미널에서 저장소 루트로 이동한 뒤 정적 서버를 켭니다.

```bash
npx --yes serve -l 8000 .
```

2. Cursor **Simple Browser** 또는 일반 브라우저에서 [http://localhost:8000](http://localhost:8000) 을 엽니다.
3. **스모크 테스트** 탭 → 실행. 67°가 90°보다 Red 면적분율이 높은지, IPF-Z/범례 PNG가 생기는지 웹에서 바로 확인합니다.

Command Palette에서 `Tasks: Run Task` → **웹 테스트 페이지 열기** 로도 같은 서버를 띄울 수 있습니다.

### GitHub

- **Actions**: `main`에 푸시하면 [CI](.github/workflows/ci.yml)가 `tests/run_smoke_test.py`를 실행합니다. 결과 PNG/CSV는 아티팩트로 내려받을 수 있습니다.
- **Pages**: 저장소가 public이면 같은 웹 UI가 GitHub Pages로 배포됩니다. 지금 저장소는 private이라 GitHub Free 플랜에서는 Pages가 비활성화됩니다. public으로 바꾸면 `https://yooyjsj-spec.github.io/Fe_dataset/` 에서 열 수 있습니다.
- **Codespaces**: 이 저장소에서 Codespace를 만들면 8000 포트 미리보기로 같은 페이지가 열립니다.

실제 장비 CTF는 웹 UI의 **CTF 업로드** 탭에 `.ctf` 파일을 넣으면 됩니다.

## 사용법

### 1) 실제 CTF 파일 준비
조건별로 하위 폴더를 만들어 그 안에 `.ctf` 파일을 넣습니다.

```
data/
  67deg/
    sample1.ctf
    sample2.ctf
  90deg/
    sample1.ctf
  180deg/
    sample1.ctf
```

### 2) 실행

```bash
cd ctf_lpbf
pip install -r requirements.txt   # numpy pandas matplotlib
python -m ctf_lpbf.pipeline --data-dir ./data --out-dir ./results
```

결과:
- `results/<조건>/<파일명>_IPF-Z.png` : 결정립 IPF-Z 맵 (보라색 = 아티팩트로 판정된 픽셀,
  검정 = 비인덱싱)
- `results/legend_<결정계>.png` : 색상 범례
- `results/<조건>/<파일명>_points_cleaned.csv.gz` : 포인트별 원본 데이터 + 정제 플래그 +
  R,G,B 값 (분석/재현용 원자료)
- `results/condition_summary.csv` : **조건별 요약 테이블** — 아래 참고

### 3) 파라미터 튜닝

실제 파일을 받으면 MAD/BC 히스토그램을 보고 아래 값들을 조정하는 것을 권장합니다
(장비/시편마다 품질 분포가 다릅니다):

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `--mad-threshold` | 1.0° | 이보다 MAD 큰 포인트는 저품질로 간주 |
| `--bc-threshold` | 자동(평균-2σ) | Band Contrast 하한 |
| `--isolation-threshold-deg` | 10° | 중심 픽셀-이웃 오배향이 이보다 크면 "튐" 후보 |
| `--neighbor-coherence-deg` | 5° | 이웃끼리 오배향이 이보다 작아야 "일관된 주변"으로 인정 |

`--isolation-threshold-deg`를 낮추면 더 많은 픽셀을 노이즈로 잡지만(민감도↑) 실제 저각
subgrain 경계를 노이즈로 오판할 위험도 커집니다. 합성 데이터 검증에서는 기본값이 실제
grain 경계를 건드리지 않으면서 고립 노이즈만 잡는 것을 확인했습니다.

## condition_summary.csv 컬럼

| 컬럼 | 의미 |
|---|---|
| `condition` | 조건 라벨 (폴더명, 예: `67deg`) |
| `indexed_fraction` | 전체 대비 인덱싱 성공 비율 |
| `artifact_fraction` | 전체 대비 아티팩트(비인덱싱+저품질+wild-spike) 비율 |
| `red_area_fraction` / `green_area_fraction` / `blue_area_fraction` | **정제된(clean) 포인트 중** 각 색이 우세한 픽셀의 면적분율 — 조건별 텍스처 경향 비교의 핵심 지표 |
| `mean_R` / `mean_G` / `mean_B` | 이산 분류 대신 연속값 채널 평균 (더 부드러운 트렌드) |
| `mean_MAD` / `mean_BC` | 데이터 품질 지표 (조건 간 측정 품질 비교/이상치 점검용) |

## 향후 ML 모델링으로 확장하는 방법

`condition_summary.csv`가 이미 "조건 → 미세조직 지표" 테이블 형태이므로:

1. `condition` 컬럼을 실제 공정 파라미터(레이저 파워, 스캔속도, 해치간격, 스캔각도,
   층두께 등)로 매핑하는 조인 테이블을 추가합니다.
2. 공정 파라미터를 입력(X), `red/green/blue_area_fraction`(또는 `mean_R/G/B`)을
   출력(y)으로 하는 회귀 모델(예: Random Forest, Gradient Boosting, 또는 데이터가
   충분히 쌓이면 신경망)을 학습합니다.
3. 샘플 수가 초기에는 적을 것이므로(조건 몇 개 x 파일 몇 개), 처음에는 트렌드 파악
   (지금 이 파이프라인의 목적)에 집중하고, 조건 수가 늘어난 뒤 본격적인 예측 모델로
   전환하는 것을 권장합니다.
4. `*_points_cleaned.csv.gz`에는 포인트 단위 원자료가 남아 있으므로, 추후 grain 단위
   피처(결정립 크기 분포, 형상비, KAM 등)로 확장할 때도 재계산 없이 바로 활용 가능합니다.

## As-built 잔류응력 문제, 어떻게 풀어나갈지

메모하신 "As-built 상태는 잔류응력이 높아서 중요함?" 질문에 대한 정리입니다.

- **EBSD(오일러각/IPF)는 결정 방위만 측정하며, 잔류응력(탄성변형)을 직접 주지 않습니다.**
  다만 잔류응력이 크면 결정격자가 국부적으로 뒤틀려 패턴 품질(Band Contrast, MAD)이
  떨어지고 비인덱싱/저품질 포인트가 늘어나는 **간접 신호**로는 나타납니다. 즉, 지금
  만든 `mean_MAD`/`mean_BC`/`artifact_fraction` 지표를 As-built vs 열처리(응력제거 등)
  시편 간에 비교하면 "잔류응력이 EBSD 측정 품질에 미치는 영향"은 간접적으로 확인할 수
  있습니다.
- 잔류응력 자체를 정량화하려면 별도 방법이 필요합니다: XRD sin²ψ법(가장 보편적, 표면
  잔류응력), HR-EBSD(cross-correlation 기반 탄성변형률 매핑, 고분해능이지만 고급 후처리
  필요), 또는 hole-drilling/contour method(파괴적, 깊이방향 프로파일).
- **연구 전략 제안**: 1) 지금은 As-built 상태로 Z면 텍스처 경향(조건별 Red/Green 발달)을
  먼저 확립하고, 2) 같은 조건에서 응력제거 열처리 전/후 CTF를 추가로 찍어 텍스처 자체는
  거의 안 변하지만(재결정 온도 이하라면) 패턴 품질 지표(MAD/BC/artifact_fraction)가
  개선되는지를 비교하면, "As-built의 높은 잔류응력이 실제로 EBSD 인덱싱 품질/노이즈에
  얼마나 영향을 주는지"를 이 파이프라인 안에서 정량적으로 보여줄 수 있습니다. 이는 별도
  장비(XRD 등) 없이 지금 데이터만으로 낼 수 있는 결론이라 우선순위로 추천합니다.

## 알려진 한계 / 실제 데이터로 검증 시 확인할 점

1. **Hexagonal 상(예: α-Ti)의 방위각 기준**: cubic은 대칭이 방향에 무관하게 축소되어
   변환이 항상 정확하지만, hexagonal은 결정 a축의 절대 방위 기준(장비 소프트웨어의
   결정학적 좌표계 정의)에 따라 삼각형 내 방위각 0°의 위치가 달라질 수 있습니다.
   실제 hexagonal 상 데이터가 들어오면 Aztec Crystal / MTEX 등 검증된 소프트웨어의
   IPF 맵과 나란히 비교해 보정할 것을 권장합니다. Cubic(가장 흔한 L-PBF 금속 상)은
   이 이슈가 없습니다.
2. **Wild-spike 탐지는 보수적으로 설계**했습니다(오탐 방지를 위해 "주변이 실제로
   코히어런트할 때만" 노이즈로 판정) — 따라서 실제 grain 경계 바로 옆에 생긴 scratch
   노이즈는 놓칠 수 있습니다. `--isolation-threshold-deg`를 낮춰 민감도를 올리거나,
   IPF 맵(보라색 오버레이)을 보고 수동으로 추가 확인하는 것을 권장합니다.
3. CTF 컬럼 구성은 장비/버전에 따라 조금씩 다를 수 있어, 파서는 고정 스키마 대신
   "Phase"로 시작하는 헤더 라인을 찾아 그 뒤부터 데이터로 처리하도록 만들었지만,
   실제 파일에서 파싱 에러가 나면 헤더 몇 줄을 공유해 주시면 바로 맞출 수 있습니다.

## 다음 단계

실제 67° 조건 CTF 파일을 주시면:
1. 파서가 실제 헤더 포맷과 맞는지 검증
2. 실제 MAD/BC 분포를 보고 `--mad-threshold`/`--bc-threshold` 기본값 조정
3. 실제 IPF-Z 맵을 보고 상을 확인(cubic/hexagonal), 필요하면 hexagonal 방위각 보정
4. 90°, 180° 데이터가 쌓이면 조건별 트렌드 비교 + ML 피처 테이블 확장
