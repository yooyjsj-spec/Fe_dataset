"""
ctf_lpbf
========
L-PBF(Laser Powder Bed Fusion) 적층체의 CTF(HKL/Oxford Channel Text File, EBSD) 파일을
파싱하여 Z면 미세조직(IPF-Z 텍스처) 경향을 분석하고, 이후 공정조건 -> 미세조직
머신러닝 예측 모델링의 입력 피처를 생성하기 위한 파이프라인.

모듈 구성
---------
ctf_io       : CTF 파일 파싱 (헤더 + Phase 테이블 + 포인트 데이터)
orientation  : Bunge Euler각 -> 회전행렬 -> 결정 대칭 적용 -> IPF-Z RGB 색상 매핑
cleaning     : 비인덱싱/저품질/scratch로 인한 고립 오배향(wild spike) 포인트 필터링
analyze      : 조건별 Red/Green/Blue 면적분율, texture 지표 집계 (ML 피처 테이블 포함)
visualize    : IPF-Z 맵 PNG 렌더링 + 표준 삼각형 범례
pipeline     : 위 모듈을 엮어 폴더 단위(조건별)로 일괄 처리하는 CLI
"""

__version__ = "0.1.0"
