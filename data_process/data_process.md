# 데이터 전처리

원본 THumanMV를 학습용 데이터셋으로 변환합니다. 브랜치에 따라 **두 가지 버전**이
필요합니다.

| 데이터셋 | 만드는 법 | 쓰는 브랜치 |
|---|---|---|
| `preprocessed` (rectified) | `--rect` | `gps_gs` |
| `preprocessed_wo_rect` (비정렬) | `--no-rect` | `dav3_2view`, `dav3_4view`, `dav3_4view_with_multiview_supervision` |

`gps_gs`는 LoFTR cross-attention이 **같은 행(row) 안에서만** 어텐션을 걸기 때문에,
두 영상의 대응점이 같은 y좌표에 있어야 합니다. 이를 위해 `gps_gs` 브랜치는
rectification을 필요로 합니다. DAv3 브랜치는 LoFTR 모듈을 쓰지 않으므로
(`raft.use_loftr_coarse: False`) rectification도 필요하지 않습니다.

---

## 1. 원본 데이터 레이아웃

```
{RAW_DATA_PATH}/
└── <seq>/                          # s1a1, s1a2, s1a3, s2a1 … s3a5
    ├── <frame>_<camserial>.jpg     # 예: 0000_22139908.jpg
    └── calibration_full.json       # 카메라별 K / distCoeff / R / T / imgSize (OpenCV 규약)
```


## 2. 실행

한 줄로 전체 세트를 만듭니다.

```bash
cd data_process

# DAv3 세 브랜치용 (비정렬)
./run_data_process.sh --no-rect \
    {RAW_DATA_PATH} \
    {PREPROCESSED_WO_RECT_PATH}

# gps_gs용 (rectified)
./run_data_process.sh --rect \
    {RAW_DATA_PATH} \
    {PREPROCESSED_PATH}
```

시퀀스 하나씩 돌리려면:

```bash
python step_0.py -i s1a1 -t train --no-rect \
    --data-root {RAW_DATA_PATH} \
    --processed-root {PREPROCESSED_WO_RECT_PATH}
python step_1.py -i s1a1 -t train \
    --data-root {RAW_DATA_PATH} \
    --processed-root {PREPROCESSED_WO_RECT_PATH}
```

- `step_0.py` — source view 0, 1 (스테레오 쌍) + mask + `0_1.json`. `--rect` 여부가 여기서만 갈립니다.
- `step_1.py` — novel view 2, 3, 4, 5 + `{id}_intrinsic.npy` / `{id}_extrinsic.npy`.
  novel view는 **어느 경우에도 rectify하지 않습니다.**
- **두 스크립트에 반드시 같은 `--processed-root`를 주어야 합니다.** 다르게 주면 source
  view와 novel view가 서로 다른 데이터셋으로 갈라집니다.
- `-j` 로 스레드 수 조절 (기본 `min(32, cpu_count)`). `run_data_process.sh` 는 `JOBS` 환경변수.

프레임 분할은 스크립트에 고정되어 있습니다.

| split | 시퀀스 | 프레임 |
|---|---|---|
| `train` | `s1a1 s1a2 s1a3 s2a1 s2a2 s2a3 s3a1 s3a2 s3a3` | 앞 300장 |
| `val` | `s1a6 s2a4 s3a5` | `[50:80]` (30장) |

## 3. 출력 레이아웃

```
{PREPROCESSED_PATH}/{train,val}/
├── img/<seq>_s<N>_<frame>/{0,1,2,3,4,5}.png      1024×1024
├── mask/<seq>_s<N>_<frame>/{0,1}.png             전부 255인 더미
└── parameter/<seq>_s<N>_<frame>/
    ├── 0_1.json                                  intr0/intr1/extr0/extr1/Tf_x
    └── {2,3,4,5}_{intrinsic,extrinsic}.npy
```


## 4. 카메라 → view index 매핑

카메라 10대가 거의 일정한 간격으로 한 줄로 늘어서 있습니다. source 카메라(●)는 3대씩
건너뛰어 놓이고, 그 사이를 novel 카메라(○) 두 대가 채웁니다. segment 는 인접한 source
카메라 쌍으로 끊기므로 **양 옆 segment 와 카메라 한 대를 공유**합니다.

```
   22139908  22139907  22070932  22139909  22053927  22053908  22139914  22053925  22053923  22139906
   ●─────────○─────────○─────────●─────────○─────────○─────────●─────────○─────────○─────────●
   └ s1 ─────────────────────────┘
   view0     view2     view3     view1
                                 └ s2 ─────────────────────────┘
                                 view0     view2     view3     view1
                                                               └ s3 ─────────────────────────┘
                                                               view0     view2     view3     view1
```

`●` source view (0, 1) · `○` novel view (2, 3) · 폴더 이름은 `<seq>_s<N>_<frame>`

두 종류의 view는 역할이 다릅니다.

- **source view** — 모델에 **입력으로 들어가는** view. 이 view들로부터 depth를 추정해
  Gaussian을 만듭니다.
- **novel view** — 입력으로는 쓰지 않습니다. 만들어진 Gaussian을 그 시점으로 렌더링해
  GT 이미지와 비교하는 **supervision 대상**입니다.

view 4 / 5 는 view 0 / 1 과 같은 카메라입니다. 같은 카메라를 novel view 번호로 한 번 더
기록해 둔 것입니다.
