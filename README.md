# GPS_plus_ETRI_SKKU_260929

희소 시점 입력으로부터 사람을 실시간 렌더링하는 feed-forward 3D Gaussian Splatting
모델입니다. 네 가지 브랜치를 학습 / 추론할 수 있습니다.

| 브랜치 | 입력 view | depth 추정 | 요약 |
|---|---|---|---|
| `gps_gs` | 2 (rectified) | RAFT-Stereo + LoFTR | 공식 GPS-Gaussian+ 베이스라인 |
| `dav3_2view` | 2 | DA3-Small (frozen + LoRA) with upsamplerV2 | RAFT-Stereo를 Depth-Anything-3로 교체 |
| `dav3_4view` | 4 | DA3-Small (frozen + LoRA) with upsamplerV2 | 카메라 4대를 모두 입력, 렌더는 가장 가까운 2대로 |
| `dav3_4view_with_multiview_supervision` | 4 | DA3-Small (frozen + LoRA) with upsamplerV2 | 위와 같고, 학습 시 3개 구간을 동시에 감독 |

브랜치별로 `config/<branch>/` 안의
`stage.yaml`과 `stereo_human_config.py`에 실험 설정 및 설정 키 기본값이 셋팅되어 있습니다.

---

## 설치

Python 3.10, CUDA 12.1 기준입니다.
torch는 PyPI에 `+cu121` 빌드가 없으므로 전용 인덱스에서 먼저 받습니다.

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r requirements.txt
```

추가로 세 가지가 필요합니다. 그중 두 개는 CUDA 확장을 직접 빌드해야 하므로, 먼저 빌드
환경 변수를 잡아 둡니다.

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST=8.6   # 본인 GPU의 compute capability
export MAX_JOBS=16                # 병렬 컴파일 (메모리가 부족하면 줄이세요)
```

`TORCH_CUDA_ARCH_LIST` 를 사용자의 GPU 값으로 바꿉니다 (`8.6` = RTX A6000 / A40 /
RTX 30 계열, `8.0` = A100, `8.9` = RTX 40 계열, `9.0` = H100). 확인은
`python -c "import torch; print(torch.cuda.get_device_capability(0))"`으로 합니다.

```bash
# 1) 3DGS 래스터라이저 (네 브랜치 모두 필수)
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization \
    --no-build-isolation

# 2) Depth-Anything-3 (dav3_* 브랜치용). 가중치는 첫 실행 시 자동 다운로드됩니다.
mkdir -p external_repos && cd external_repos
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git && cd ..

# 3) pytorch3d v0.7.9 (gps_gs 의 chamfer loss용). source 빌드가 필요합니다.
git clone --branch v0.7.9 https://github.com/facebookresearch/pytorch3d.git
pip install -e pytorch3d --no-build-isolation
```

## W&B

학습 스크립트는 기본적으로 Weights & Biases에 로그를 올립니다
(`stage.yaml` 의 `wandb.project: ETRI_GPS_plus`). 로그인하지 않은 상태로 학습을 시작하면
`wandb.init` 단계에서 멈추거나 실패하므로, **둘 중 하나는 해야 합니다.**

W&B를 쓸 경우 미리 로그인합니다.

```bash
wandb login
```

쓰지 않을 경우 `project` 를 `force_none` 으로 지정해 끕니다.

```bash
./scripts/train_dav3_4view.sh --opts wandb.project force_none
```

## 데이터 준비

브랜치에 따라 두 가지 데이터셋이 필요합니다. `gps_gs` 만 rectification이 필요합니다.

```bash
cd data_process

# dav3_* 세 브랜치용
./run_data_process.sh --no-rect {RAW_DATA_PATH} {PREPROCESSED_WO_RECT_PATH}

# gps_gs 용
./run_data_process.sh --rect    {RAW_DATA_PATH} {PREPROCESSED_PATH}
```

`{RAW_DATA_PATH}` 는 THumanMV 원본(`<seq>/<frame>_<camserial>.jpg` +
`calibration_full.json`)이 있는 디렉토리, 나머지 둘은 생성할 데이터셋을 둘 위치입니다.
자세한 내용과 카메라 매핑은 [data_process/data_process.md](data_process/data_process.md)에 있습니다.

### 경로 설정

실행 전에 아래 자리표시자를 본인 경로에 맞게 변경해야 합니다.

| 자리표시자 | 의미 | 쓰이는 곳 |
|---|---|---|
| `{RAW_DATA_PATH}` | THumanMV 원본 | `data_process/` 실행 인자 |
| `{PREPROCESSED_PATH}` | `--rect` 로 만든 데이터셋 | `config/gps_gs/stage.yaml` |
| `{PREPROCESSED_WO_RECT_PATH}` | `--no-rect` 로 만든 데이터셋 | `config/dav3_*/stage.yaml` |

config 쪽은 브랜치마다 `dataset.local_data_root` / `train_data_root` / `val_data_root`
세 줄을 변경하면 됩니다. 예를 들어 `dav3_4view` 는 다음과 같이 채웁니다.

```yaml
  local_data_root: '/your/path/preprocessed_wo_rect'
  train_data_root: '/your/path/preprocessed_wo_rect/train'
  val_data_root:   '/your/path/preprocessed_wo_rect/val'
```

`stage.yaml` 을 건드리지 않고 실행할 때만 덮어쓰고 싶다면, `--opts` 를 사용합니다.

```bash
./scripts/train_dav3_4view.sh --opts \
    dataset.local_data_root /your/path/preprocessed_wo_rect \
    dataset.train_data_root /your/path/preprocessed_wo_rect/train \
    dataset.val_data_root   /your/path/preprocessed_wo_rect/val
```

## 학습

```bash
./scripts/train_gps_gs.sh
./scripts/train_dav3_2view.sh
./scripts/train_dav3_4view.sh
./scripts/train_dav3_4view_with_multiview_supervision.sh
```

### 결과물

결과는 `experiments/<name>_<MMDD>/` 아래에 저장됩니다.

```
experiments/dav3_4view_0929/
├── ckpt/   iter<N>.pth, iter<N>_ema1_e3.pth, iter<N>_ema1_e4.pth
│           dav3_4view_latest.pth       (loss_freq마다 덮어쓰기)
│           dav3_4view_final.pth        (학습 종료 시)
├── show/   학습 중 저장되는 렌더 샘플
├── logs/
└── file/   이 run을 띄운 config 사본 + cfg.json
```

### 주요 config

`config/<branch>/stage.yaml` 을 수정하거나, 학습 스크립트에 `--opts <키> <값>` 으로 덮어쓸 수 있습니다.

<details>
<summary><b>학습</b></summary>

| 키 | 기본값 | 설명 |
|---|---|---|
| `num_steps` | 100000 | 학습 step 수 |
| `lr` | 2e-4 | OneCycleLR의 최대 learning rate |
| `batch_size` | 1 | |
| `restore_ckpt` | `None` | 중단된 학습 이어하기 (optimizer / step 까지 복원) |
| `use_chamfer` | `gps_gs` 만 `True` | 두 source view 점군 간 chamfer loss |
| `record.save_freq` | 5000 | `iter<N>.pth` 저장 주기 |
| `record.eval_freq` | 1000 | val PSNR 측정 주기 |
| `record.use_ema` | `True` | EMA 가중치(`ema1_e3`, `ema1_e4`)도 함께 저장 |
| `wandb.project` | `ETRI_GPS_plus` | `force_none` 이면 W&B를 끕니다 |

</details>

<details>
<summary><b>데이터 / view 구성</b></summary>

| 키 | 기본값 | 설명 |
|---|---|---|
| `dataset.num_source_views` | 2 (4view는 4) | 입력 view 개수. 2보다 크면 멀티뷰 로더를 씁니다 |
| `dataset.mv_source_chain` | `[[1,0],[1,1],[2,1],[3,1]]` | 4-view 입력일 때 각 입력 view를 `[segment 인덱스, 쌍 내 인덱스]` 로 지정합니다. 기본값은 순서대로 s1의 카메라 0 / s1의 카메라 1 / s2의 카메라 1 / s3의 카메라 1 이고, 이것이 `view0`~`view3` 이 됩니다 |
| `dataset.render_nearest_k` | 0 (4view는 2) | novel view 한 장을 렌더할 때 입력 view 중 **가까운 것부터 몇 개**의 Gaussian을 병합할지. `0` 이면 입력 view 전부를 병합합니다 |
| `dataset.novel_per_segment` | `False` (mvs는 `True`) | 학습에만 적용됩니다. `True` 면 매 step마다 segment 3개(s1/s2/s3)에서 novel view를 1장씩 뽑아 3장을 감독하고 loss를 평균합니다. `False` 면 그 샘플이 속한 segment에서 1장만 감독합니다 |
| `dataset.train_novel_id` | `[2, 3, 4, 5]` | 학습 시 감독할 novel view 후보 |
| `dataset.val_novel_id` | `[3]` | 평가에 쓰는 novel view |
| `dataset.eval_img_hcrop` | 0.1 | PSNR 계산 전에 상하단에서 잘라내는 비율 |
| `dataset.sample_glob` | 없음 | 학습 샘플을 glob으로 제한 (예: `'s1a*_s*_*'`). val은 항상 전체 |

</details>

<details>
<summary><b>DAv3 백본</b> (<code>dav3_*</code> 브랜치)</summary>

| 키 | 기본값 | 설명 |
|---|---|---|
| `dav3.freeze_backbone` | `True` | DA3 backbone 동결 |
| `dav3.tuning_mode` | `'lora'` | `none` / `lora` / `bitfit` / `layernorm` |
| `dav3.lora.rank`, `alpha` | 8, 16.0 | LoRA 용량 |
| `dav3.process_res` | 504 | DA3에 넣기 전 리사이즈 해상도 |

</details>

## 추론 / 렌더링

학습된 모델로 추론 및 렌더링 하기 위해, 학습된 checkpoint의 경로를 CKPT 인자로 주어 아래와 같이 실행시킵니다.

```bash
CKPT=experiments/<name>_<MMDD>/ckpt/iter<N>_ema1_e3.pth \
  ./scripts/test_dav3_4view.sh
```

### 결과물

val set 전체를 렌더해 `experiments/<name>_<MMDD>/test_show_val/` 에 PNG로 저장하고, 끝나면
아래와 같이 PSNR과 forward 속도를 출력합니다.

```
  config        dav3_4view
  checkpoint    experiments/dav3_4view_0929/ckpt/iter95000_ema1_e3.pth
  samples       270 x 1 novel view = 270 scored
  PSNR (full)   33.7794
  PSNR (crop)   33.7829   (eval_img_hcrop=0.1)
  forward       249.1 ms   (median 249.2, min 247.9, std 0.75)
  peak memory   4122 MB
                1 forward(s) covering all 4 source cameras of one frame,
                100 iterations after 50 warm-up, inputs already on the GPU;
                model forward only -- data IO and render excluded.
                Needs an otherwise idle GPU to be comparable.
```

### 추론 시 주요 config

`stage.yaml` 을 고치거나 `--opts <키> <값>` 으로 덮어쓸 수 있습니다. 스크립트에 붙인 인자는
`test.py` 로 그대로 전달됩니다.

<details>
<summary><b>추론 / 평가</b></summary>

| 키 | 기본값 | 설명 |
|---|---|---|
| `dataset.val_novel_id` | `[3]` | 렌더하고 PSNR을 계산할 novel view |
| `dataset.eval_img_hcrop` | 0.1 | PSNR 계산 전에 상하단에서 잘라내는 비율 |
| `dataset.test_save_hcrop` | 0.0 | PNG로 **저장**할 때만 잘라내는 비율 |
| `dataset.render_nearest_k` | 0 (4view는 2) | novel view 한 장을 렌더할 때 가까운 것부터 몇 개의 Gaussian을 병합할지 |
| `dataset.*_data_root` | 자리표시자 | 평가에 쓸 데이터셋 경로 |
| `batch_size` | 1 | |

</details>

<details>
<summary><b>학습 때와 반드시 같아야 하는 값</b> (<code>dav3_*</code> 브랜치)</summary>

아래 두 값은 upsampler head의 forward 식에 상수로 들어가는데 체크포인트에 저장되지 않아,
학습 때와 다른 값을 주면 가중치가 같아도 다른 depth가 나옵니다.

| 키 | 기본값 |
|---|---|
| `dataset.inverse_depth_init` | 0.2 (`gps_gs` 는 0.3) |
| `dav3.upsampler_log_depth_range` | 3.0 |

</details>

## 성능 / 속도

아래는 전체 val set 270 샘플, novel view(`val_novel_id: [3]`)로 평가 했을 때 PSNR 성능 및 속도 입니다.

### PSNR

| 브랜치 | 체크포인트 | PSNR |
|---|---|---|
| `gps_gs` | iter75000, ema1_e3 | 33.18 |
| `dav3_2view` | iter95000, ema1_e3 | 33.55 |
| `dav3_4view` | iter95000, ema1_e3 | 33.78 |
| `dav3_4view_with_multiview_supervision` | iter95000, ema1_e3 | **34.24** |

상하단 10%를 잘라낸 crop PSNR입니다 (`eval_img_hcrop: 0.1`). `gps_gs` 는 rectification
때문에 이미지 상하단에 검은 테두리가 생겨 full-frame PSNR(22.09)이 크게 낮게 나오므로,
네 브랜치를 같은 기준으로 비교하려면 crop 값을 봐야 합니다.

### 속도

| 브랜치 | forward 횟수 | 속도 | peak memory |
|---|---|---|---|
| `gps_gs` | 2-view × 3회 | 394.1 ms (2.5 fps) | 2.2 GB |
| `dav3_2view` | 2-view × 3회 | 448.4 ms (2.2 fps) | 2.4 GB |
| `dav3_4view` | 4-view × 1회 | 249.1 ms (4.0 fps) | 4.0 GB |
| `dav3_4view_with_multiview_supervision` | 4-view × 1회 | **249.8 ms (4.0 fps)** | 4.0 GB |

위 값은 `scripts/test_<branch>.sh` 를 돌리면 그대로 출력됩니다.

**측정 구간.** 입력 이미지를 넣어 Gaussian 파라미터(xyz / rot / scale / opacity)가
나오기까지의 모델 forward만 측정했습니다. 즉 아래 두 구간은 **제외**되어 있습니다.

- data IO: 디스크 읽기와 CPU→GPU 전송입니다.
- render: Gaussian을 이미지로 굽는 rasterization입니다. `pts2render` 를 호출하지 않습니다.

**forward 횟수.** 표의 시간은 카메라 4대를 입력했을 때 세 쌍(s1 / s2 / s3)의 Gaussian
파라미터가 모두 나오기까지 걸리는 시간입니다. 4-view 브랜치는 네 view를 한 번에 처리하므로
forward 1회, 2-view 브랜치는 두 view(카메라 쌍)를 한 번씩 처리하므로 forward 3회를 합한 값입니다.

### 측정 환경

- GPU: NVIDIA RTX A6000 (48 GB) 1장
- 입력 이미지: 1024×1024
- 속도 측정 시 반복 횟수: 같은 입력으로 forward 150번 중 앞 50번은 warm-up으로 버리고 뒤 100번의 평균
  (GPU 클럭이 올라가고 CUDA 커널이 선택·캐시되기까지 첫 몇 번이 느립니다)
- 속도 측정 시 peak memory: 그 100번 동안 PyTorch가 할당한 GPU memory 최대량

---

코드 구조와 브랜치별 동작 방식 다이어그램은 [AGENTS.md](AGENTS.md)를 보세요.
