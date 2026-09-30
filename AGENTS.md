# 코드베이스 안내서

## 1. 코드 구조

```
train.py                  Trainer 전체 + 모델/데이터셋 dispatch + __main__
test.py                   체크포인트로 val set을 렌더해 PNG로 저장
eval_psnr_wandb.py        ckpt 디렉토리를 스윕하며 iteration별 PSNR을 W&B에 기록

config/<branch>/
  stereo_human_config.py  그 브랜치의 yacs 기본값 (브랜치마다 다름)
  stage.yaml              실제 실험 설정

lib/
  human_loader.py         StereoHumanDataset (2 view) + MultiViewStereoHumanDataset (N view)
  network.py              RtStereoHumanModel — gps_gs 의 모델
  GaussianRender.py       pts2render: view별 Gaussian을 모아 novel view로 래스터화
  view_select.py          select_nearest_views — render_nearest_k 구현
  runtime.py              train.py / test.py 공용 헬퍼 (freeze_bn, device 이동, view 선택)
  paths.py                PROJECT_ROOT / EXTERNAL_REPOS_DIR (경로 가정은 여기 한 곳만)
  gs_parm_network.py      GSRegresser — depth+RGB feature → rot/scale/opacity/Δdepth
  attention_module.py     LoFTR (gps_gs 전용)
  train_recoder.py        Logger (W&B), file_backup
  loss.py  utils.py  embedder.py  gs_utils/

gaussian_renderer/        diff_gaussian_rasterization 래퍼

core/                     RAFT-Stereo (gps_gs 전용). 단 extractor.py 의 UnetExtractor 는 네 브랜치 모두 사용

models/                   DAv3 스택 (DAv3 세 브랜치 전용)
  dav3_model.py           RtStereoHumanDAV3Model + DAV3Model_MK 래퍼 (2 view)
  dav3_mv_model.py        RtStereoHumanDAV3ModelUpsamplerMV + DAV3Model_MK_Upsampler_MV (N view)
  depth_anything_v3_model.py  DA3-Small 로딩/실행 래퍼
  lora.py                 backbone에 LoRA 어댑터 주입

data_process/             원본 THumanMV → 학습용 데이터셋
scripts/                  브랜치별 train_*.sh / test_*.sh
```


## 2. 브랜치별 상세


### 2.1 `gps_gs` — 공식 GPS-Gaussian+ 베이스라인

![gps_gs 파이프라인](docs/img/gps_gs.svg)

### 2.2 `dav3_2view` — RAFT-Stereo 를 Depth-Anything-3 로 교체

![dav3_2view 파이프라인](docs/img/dav3_2view.svg)

### 2.3 `dav3_4view` — 카메라 4대 모두 입력, 가장 가까운 2대로 렌더

![dav3_4view 파이프라인](docs/img/dav3_4view.svg)

### 2.4 `dav3_4view_with_multiview_supervision` — segment 마다 novel view 감독

![dav3_4view_with_multiview_supervision 파이프라인](docs/img/dav3_4view_with_multiview_supervision.svg)
