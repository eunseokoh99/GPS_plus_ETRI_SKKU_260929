from yacs.config import CfgNode as CN


class ConfigStereoHuman:
    def __init__(self):
        self.cfg = CN()
        self.cfg.name = ''
        self.cfg.stage1_ckpt = None
        self.cfg.restore_ckpt = None
        self.cfg.lr = 0.0
        self.cfg.wdecay = 0.0
        self.cfg.batch_size = 0
        self.cfg.num_steps = 0

        # Which model class train.py/test.py instantiate (see train.MODELS).
        self.cfg.model_type = 'DAV3Model_MK'
        # No chamfer term on the DAv3 branches.
        self.cfg.use_chamfer = False

        self.cfg.dataset = CN()
        self.cfg.dataset.source_id = None
        self.cfg.dataset.train_novel_id = None
        self.cfg.dataset.val_novel_id = None
        self.cfg.dataset.use_hr_img = None
        self.cfg.dataset.use_depth_init = None
        self.cfg.dataset.use_local_data = None
        self.cfg.dataset.local_data_root = ''
        self.cfg.dataset.train_data_root = ''
        self.cfg.dataset.val_data_root = ''
        # gsussian render settings
        self.cfg.dataset.bg_color = [0, 0, 0]
        self.cfg.dataset.zfar = 100.0
        self.cfg.dataset.znear = 0.01
        self.cfg.dataset.trans = [0.0, 0.0, 0.0]
        self.cfg.dataset.scale = 1.0
        self.cfg.dataset.inverse_depth_init = 1.0
        # Top/bottom crop fraction applied to images before validation PSNR
        # (0.0 = no crop). Eval scores every view in ``val_novel_id``.
        self.cfg.dataset.eval_img_hcrop = 0.0

        self.cfg.raft = CN()
        self.cfg.raft.mixed_precision = None
        self.cfg.raft.train_iters = 0
        self.cfg.raft.val_iters = 0
        self.cfg.raft.corr_implementation = 'reg' # 'reg_cuda'
        self.cfg.raft.corr_levels = 4
        self.cfg.raft.corr_radius = 4
        self.cfg.raft.n_downsample = 3  # feature down sample rate, 3 means x8 down sample
        self.cfg.raft.context_norm = 'group'
        self.cfg.raft.n_gru_layers = 1  # down sample based on x8 down sample, 3 means x8 x16 x32 in gru, set 1 for fast
        self.cfg.raft.slow_fast_gru = None  # not valid if n_gru_layers==1
        self.cfg.raft.encoder_dims = [64, 96, 128]
        self.cfg.raft.hidden_dims = [128]*3
        # Bypass the cross-view coarse transformer: the DA3 backbone already
        # relates the views, and LoFTR's row attention assumes rectified input.
        # MUST stay False -- flipping it adds a loftr_coarse submodule to the
        # model, which shifts the state_dict and breaks existing checkpoints.
        self.cfg.raft.use_loftr_coarse = False

        self.cfg.gsnet = CN()
        self.cfg.gsnet.use_pe = None
        self.cfg.gsnet.use_depth_net = None
        self.cfg.gsnet.use_warped_depth = None
        self.cfg.gsnet.encoder_dims = None
        self.cfg.gsnet.decoder_dims = None
        self.cfg.gsnet.parm_head_dim = None

        # --- Depth-Anything-3 depth branch ---
        self.cfg.dav3 = CN()
        self.cfg.dav3.load_pretrained = True
        self.cfg.dav3.freeze_backbone = True
        self.cfg.dav3.process_res = 504
        self.cfg.dav3.process_res_method = 'upper_bound_resize'
        self.cfg.dav3.adapter_hidden_dim = 64
        self.cfg.dav3.min_inverse_depth = 1e-4
        self.cfg.dav3.max_inverse_depth = 20.0
        self.cfg.dav3.apply_gs_resdepth = True
        # The depth head is the backbone-direct learned upsampler (UpsamplerV2):
        # it bypasses DA3's DPT head and upsamples the backbone tokens straight
        # to full-res depth.
        self.cfg.dav3.upsampler_log_depth_range = 3.0
        # LayerNorm the first half of each backbone token block before the head.
        self.cfg.dav3.add_layernorm_upsampler = True
        # Train DA3 with LoRA: the backbone is frozen except the injected
        # adapters. One of: 'none', 'lora', 'bitfit', 'layernorm'.
        self.cfg.dav3.tuning_mode = 'lora'
        self.cfg.dav3.lora = CN()
        self.cfg.dav3.lora.rank = 8
        self.cfg.dav3.lora.alpha = 16.0
        self.cfg.dav3.lora.dropout = 0.0
        self.cfg.dav3.lora.target_modules = ['qkv', 'proj']

        self.cfg.external_models = CN()
        self.cfg.external_models.depth_anything_v3 = CN()
        self.cfg.external_models.depth_anything_v3.device = None
        self.cfg.external_models.depth_anything_v3.weights_path = None
        self.cfg.external_models.depth_anything_v3.hf_repo_id = None
        self.cfg.external_models.depth_anything_v3.config_key = None
        self.cfg.external_models.depth_anything_v3.load_pretrained = True
        self.cfg.external_models.depth_anything_v3.process_res = 504
        self.cfg.external_models.depth_anything_v3.process_res_method = 'upper_bound_resize'

        self.cfg.record = CN()
        self.cfg.record.ckpt_path = None
        self.cfg.record.show_path = None
        self.cfg.record.logs_path = None
        self.cfg.record.file_path = None
        self.cfg.record.save_freq = 10000
        self.cfg.record.loss_freq = 0
        self.cfg.record.eval_freq = 0
        # Keep EMA (0.999 / 0.9999) model versions alongside the raw weights.
        self.cfg.record.use_ema = False

        # Weights & Biases. project='force_none' disables W&B; set a real
        # project (and name) in stage.yaml to enable it.
        self.cfg.wandb = CN()
        self.cfg.wandb.project = 'force_none'
        self.cfg.wandb.name = None
        self.cfg.wandb.resume = None
        self.cfg.wandb.id = None

    def get_cfg(self):
        return self.cfg.clone()

    def load(self, config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
        self.cfg.freeze()


if __name__ == '__main__':
    cc = ConfigStereoHuman()
    cc.load("./stage.yaml")
    print(cc.cfg)
