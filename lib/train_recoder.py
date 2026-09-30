
import os
import json
import shutil
import logging
from pathlib import Path

from yacs.config import CfgNode as CN
import socket


def file_backup(exp_path, cfg, config_dir):
    """Record what this run was launched with: the resolved config as
    ``cfg.json`` plus a verbatim copy of the config directory that produced it.

    Only the config is copied -- the source tree itself is tracked in git, and
    copying it per run duplicated ~10k lines into every experiment folder.
    """
    exp_path = Path(exp_path)
    config_dir = Path(config_dir)
    shutil.copytree(config_dir, exp_path / config_dir.name, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))

    with open(exp_path / 'cfg.json', 'w') as json_file:
        json.dump(_cfgnode_to_dict(cfg), json_file, indent=1)


def _cfgnode_to_dict(x):
    """Recursively convert yacs.CfgNode (or nested types) to vanilla dict/list/values."""
    if isinstance(x, CN):
        return {k: _cfgnode_to_dict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_cfgnode_to_dict(v) for v in x]
    if isinstance(x, dict):
        return {k: _cfgnode_to_dict(v) for k, v in x.items()}
    return x


class Logger:
    """
    W&B logger.
    - Call set_wandb(wandb_cfg, full_cfg) once if you want W&B enabled.
    - Use push()/write_dict() for dict metrics.
    - Use add_scalar() for single scalars.
    """
    def __init__(self, scheduler, cfg):
        self.scheduler = scheduler
        self.sum_freq = cfg.loss_freq
        self.log_dir = cfg.logs_path
        self.total_steps = 0
        self.running_loss = {}

        # W&B related
        self._wandb = None          # module handle (wandb)
        self._wandb_run = None      # active run
        self._wandb_enabled = False
        self._cfg = cfg             # keep cfg so set_wandb can read cfg.wandb

    # ---------- W&B setup ----------
    def set_wandb(self, wandb_cfg, full_cfg=None):
        """
        Initialize W&B from cfg.wandb and push the entire training cfg to wandb.config.
        Required:
            - project (use 'force_none' to disable W&B)
            - name (or run_name)
        Optional:
            - resume (True/'allow'/'must') and id (only if resume is truthy)
        """
        wb_cfg = wandb_cfg
        if wb_cfg is None:
            raise AssertionError("Please set wandb config in cfg.wandb")

        project = wb_cfg.get('project')
        if project == "force_none":
            self._wandb_enabled = False
            return

        if project is None:
            raise AssertionError("cfg.wandb.project must be set")

        name = full_cfg.name  # get original name, not wandb name.

        init_kwargs = {
            'project': project,
            'name': name,
            'dir': str(Path(self.log_dir).resolve().parent),
        }

        resume = wb_cfg.get('resume', None)
        if resume:
            init_kwargs['resume'] = resume
            if 'id' in wb_cfg:
                init_kwargs['id'] = wb_cfg['id']

        import wandb
        self._wandb = wandb
        self._wandb_run = wandb.init(**init_kwargs)
        self._wandb_enabled = True

        logging.info(
            "Initialized Weights & Biases run%s in project '%s'%s",
            f" '{name}'" if name else "",
            project,
            f" (resumed id={wb_cfg.get('id')})" if resume and 'id' in wb_cfg else ""
        )

        # ---- Push server name into W&B config
        # Set $SERVER_NAME to label runs by machine; otherwise fall back to the
        # outbound IP of this host.
        server_name = os.environ.get("SERVER_NAME")
        if not server_name:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            server_name = s.getsockname()[0]
            s.close()

        wandb.config.update({"server": server_name}, allow_val_change=True)

        # ---- Push the full training cfg into W&B config under "cfg"
        if full_cfg is not None:
            try:
                cfg_dict = _cfgnode_to_dict(full_cfg)
                if isinstance(cfg_dict, dict) and "wandb" in cfg_dict:
                    cfg_dict = dict(cfg_dict)  # shallow copy
                    cfg_dict.pop("wandb", None)
                wandb.config.update({"cfg": cfg_dict}, allow_val_change=True)
            except Exception as e:
                logging.warning(f"Failed to push full cfg to W&B config: {e}")

    def _wb_log(self, data_dict, step=None):
        if self._wandb_enabled and self._wandb_run is not None and isinstance(data_dict, dict) and data_dict:
            if step is None:
                step = self.total_steps
            self._wandb_run.log(dict(data_dict), step=step)

    # ---------- training status ----------
    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / self.sum_freq for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        step = self.total_steps
        to_log = {}
        for k in self.running_loss:
            value = self.running_loss[k] / self.sum_freq
            to_log[k] = value
            self.running_loss[k] = 0.0

        self._wb_log(to_log, step=step)

    # ---------- public logging APIs ----------
    def add_scalar(self, key, value, step):
        """Log a single scalar to W&B."""
        self._wb_log({key: value}, step=step)

    def push(self, metrics):
        """
        Accumulate per-step metrics dict (e.g., {'l1': ..., 'ssim': ...}).
        Every sum_freq steps, write averaged values to W&B.
        """
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]

        next_step = self.total_steps + 1
        if next_step % self.sum_freq == 0:
            self.total_steps = next_step
            self._print_training_status()
            self.running_loss = {}
        else:
            self.total_steps = next_step

    def write_dict(self, results, write_step):
        """
        Immediately write a dict of metrics (no averaging), e.g. validation metrics.
        """
        self._wb_log(results, step=write_step)

    def log_image(self, key, image, step):
        if self._wandb_enabled and self._wandb_run is not None:
            self._wandb_run.log({key: self._wandb.Image(image)}, step=step)

    def close(self):
        if self._wandb_enabled and self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception as e:
                logging.warning(f"W&B finish raised: {e}")
