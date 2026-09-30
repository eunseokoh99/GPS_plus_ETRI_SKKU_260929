from __future__ import print_function, division

"""Sweep saved checkpoints and log validation PSNR / crop-PSNR to W&B.

For every ``iter{N}.pth`` (tag ``bare``) or ``iter{N}_{tag}.pth`` checkpoint in
an experiment's ckpt dir this loads the weights, runs one full validation pass,
and logs

    eval/{tag}/psnr        (full-frame PSNR)
    eval/{tag}/crop_psnr   (top/bottom cropped PSNR, uses dataset.eval_img_hcrop)

to a single W&B run at ``_step = N``. That is exactly the pair
``_build_report`` expects, so once this finishes you can point
``_build_report/config_list.json`` at the new run id and rebuild the report.

The full/crop pair comes from ``train.Trainer._eval_psnr_once``, which now
returns both values from one pass.

Config is loaded exactly like ``train.py`` (a config directory holding
``stereo_human_config.py`` + ``stage.yaml``, imported by dotted path).
"""

import argparse
import importlib
import logging
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Log full/crop PSNR of checkpoints to W&B")
    p.add_argument('--config', type=str, required=True,
                   help='Config dir (same as train.py), e.g. config/exp1:only_s1s2')
    p.add_argument('--ckpt-dir', type=str, required=True,
                   help='Directory holding the iter{N}[_{tag}].pth checkpoints')
    p.add_argument('--tag', type=str, default='bare',
                   help="Checkpoint / metric namespace tag. 'bare' (default) means "
                        "the raw iter{N}.pth weights; anything else expects "
                        "iter{N}_{tag}.pth (e.g. ema1_e3)")
    p.add_argument('--min-iter', type=int, default=5000,
                   help='First iteration to evaluate, inclusive (default: 5000)')
    p.add_argument('--max-iter', type=int, default=100000,
                   help='Last iteration to evaluate, inclusive (default: 100000)')
    p.add_argument('--iter-interval', type=int, default=5000,
                   help='Iteration step between checkpoints (default: 5000)')
    p.add_argument('--wandb-project', type=str, default=None,
                   help='W&B project; defaults to cfg.wandb.project')
    p.add_argument('--wandb-name', type=str, default=None,
                   help='W&B run name; defaults to "{cfg.name}{--run-suffix}"')
    p.add_argument('--run-suffix', type=str, default='',
                   help='Suffix appended to cfg.name for the default run name '
                        '(empty by default: the eval run reuses cfg.name)')
    p.add_argument('--wandb-id', type=str, default=None,
                   help='Existing W&B run id to resume into instead of a new run')
    p.add_argument('--wandb-resume', type=str, default=None,
                   help="W&B resume mode when --wandb-id is set (e.g. 'allow', 'must')")
    return p.parse_args()


def load_cfg(config_arg):
    config_dir = Path(config_arg)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory does not exist: {config_dir}")
    module_path = config_arg.rstrip('/').replace('./', '').replace('/', '.')
    config_module = importlib.import_module('%s.stereo_human_config' % module_path)
    ConfigStereoHuman = getattr(config_module, 'ConfigStereoHuman')
    cfg_obj = ConfigStereoHuman()
    cfg_obj.load(str(config_dir / 'stage.yaml'))
    return cfg_obj.get_cfg()


def ckpt_name(iteration, tag):
    """'bare' is the raw checkpoint (no tag suffix); everything else is suffixed."""
    if tag == 'bare':
        return 'iter%d.pth' % iteration
    return 'iter%d_%s.pth' % (iteration, tag)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    )
    args = parse_args()

    cfg = load_cfg(args.config)
    tag = args.tag

    cfg.defrost()
    # We load each checkpoint explicitly in the loop -- nothing auto-loads at
    # construction time.
    cfg.restore_ckpt = None
    cfg.stage1_ckpt = None
    # No EMA shadows: we load the saved weights directly, so the bookkeeping is
    # pure overhead and cannot change the numbers.
    cfg.record.use_ema = False

    # Fresh output dir for this eval sweep (no images are dumped: save_vis=False).
    run_name = args.wandb_name or '%s%s' % (cfg.name, args.run_suffix)
    eval_root = Path('experiments') / 'eval_psnr_wandb' / run_name
    (eval_root / 'logs').mkdir(parents=True, exist_ok=True)
    cfg.record.ckpt_path = str(eval_root / 'ckpt')
    cfg.record.show_path = str(eval_root)
    cfg.record.logs_path = str(eval_root / 'logs')
    cfg.record.file_path = str(eval_root / 'file')

    # W&B target: a dedicated eval run by default (new id each launch), or resume
    # into an existing run id when requested. Logger.set_wandb names the run from
    # cfg.name, so that is what we override.
    cfg.wandb.project = args.wandb_project or cfg.wandb.project
    cfg.name = run_name
    cfg.wandb.name = run_name
    if args.wandb_id is not None:
        cfg.wandb.id = args.wandb_id
        cfg.wandb.resume = args.wandb_resume or 'allow'
    else:
        cfg.wandb.resume = None
        cfg.wandb.id = None
    cfg.exp_name = run_name
    cfg.freeze()

    if cfg.wandb.project in (None, 'force_none'):
        raise ValueError(
            "W&B is disabled (project is None/force_none). Pass --wandb-project."
        )

    import train as train_mod
    # Trainer reads a few things off train.py's module-level `cfg` global (the
    # Logger and the image dump path), which normally comes from its __main__
    # block. Point it at ours.
    train_mod.cfg = cfg

    logging.info("Building trainer ...")
    trainer = train_mod.Trainer(cfg)
    # Model state is left exactly as Trainer.__init__ leaves it (train() with
    # raft BatchNorm frozen), which is the state training-time validation runs
    # in. There is no dropout in the model, so this is deterministic.

    ckpt_dir = Path(args.ckpt_dir)
    iters = list(range(args.min_iter, args.max_iter + 1, args.iter_interval))
    logging.info(
        "Sweeping %d checkpoints (%d..%d step %d), tag=%s, from %s",
        len(iters), args.min_iter, args.max_iter, args.iter_interval, tag, ckpt_dir,
    )

    n_logged, n_missing, n_failed = 0, 0, 0
    for n in iters:
        ckpt_path = ckpt_dir / ckpt_name(n, tag)
        if not ckpt_path.exists():
            logging.warning("[iter %d] missing checkpoint %s -- skipped", n, ckpt_path)
            n_missing += 1
            continue
        try:
            trainer.load_ckpt(str(ckpt_path), load_optimizer=False, strict=True)
            with torch.no_grad():
                full_psnr, crop_psnr = trainer._eval_psnr_once(tag=tag, save_vis=False)
            trainer.total_steps = n
            trainer.logger.write_dict(
                {
                    'eval/%s/psnr' % tag: float(full_psnr),
                    'eval/%s/crop_psnr' % tag: float(crop_psnr),
                },
                write_step=n,
            )
            logging.info(
                "[iter %d] logged eval/%s/psnr=%.4f  eval/%s/crop_psnr=%.4f",
                n, tag, float(full_psnr), tag, float(crop_psnr),
            )
            n_logged += 1
        except Exception as exc:  # keep sweeping even if one checkpoint fails
            logging.exception("[iter %d] eval failed: %s", n, exc)
            n_failed += 1

    trainer.logger.close()
    logging.info(
        "Done. logged=%d missing=%d failed=%d (project=%s name=%s)",
        n_logged, n_missing, n_failed, cfg.wandb.project, run_name,
    )


if __name__ == '__main__':
    main()
