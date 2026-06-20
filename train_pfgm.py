"""
Train an unconditional PFGM (Poisson Flow Generative Model) on Mayo CT slices.

This is the unconditional training entry-point for the FORCE algorithm.
See readme.md / train.sh for usage.

Example (single GPU):
    python train_pfgm.py \
        --data_dir ./data/mayo_npy \
        --image_size 512 \
        --in_channels 1 --out_channels 1 \
        --channel_mult 1,2,4,8,16 --num_res_blocks 1 \
        --batch_size 2 --use_fp16 True \
        --lr 1e-4 --lr_anneal_steps 500000 \
        --max_norm 1.0

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=NUM_GPUS train_pfgm.py ...
"""
import os
import datetime
import argparse

from utils import dist_util, logger
from torch.utils.data import DataLoader
from dataset.mayo import MayoDataset, augment_defaults
from diffusion.resample import create_named_schedule_sampler
from utils.script_util import (
    args_to_dict,
    add_dict_to_argparser,
    data_wraper,
)
from utils.pfgm_util import (
    model_and_sde_defaults,
    create_model_and_sde,
)
from utils.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(
        dir=os.path.join(
            "./logs", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        )
    )

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_sde(
        **args_to_dict(args, model_and_sde_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion
    )

    logger.log("creating data loader...")
    aug_kwargs = args_to_dict(args, augment_defaults().keys())
    dataset = MayoDataset(args.data_dir, split="train", **aug_kwargs)
    logger.log(f"  dataset: {len(dataset)} slices from {args.data_dir}")
    data = data_wraper(
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.ncpus,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        checkpointdir=args.checkpointdir,
        use_fp16=args.use_fp16,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        max_norm=args.max_norm,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="./data/mayo_npy",
        ncpus=4,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=500000,
        max_norm=1.0,
        batch_size=2,
        microbatch=-1,           # -1 disables microbatches
        ema_rate=0.999,
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",
        checkpointdir="checkpoints_pfgm",
        use_fp16=True,
    )
    defaults.update(model_and_sde_defaults())
    defaults.update(augment_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
