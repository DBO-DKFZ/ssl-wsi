import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import jsonargparse
import lightning as L
import torch
import torchvision
from data import PretrainDataModule
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch import nn
from transforms import DINOTransform, get_transform
from utils import cosine_scheduler
from vit import get_vit

cv2.setNumThreads(1)  # Just to be sure that opencv does not use all threads
torch.set_float32_matmul_precision("high")


@dataclass
class DataOptions:
    root: str
    frame: str
    limit_slides: int = 0
    level: int = 0
    verbose: bool = False
    epoch_length: Optional[int] = None
    tile_size_multiple: int = 1
    steps_per_epoch: Optional[int] = None
    seed: int = 42
    image_size: int = 224
    natural_ratio: float = 0.0
    natural_root: Optional[str] = None
    natural_replace: bool = False
    batch_size_per_device: int = 1
    num_workers: int = 8
    simplify_tolerance: int = 100
    centroid_in_annotation: bool = True
    allow_out_of_bounds: bool = False
    simple_epoch: bool = True
    without_wsi: bool = False
    backend: str = "cucim"


class DINO(L.LightningModule):
    def __init__(
        self,
        arch: str = "tiny",
        image_size: int = 224,
        patch_size: int = 16,
        drop_path_rate: float = 0.0,
        pos_emb: str = "learned",
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        batch_norm: bool = False,
        freeze_last_layer: int = -1,
        norm_last_layer: bool = True,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        lr_start: float = 0.0005,
        lr_final: float = 1e-6,
        lr_warmup_epochs: int = 10,
        wd_start: float = 0.04,
        wd_final: float = 0.4,
        mm_start: float = 0.996,
        mm_final: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.student_backbone = get_vit(
            arch=self.hparams.arch,
            image_size=self.hparams.image_size,
            patch_size=self.hparams.patch_size,
            drop_path_rate=self.hparams.drop_path_rate,
            pos_emb=self.hparams.pos_emb,
        )
        self.student_head = DINOProjectionHead(
            input_dim=self.student_backbone.num_features,
            hidden_dim=self.hparams.hidden_dim,
            bottleneck_dim=self.hparams.bottleneck_dim,
            output_dim=self.hparams.output_dim,
            batch_norm=self.hparams.batch_norm,
            freeze_last_layer=self.hparams.freeze_last_layer,
            norm_last_layer=self.hparams.norm_last_layer,
        )
        self.teacher_backbone = get_vit(
            arch=self.hparams.arch,
            image_size=self.hparams.image_size,
            patch_size=self.hparams.patch_size,
            drop_path_rate=0.0,
            pos_emb=self.hparams.pos_emb,
        )
        self.teacher_head = DINOProjectionHead(
            input_dim=self.student_backbone.num_features,
            hidden_dim=self.hparams.hidden_dim,
            bottleneck_dim=self.hparams.bottleneck_dim,
            output_dim=self.hparams.output_dim,
            batch_norm=self.hparams.batch_norm,
            norm_last_layer=self.hparams.norm_last_layer,
        )

        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(
            output_dim=self.hparams.output_dim,
            warmup_teacher_temp=self.hparams.warmup_teacher_temp,
            teacher_temp=self.hparams.teacher_temp,
            warmup_teacher_temp_epochs=self.hparams.warmup_teacher_temp_epochs,
            student_temp=self.hparams.student_temp,
            center_momentum=self.hparams.center_momentum,
        )

        self.lr_sch = None
        self.wd_sch = None
        self.mm_sch = None

    def forward(self, x):
        y = self.student_backbone(x)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x)
        z = self.teacher_head(y)
        return z

    def on_train_start(self):
        assert self.trainer.estimated_stepping_batches is not None
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = math.ceil(total_steps / self.trainer.max_epochs)

        self.lr_sch = cosine_scheduler(
            start_value=self.hparams.lr_start,
            final_value=self.hparams.lr_final,
            warmup_steps=self.hparams.lr_warmup_epochs * steps_per_epoch,
            total_steps=total_steps,
        )
        self.wd_sch = cosine_scheduler(
            start_value=self.hparams.wd_start,
            final_value=self.hparams.wd_final,
            total_steps=total_steps,
        )
        self.mm_sch = cosine_scheduler(
            start_value=self.hparams.mm_start,
            final_value=self.hparams.mm_final,
            total_steps=total_steps,
        )

    def update_rates(self):
        lr = self.lr_sch[self.global_step]
        wd = self.wd_sch[self.global_step]
        mm = self.mm_sch[self.global_step]

        update_momentum(self.student_backbone, self.teacher_backbone, m=mm)
        update_momentum(self.student_head, self.teacher_head, m=mm)
        # update learning rate, weight decay
        for i, param_group in enumerate(self.optimizers().param_groups):
            param_group["lr"] = lr
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd

        return lr, wd, mm

    def training_step(self, batch, batch_idx):
        lr, wd, mm = self.update_rates()

        views = [view.to(self.device) for view in batch]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        self.log("learning_rate", lr)
        self.log("weight_decay", wd)
        self.log("momentum", mm)
        self.log("teacher_temp", self.criterion.teacher_temp)
        self.log("loss", loss, True)

        if not torch.isfinite(loss):
            self.trainer.should_stop = True

        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        regularized, not_regularized = [], []
        for module in [self.student_backbone, self.student_head]:
            for n, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                # we do not regularize biases nor Norm parameters
                if n.endswith(".bias") or len(p.shape) == 1:
                    not_regularized.append(p)
                else:
                    regularized.append(p)
        param_groups = [
            {"params": regularized},
            {"params": not_regularized, "weight_decay": 0.0},
        ]

        opt = torch.optim.AdamW(param_groups, lr=self.hparams.lr_start)
        return opt


def main(args):
    L.seed_everything(args.data.seed, workers=True)

    # logger = TensorBoardLogger(
    #    save_dir=args.logger.save_dir,
    #    name=args.logger.name,
    # )

    logger = WandbLogger(
        save_dir=args.logger.save_dir,
        name=args.logger.name,
        mode=args.logger.mode,
        project=args.logger.project,
    )

    pbar_callback = TQDMProgressBar(refresh_rate=args.pbar.refresh_rate)
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=args.ckpt.every_n_epochs,
        save_last=args.ckpt.save_last,
    )

    trainer = L.Trainer(
        default_root_dir=args.logger.save_dir,
        max_epochs=args.trainer.max_epochs,
        devices=args.trainer.devices,
        accelerator=args.trainer.accelerator,
        precision=args.trainer.precision,
        strategy=args.trainer.strategy,
        sync_batchnorm=args.trainer.sync_batchnorm,
        use_distributed_sampler=args.trainer.use_distributed_sampler,
        gradient_clip_val=args.trainer.gradient_clip_val,
        detect_anomaly=args.trainer.detect_anomaly,
        logger=logger,
        benchmark=args.trainer.benchmark,
        callbacks=[checkpoint_callback, pbar_callback],
    )

    args.model.lr_start = (
        args.model.lr_start
        * (args.data.batch_size_per_device * trainer.world_size)
        * args.trainer.accumulate_grad_batches
        / 256.0
    )

    compile_model = args.model.pop("compile")
    model = DINO(**args.model)

    transform_class = get_transform("dino")
    datamodule = PretrainDataModule(args, transform_class)

    fit_model = model

    if compile_model:
        fit_model = torch.compile(model)

    logger.log_hyperparams(args)
    trainer.fit(model=fit_model, datamodule=datamodule)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser("DINO")
    parser.add_class_arguments(DINO, "model")
    parser.add_class_arguments(L.Trainer, "trainer")
    # parser.add_class_arguments(TensorBoardLogger, "logger")
    parser.add_class_arguments(WandbLogger, "logger")
    parser.add_class_arguments(TQDMProgressBar, "pbar")
    parser.add_class_arguments(ModelCheckpoint, "ckpt")
    parser.add_class_arguments(DINOTransform, "transform")
    parser.add_dataclass_arguments(DataOptions, "data")

    parser.add_argument("--model.compile", type=bool, default=False)
    if "logger.mode" not in [a.dest for a in parser._actions]:
        parser.add_argument("--logger.mode", type=str, default="disabled")
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)

    args = parser.parse_args()

    main(args)
