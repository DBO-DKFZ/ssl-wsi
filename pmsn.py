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
from lightly.loss import PMSNLoss
from lightly.models.modules.heads import MSNProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch import nn
from transforms import MSNTransform, get_transform
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


class PMSN(L.LightningModule):
    def __init__(
        self,
        arch: str = "tiny",
        image_size: int = 224,
        patch_size: int = 16,
        drop_path_rate: float = 0.0,
        pos_emb: str = "learned",
        hidden_dim: int = 2048,
        output_dim: int = 256,
        mask_ratio: float = 0.15,
        num_proto: int = 1024,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        regularization_weight: float = 1.0,
        power_law_exponent: float = 0.25,
        gather_distributed: bool = True,
        lr_start: float = 0.001,
        lr_final: float = 1e-6,
        lr_warmup_epochs: int = 15,
        wd_start: float = 0.04,
        wd_final: float = 0.4,
        mm_start: float = 0.996,
        mm_final: float = 1.0,
        T_start: float = 0.25,
        T_final: float = 0.25,
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
        self.teacher_backbone = get_vit(
            arch=self.hparams.arch,
            image_size=self.hparams.image_size,
            patch_size=self.hparams.patch_size,
            drop_path_rate=0.0,
            pos_emb=self.hparams.pos_emb,
        )
        self.student_head = MSNProjectionHead(
            input_dim=self.student_backbone.num_features,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=self.hparams.output_dim,
        )
        self.teacher_head = MSNProjectionHead(
            input_dim=self.student_backbone.num_features,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=self.hparams.output_dim,
        )

        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.prototypes = nn.Linear(
            self.hparams.output_dim, self.hparams.num_proto, bias=False
        ).weight

        self.criterion = PMSNLoss(
            temperature=self.hparams.temperature,
            sinkhorn_iterations=self.hparams.sinkhorn_iterations,
            regularization_weight=self.hparams.regularization_weight,
            power_law_exponent=self.hparams.power_law_exponent,
            gather_distributed=self.hparams.gather_distributed,
        )

        self.lr_sch = None
        self.wd_sch = None
        self.mm_sch = None
        self.T_sch = None

    def on_train_start(self):
        assert self.trainer.estimated_stepping_batches is not None
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = math.ceil(total_steps / self.trainer.max_epochs)

        # Note: all schedules in MSN have total_steps = 1.25 * total_steps
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
        # Note: here cosine, linear in MSN
        self.mm_sch = cosine_scheduler(
            start_value=self.hparams.mm_start,
            final_value=self.hparams.mm_final,
            total_steps=total_steps,
        )
        # Note: here cosine, linear in MSN
        self.T_sch = cosine_scheduler(
            start_value=self.hparams.T_start,
            final_value=self.hparams.T_final,
            total_steps=total_steps,
        )

    def update_rates(self):
        lr = self.lr_sch[self.global_step]
        wd = self.wd_sch[self.global_step]
        mm = self.mm_sch[self.global_step]
        T = self.T_sch[self.global_step]

        update_momentum(self.student_backbone, self.teacher_backbone, m=mm)
        update_momentum(self.student_head, self.teacher_head, m=mm)
        # update learning rate, weight decay
        for i, param_group in enumerate(self.optimizers().param_groups):
            param_group["lr"] = lr
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd

        return lr, wd, mm, T

    def training_step(self, batch, batch_idx):
        lr, wd, mm, T = self.update_rates()

        views = [view.to(self.device, non_blocking=True) for view in batch]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_out = self.teacher_backbone(targets)
        targets_out = self.teacher_head(targets_out)
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = self.criterion(
            anchors=anchors_out,
            targets=targets_out,
            prototypes=self.prototypes.data,
            target_sharpen_temperature=T,
        )

        self.log("learning_rate", lr)
        self.log("weight_decay", wd)
        self.log("momentum", mm)
        self.log("teacher_temp", T)
        self.log("loss", loss, True)

        return loss

    def encode_masked(self, anchors):
        batch_size, _, _, width = anchors.shape
        seq_length = (width // self.hparams.patch_size) ** 2
        num_keep = int(seq_length * (1 - self.hparams.mask_ratio))
        noise = torch.rand(batch_size, seq_length, device=self.device)
        indices = torch.argsort(noise, dim=1)
        idx_keep = indices[:, :num_keep]
        out = self.student_backbone(anchors, idx_keep)
        return self.student_head(out)

    def configure_optimizers(self):
        regularized, not_regularized = [], [self.prototypes]
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
        profiler=args.trainer.profiler,
        accumulate_grad_batches=args.trainer.accumulate_grad_batches,
        callbacks=[checkpoint_callback, pbar_callback],
        limit_train_batches=args.trainer.limit_train_batches,
    )

    args.model.lr_start = (
        args.model.lr_start
        * (args.data.batch_size_per_device * trainer.world_size)
        * args.trainer.accumulate_grad_batches
        / 1024.0
    )

    compile_model = args.model.pop("compile")
    model = PMSN(**args.model)

    transform_class = get_transform("pmsn")
    datamodule = PretrainDataModule(args, transform_class)

    fit_model = model

    if compile_model:
        fit_model = torch.compile(model)

    logger.log_hyperparams(args)
    trainer.fit(model=fit_model, datamodule=datamodule)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser("PMSN")
    parser.add_class_arguments(PMSN, "model")
    parser.add_class_arguments(L.Trainer, "trainer")
    # parser.add_class_arguments(TensorBoardLogger, "logger")
    parser.add_class_arguments(WandbLogger, "logger")
    parser.add_class_arguments(TQDMProgressBar, "pbar")
    parser.add_class_arguments(ModelCheckpoint, "ckpt")
    parser.add_class_arguments(MSNTransform, "transform")
    parser.add_dataclass_arguments(DataOptions, "data")

    parser.add_argument("--model.compile", type=bool, default=False)
    if "logger.mode" not in [a.dest for a in parser._actions]:
        parser.add_argument("--logger.mode", type=str, default="disabled")
    parser.add_argument("--config", action=jsonargparse.ActionConfigFile)

    args = parser.parse_args()

    main(args)
