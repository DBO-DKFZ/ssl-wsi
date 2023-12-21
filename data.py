import logging
import os
from collections import defaultdict
from typing import Optional

import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from slide_tools.tile_level import TileLevelDataset

cv2.setNumThreads(1)  # Just to be sure that opencv does not use all threads

logger = logging.getLogger("lightning.pytorch")


class PILFromNumpy:
    """For enabling PIL transforms on WSI regions"""

    def __init__(self, after=None):
        self.after = after

    def __call__(self, image):
        if self.after is not None:
            return self.after(Image.fromarray(image))
        else:
            return Image.fromarray(image)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class ModifiedTileLevelDataset(TileLevelDataset):
    """TileLevelDataset for pre-training (without targets)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        return super().__getitem__(idx)["img"]


class ModifiedImageFolder(datasets.ImageFolder):
    """ImageFolder with tunable length and class-balanced reloading."""

    def __init__(
        self,
        *args,
        img_per_epoch: Optional[int] = None,
        replace=False,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reloadable = False
        self.replace = False
        self.seed = seed
        self.all_samples = self.samples

        if img_per_epoch is not None:
            if img_per_epoch < len(self):
                self.img_per_class = max(1, img_per_epoch // len(self.classes))
                self.all_samples = self.samples[:]
                self.class_indices = defaultdict(list)

                for idx, (_, class_idx) in enumerate(self.all_samples):
                    self.class_indices[class_idx].append(idx)

                self.reloadable = True
                self.reload(epoch=0)

    def reload(self, epoch=None):
        if self.reloadable:
            samples = []
            seed = self.seed + epoch if epoch is not None else None
            rng = np.random.default_rng(seed)

            for _, indices in self.class_indices.items():
                size = self.img_per_class
                if not self.replace:
                    size = min(len(indices), size)
                class_samples = rng.choice(indices, size=size, replace=self.replace)
                samples.extend([self.all_samples[i] for i in class_samples])

            self.samples = samples

    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]


class CombinedDataset(torch.utils.data.ConcatDataset):
    """ConcatDataset with reloading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reload(self, epoch=None):
        for ds in self.datasets:
            ds.reload(epoch=epoch)


class PretrainDataModule(L.LightningDataModule):
    def __init__(self, args, transform_class):
        super().__init__()
        self.save_hyperparameters(args)
        self.transform_class = transform_class

    def setup(self, stage: Optional[str] = None):
        assert stage == "fit"

        # Helper to join root onto relative path series
        rootify_ = lambda path: os.path.join(self.hparams.data.root, path)

        def rootify(series):
            if series is None:
                return None
            else:
                return series.apply(rootify_)

        transform = self.transform_class(**self.hparams.transform)

        datasets = []

        if self.trainer.global_rank == 0:
            logger.info("Datasets")
            logger.info("--------")

        if not self.hparams.data.without_wsi:
            frame = pd.read_csv(self.hparams.data.frame)

            if self.hparams.data.limit_slides > 0:
                frame = frame.iloc[: self.hparams.data.limit_slides]

            tiles_per_slide = None
            if self.hparams.data.epoch_length is not None:
                tiles_per_slide = self.hparams.data.epoch_length / len(frame)
                if self.hparams.data.natural_ratio > 0:
                    tiles_per_slide *= 1 - self.hparams.data.natural_ratio
                tiles_per_slide = int(tiles_per_slide)

            wsi_dataset = ModifiedTileLevelDataset(
                slide_paths=rootify(frame.slide),
                annotation_paths=rootify(frame.get("annotation")),
                img_tfms=transform,
                simplify_tolerance=self.hparams.data.simplify_tolerance,
                level=self.hparams.data.level,
                centroid_in_annotation=self.hparams.data.centroid_in_annotation,
                allow_out_of_bounds=self.hparams.data.allow_out_of_bounds,
                shuffle=False,
                verbose=self.hparams.data.verbose,
                simple_epoch=self.hparams.data.simple_epoch,
                balance_size_by=tiles_per_slide,
                seed=self.hparams.data.seed,
                size=self.hparams.data.tile_size_multiple,
                backend=self.hparams.data.backend,
            )
            datasets.append(wsi_dataset)

            if self.trainer.global_rank == 0:
                logger.info(f"  WSI: {len(wsi_dataset.slides)} slides")
                logger.info(
                    f"  WSI: {sum(len(s.regions) for s in wsi_dataset.slides)} regions"
                )
                logger.info(
                    f"  WSI: {tiles_per_slide or 'unspecified'} regions per slide"
                )
                logger.info(f"  WSI: {len(wsi_dataset)} regions per epoch")

        if self.hparams.data.natural_ratio > 0 or self.hparams.data.without_wsi:

            def loader(path):
                return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

            img_per_epoch = None
            if not self.hparams.data.without_wsi:
                epoch_length = self.hparams.data.epoch_length or len(wsi_dataset)
                img_per_epoch = int(epoch_length * self.hparams.data.natural_ratio)

            imagenet_dataset = ModifiedImageFolder(
                self.hparams.data.natural_root,
                transform=transform,
                img_per_epoch=img_per_epoch,
                replace=self.hparams.data.natural_replace,
                loader=loader,
            )
            datasets.append(imagenet_dataset)

            if self.trainer.global_rank == 0:
                logger.info(f"  Natural: {len(imagenet_dataset.all_samples)} images")
                logger.info(f"  Natural: {len(imagenet_dataset.classes)} classes")
                logger.info(f"  Natural: {len(imagenet_dataset)} images per epoch")

        self.dataset = CombinedDataset(datasets) if len(datasets) > 1 else datasets[0]
        if self.trainer.global_rank == 0:
            logger.info(f"  -> {len(self.dataset)} samples per epoch")

    def train_dataloader(
        self,
        epoch: Optional[int] = None,
    ):
        self.dataset.reload(epoch or self.trainer.current_epoch)
        return DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.hparams.data.batch_size_per_device,
            drop_last=True,
            num_workers=self.hparams.data.num_workers,
            pin_memory=True,
        )
