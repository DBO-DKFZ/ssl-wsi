from typing import Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torch import Tensor

cv2.setNumThreads(1)


def get_transform(name: str):
    name = name.strip().lower()
    if name == "dino":
        return DINOTransform
    if (name == "msn") or (name == "pmsn"):
        return MSNTransform
    else:
        raise NotImplementedError(f"{name} is not implemented.")


class AlbumentationWrapper:
    """Wrapper around albumentations."""

    def __init__(self, transforms):
        self.tfms = A.Compose(transforms)

    def __call__(self, image):
        return self.tfms(image=image)["image"]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.tfms.__repr__()})"


class ToTensor:
    """Wrapper around albumentations."""

    def __call__(self, image):
        return torch.from_numpy(image.transpose(2, 0, 1)) / 255

    def __repr__(self):
        return f"{self.__class__.__name__}"


class DINOTransform(MultiViewTransform):
    """Implements the global and local view augmentations for DINO [0].

    This class generates two global and a user defined number of local views
    for each image in a batch. The code is adapted from [1].

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        global_crop_size:
            Crop size of the global views.
        global_crop_scale:
            Tuple of min and max scales relative to global_crop_size.
        local_crop_size:
            Crop size of the local views.
        local_crop_scale:
            Tuple of min and max scales relative to local_crop_size.
        n_local_views:
            Number of generated local views.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Tuple of probabilities to apply gaussian blur on the different
            views. The input is ordered as follows:
            (global_view_0, global_view_1, local_views)
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        kernel_scale:
            Old argument. Value is deprecated in favor of sigmas. If set, the old behavior applies and `sigmas` is ignored.
            Used to scale the `kernel_size` of a factor of `kernel_scale`
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        solarization:
            Probability to apply solarization on the second global view.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        global_crop_size: int = 224,
        global_crop_scale: Tuple[float, float] = (0.4, 1.0),
        local_crop_size: int = 96,
        local_crop_scale: Tuple[float, float] = (0.05, 0.4),
        n_local_views: int = 6,
        flip_prob: float = 0.5,
        rr_prob: float = 0.5,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: Tuple[float, float, float] = (1.0, 0.1, 0.5),
        sigmas: Tuple[float, float] = (0.1, 2),
        solarization_prob: float = 0.2,
        normalize: Union[None, dict] = IMAGENET_NORMALIZE,
    ):
        # first global crop
        global_transform_0 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            flip_prob=flip_prob,
            rr_prob=rr_prob,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[0],
            sigmas=sigmas,
            solarization_prob=0,
            normalize=normalize,
        )

        # second global crop
        global_transform_1 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            flip_prob=flip_prob,
            rr_prob=rr_prob,
            cj_prob=cj_prob,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[1],
            sigmas=sigmas,
            solarization_prob=solarization_prob,
            normalize=normalize,
        )

        # transformation for the local small crops
        local_transform = DINOViewTransform(
            crop_size=local_crop_size,
            crop_scale=local_crop_scale,
            flip_prob=flip_prob,
            rr_prob=rr_prob,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[2],
            sigmas=sigmas,
            solarization_prob=0,
            normalize=normalize,
        )
        local_transforms = [local_transform] * n_local_views
        transforms = [global_transform_0, global_transform_1]
        transforms.extend(local_transforms)
        super().__init__(transforms)


class DINOViewTransform:
    def __init__(
        self,
        crop_size: int = 224,
        crop_scale: Tuple[float, float] = (0.4, 1.0),
        flip_prob: float = 0.5,
        rr_prob: float = 0.5,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 1.0,
        sigmas: Tuple[float, float] = (0.1, 2),
        solarization_prob: float = 0.2,
        normalize: Optional[dict] = IMAGENET_NORMALIZE,
    ):
        cj_args = dict(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            AlbumentationWrapper(
                [
                    A.RandomResizedCrop(
                        crop_size,
                        crop_size,
                        scale=crop_scale,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    A.RandomRotate90(p=rr_prob),
                    A.Flip(p=flip_prob),
                    A.ColorJitter(p=cj_prob, **cj_args),
                    A.ToGray(p=random_gray_scale),
                    A.GaussianBlur(
                        blur_limit=(3, 7), sigma_limit=sigmas, p=gaussian_blur
                    ),
                    A.Solarize(p=solarization_prob),
                ]
            ),
            ToTensor(),
        ]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: np.ndarray) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        return self.transform(image)


class MSNTransform(MultiViewTransform):
    """Implements the transformations for MSN [0].

    Generates a set of random and focal views for each input image. The generated output
    is (views, target, filenames) where views is list with the following entries:
    [random_views_0, random_views_1, ..., focal_views_0, focal_views_1, ...].

    - [0]: Masked Siamese Networks, 2022: https://arxiv.org/abs/2204.07141

    Attributes:
        random_size:
            Size of the random image views in pixels.
        focal_size:
            Size of the focal image views in pixels.
        random_views:
            Number of random views to generate.
        focal_views:
            Number of focal views to generate.
        random_crop_scale:
            Minimum and maximum size of the randomized crops for the relative to random_size.
        focal_crop_scale:
            Minimum and maximum size of the randomized crops relative to focal_size.
        cj_prob:
            Probability that color jittering is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        random_gray_scale:
            Probability of conversion to grayscale.
        flip_prob:
            Probability that vertical/horizontal flip is applied.
        rr_prob:
            Probability that 90-degree rotations are applied.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.
    """

    def __init__(
        self,
        random_size: int = 224,
        focal_size: int = 96,
        random_views: int = 2,
        focal_views: int = 10,
        random_crop_scale: Tuple[float, float] = (0.3, 1.0),
        focal_crop_scale: Tuple[float, float] = (0.05, 0.3),
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        random_gray_scale: float = 0.2,
        flip_prob: float = 0.5,
        rr_prob: float = 0.5,
        normalize: dict = IMAGENET_NORMALIZE,
    ):
        random_view_transform = MSNViewTransform(
            crop_size=random_size,
            crop_scale=random_crop_scale,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            random_gray_scale=random_gray_scale,
            flip_prob=flip_prob,
            rr_prob=rr_prob,
            normalize=normalize,
        )
        focal_view_transform = MSNViewTransform(
            crop_size=focal_size,
            crop_scale=focal_crop_scale,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            random_gray_scale=random_gray_scale,
            flip_prob=flip_prob,
            rr_prob=rr_prob,
            normalize=normalize,
        )
        transforms = [random_view_transform] * random_views
        transforms += [focal_view_transform] * focal_views
        super().__init__(transforms=transforms)


class MSNViewTransform:
    def __init__(
        self,
        crop_size: int = 224,
        crop_scale: Tuple[float, float] = (0.3, 1.0),
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        random_gray_scale: float = 0.2,
        flip_prob: float = 0.5,
        rr_prob: float = 0.5,
        normalize: dict = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        cj_args = dict(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            AlbumentationWrapper(
                [
                    A.RandomResizedCrop(
                        crop_size,
                        crop_size,
                        scale=crop_scale,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    A.RandomRotate90(p=rr_prob),
                    A.Flip(p=flip_prob),
                    A.ColorJitter(p=cj_prob, **cj_args),
                    A.ToGray(p=random_gray_scale),
                    A.GaussianBlur(
                        blur_limit=(3, 7), sigma_limit=sigmas, p=gaussian_blur
                    ),
                ]
            ),
            ToTensor(),
        ]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: np.ndarray) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        return self.transform(image)
