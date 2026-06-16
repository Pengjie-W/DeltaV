# from https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py
import numpy as np
import math, random
from PIL import Image
import torch

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])
    
def center_crop_to_target_aspect_ratio(
    image: torch.Tensor,
    target_height: int,
    target_width: int
) -> torch.Tensor:
    """
    Center-crop an image to the target aspect ratio.

    Args:
        image: input image tensor [C, H, W]
        target_height: target height
        target_width: target width

    Returns:
        cropped image tensor [C, H_crop, W_crop]
    """

    _, orig_height, orig_width = image.shape

    target_aspect_ratio = target_width / target_height
    orig_aspect_ratio = orig_width / orig_height

    if orig_aspect_ratio > target_aspect_ratio:
        # image too wide -> crop width
        crop_height = orig_height
        crop_width = int(orig_height * target_aspect_ratio)

        x_offset = (orig_width - crop_width) // 2
        y_offset = 0

    elif orig_aspect_ratio < target_aspect_ratio:
        # image too tall -> crop height
        crop_width = orig_width
        crop_height = int(orig_width / target_aspect_ratio)

        x_offset = 0
        y_offset = (orig_height - crop_height) // 2

    else:
        # aspect ratio already matches
        return image

    cropped_image = image[
        :,
        y_offset:y_offset + crop_height,
        x_offset:x_offset + crop_width
    ]

    return cropped_image
def random_crop_to_target_aspect_ratio(
    image: torch.Tensor,
    target_height: int,
    target_width: int,
    crop_u: float,
    crop_v: float,
    random_seed: int = None
) -> torch.Tensor:
    """
    Randomly crop an image to match the target aspect ratio, avoiding distortion from a later resize.

    Args:
        image: input image tensor with shape [C, H, W]
        target_height: target height
        target_width: target width
        random_seed: random seed (optional), for reproducibility

    Returns:
        cropped image tensor, still with shape [C, H_crop, W_crop] but with the target aspect ratio
    """
    # set the random seed (if provided)
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    # get the original size of the input image [C, H, W]
    _, orig_height, orig_width = image.shape

    # compute the target and original aspect ratios
    target_aspect_ratio = target_width / target_height
    orig_aspect_ratio = orig_width / orig_height

    # choose the crop strategy based on the aspect ratio
    if orig_aspect_ratio > target_aspect_ratio:
        # original image is wider: crop width, keep height unchanged
        crop_height = orig_height
        # compute the required crop width (to match the target aspect ratio)
        crop_width = int(orig_height * target_aspect_ratio)
        # ensure the crop width does not exceed the original width (a safety check, in theory it never does)
        crop_width = min(crop_width, orig_width)
        # randomly determine the crop start position along the width
        max_x_offset = orig_width - crop_width
        if crop_u is None:
            x_offset = random.randint(0, max_x_offset) if max_x_offset > 0 else 0
        else:
            x_offset = int(crop_u * max_x_offset)

        y_offset = 0
    elif orig_aspect_ratio < target_aspect_ratio:
        # original image is taller: crop height, keep width unchanged
        crop_width = orig_width
        # compute the required crop height (to match the target aspect ratio)
        crop_height = int(orig_width / target_aspect_ratio)
        # ensure the crop height does not exceed the original height
        crop_height = min(crop_height, orig_height)
        # randomly determine the crop start position along the height
        max_y_offset = orig_height - crop_height
        if crop_v is None:
            y_offset = random.randint(0, max_y_offset) if max_y_offset > 0 else 0
        else:
            y_offset = int(crop_v * max_y_offset)


        x_offset = 0
    else:
        # aspect ratio already matches, no crop needed
        return image

    # perform the crop (torch.Tensor indexing: [C, y:y+crop_h, x:x+crop_w])
    cropped_image = image[
        :,
        y_offset: y_offset + crop_height,
        x_offset: x_offset + crop_width
    ]
    return cropped_image

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F_pad
import random


def _compute_padded_size(orig_h: int, orig_w: int, target_h: int, target_w: int):
    """
    Compute the smallest integer size (H_pad, W_pad) that contains the original image and matches the target aspect ratio.
    Target aspect ratio r = target_w / target_h
    """
    r = target_w / target_h
    orig_r = orig_w / orig_h

    if abs(orig_r - r) < 1e-12:
        return orig_h, orig_w

    if orig_r > r:
        # original image is wider: increase height (pad height), keep width unchanged
        W_pad = orig_w
        H_pad = int((W_pad / r) + 0.999999999)  # ceil
        return H_pad, W_pad
    else:
        # original image is taller: increase width (pad width), keep height unchanged
        H_pad = orig_h
        W_pad = int((H_pad * r) + 0.999999999)  # ceil
        return H_pad, W_pad


def center_pad_to_target_aspect_ratio(
    image: torch.Tensor,
    target_height: int,
    target_width: int,
    pad_value: int = 255,
) -> torch.Tensor:
    """
    Center padding: pad on all sides to reach the target aspect ratio (no cropping).
    """
    assert image.ndim == 3, f"Expect [C,H,W], got {tuple(image.shape)}"
    _, orig_h, orig_w = image.shape

    H_pad, W_pad = _compute_padded_size(orig_h, orig_w, target_height, target_width)
    if H_pad == orig_h and W_pad == orig_w:
        return image

    pad_h = H_pad - orig_h
    pad_w = W_pad - orig_w
    assert pad_h >= 0 and pad_w >= 0

    # center: split top/bottom and left/right evenly
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # F.pad order is (left, right, top, bottom)
    return F_pad.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=pad_value)


def random_pad_to_target_aspect_ratio(
    image: torch.Tensor,
    target_height: int,
    target_width: int,
    pad_u: float = None,
    pad_v: float = None,
    random_seed: int = None,
    pad_value: int = 255,
) -> torch.Tensor:
    """
    Random padding: the total pad amount is fixed, but the position of the original image on the new canvas is randomized (no cropping).
    """
    assert image.ndim == 3, f"Expect [C,H,W], got {tuple(image.shape)}"
    _, orig_h, orig_w = image.shape

    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    H_pad, W_pad = _compute_padded_size(orig_h, orig_w, target_height, target_width)
    if H_pad == orig_h and W_pad == orig_w:
        return image

    pad_h = H_pad - orig_h
    pad_w = W_pad - orig_w
    assert pad_h >= 0 and pad_w >= 0

    # random: assign a random amount from [0, pad_h] / [0, pad_w] to top/left, the rest goes to bottom/right
    if pad_v is None:
        pad_top = random.randint(0, pad_h) if pad_h > 0 else 0
    else:
        pad_top = int(pad_v * pad_h)
    pad_bottom = pad_h - pad_top
    
    if pad_u is None:
        pad_left = random.randint(0, pad_w) if pad_w > 0 else 0
    else:
        pad_left = int(pad_u * pad_w)
    pad_right = pad_w - pad_left

    return F_pad.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=pad_value)