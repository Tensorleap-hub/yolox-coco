from typing import Union, Tuple

import numpy as np
from PIL import Image, ImageOps

from yolox.config import CONFIG
from yolox.utils.dataclasses import RescaleMetadata, PaddingCoordinates


def preprocess_image(image: Image, get_metadata: bool = False) -> Union[
    np.ndarray, Tuple[RescaleMetadata, PaddingCoordinates]]:
    width, height = image.size
    ar = width / height
    scaling_size = CONFIG['SCALE_TO_SIZE']
    target_shape = CONFIG['IMAGE_SIZE']
    # Calculate the new dimensions
    if width >= height:
        new_width = scaling_size
        new_height = int(scaling_size // ar)
    else:
        new_height = scaling_size
        new_width = int(scaling_size * ar)

    # Resize the image
    resized_img = image.resize((new_width, new_height))

    # Calculate padding values
    padding_top = 0  # No padding at the top
    padding_left = 0  # No padding on the left
    padding_bottom = target_shape[1] - new_height
    padding_right = target_shape[0] - new_width
    pad_width = max(0, padding_right)
    pad_height = max(0, padding_bottom)

    # Calculate left, top, right, bottom padding
    padding = (0, 0, pad_width, pad_height)

    # Pad the image with zeros
    padded_img = ImageOps.expand(resized_img, padding, fill=0)

    image_array = np.asarray(padded_img) / 255.0
    if get_metadata:
        rescale_params = RescaleMetadata(original_shape=(width, height),
                                         scale_factor_h=new_height / height,
                                         scale_factor_w=new_width / width)
        pad_params = PaddingCoordinates(*padding)
        return rescale_params, pad_params
    return image_array.astype(np.float32)
