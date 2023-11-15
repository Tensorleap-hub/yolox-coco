import os

import numpy as np
from PIL import Image, ImageOps


def load_set(coco, local_filepath, load_union=False):
    # get all images containing given categories
    CATEGORIES = []
    catIds = coco.getCatIds(CATEGORIES)  # Fetch class IDs only corresponding to the Classes
    if not load_union:
        imgIds = coco.getImgIds(catIds=catIds)  # Get all images containing the Category IDs together
    else:  # get images contains any of the classes
        imgIds = set()
        for cat_id in catIds:
            image_ids = coco.getImgIds(catIds=[cat_id])
            imgIds.update(image_ids)
        imgIds = list(imgIds)
    imgs = coco.loadImgs(imgIds)

    image_list = [img for img in os.listdir(os.path.join(local_filepath, 'images')) if img.endswith('.jpg')]
    imgs = [img for img in imgs if img['file_name'] in image_list]
    return imgs


def preprocess_image(image: Image) -> np.ndarray:
    width, height = image.size
    ar = width / height
    scaling_size = 636
    target_shape = [640, 640]
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
    pad_width = max(0, target_shape[0] - new_width)
    pad_height = max(0, target_shape[1] - new_height)

    # Calculate left, top, right, bottom padding
    padding = (0, 0, pad_width, pad_height)

    # Pad the image with zeros
    padded_img = ImageOps.expand(resized_img, padding, fill=0)

    image_array = np.asarray(padded_img) / 255.0
    return image_array.astype(np.float32)
