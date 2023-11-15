from copy import deepcopy
from typing import Union, Optional, List, Dict

import numpy as np
import tensorflow as tf
from code_loader.contract.responsedataclasses import BoundingBox
from numpy._typing import NDArray
from src.config import CONFIG


def polygon_to_bbox(vertices):
    """
    Converts a polygon representation to a bounding box representation.

    Args:
        vertices (list): List of vertices defining the polygon. The vertices should be in the form [x1, y1, x2, y2, ...].

    Returns:
        list: Bounding box representation of the polygon in the form [x, y, width, height].

    Note:
        - The input list of vertices should contain x and y coordinates in alternating order.
        - The function calculates the minimum and maximum values of the x and y coordinates to determine the bounding box.
        - The bounding box representation is returned as [x, y, width, height], where (x, y) represents the center point of the
          bounding box, and width and height denote the size of the bounding box.
    """

    xs = [x for i, x in enumerate(vertices) if i % 2 == 0]
    ys = [x for i, x in enumerate(vertices) if i % 2 != 0]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    # Bounding box representation: (x, y, width, height)
    bbox = [(min_x + max_x) / 2., (min_y + max_y) / 2., max_x - min_x, max_y - min_y]

    return bbox


def count_obj_masks_occlusions(masks: Union[np.ndarray, None], occlusion_threshold: float) -> int:
    """
    Counts the occluded masks based on Intersection over Union (IOU).

    Args:
        masks (Union[np.ndarray, None]): Masks represented as a NumPy array. Can be None if no masks are provided.
        occlusion_threshold (float): Threshold value for determining occlusion based on IOU.

    Returns:
        int: Number of occluded masks based on the specified occlusion threshold.

    """

    if masks is None:
        return 0

    if masks[0, ...].shape != CONFIG["IMAGE_SIZE"]:
        masks = tf.image.resize(masks[..., None], CONFIG["IMAGE_SIZE"], tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
        masks = masks.numpy()

    num_masks = len(masks)

    # Reshape masks to have a third dimension
    masks = np.expand_dims(masks, axis=-1)

    # Create tiled versions of masks for element-wise addition
    tiled_masks = np.broadcast_to(masks, (num_masks, num_masks, masks.shape[1], masks.shape[2], 1))
    tiled_masks_transposed = np.transpose(tiled_masks, axes=(1, 0, 2, 3, 4))

    # Compute overlay matrix
    overlay = tiled_masks + tiled_masks_transposed

    # Exclude same mask occlusions and duplicate pairs
    mask_indices = np.triu_indices(num_masks, k=1)
    overlay = overlay[mask_indices]

    intersection = np.sum(overlay > 1, axis=(-1, -2, -3))
    union = np.sum(overlay > 0, axis=(-1, -2, -3))

    iou = intersection / union
    return int(np.sum(iou > occlusion_threshold))


def remove_label_from_bbs(bbs_object_array, removal_label, add_to_label):
    new_bb_arr = []
    for bb in bbs_object_array:
        if bb.label != removal_label:
            new_bb = deepcopy(bb)
            new_bb.label = new_bb.label + "_" + add_to_label
            new_bb_arr.append(new_bb)
    return new_bb_arr


def calculate_overlap(box1, box2):
    # Extract coordinates of the bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    # Calculate the overlap area
    overlap_area = w_intersection * h_intersection

    return overlap_area


def get_argmax_map_and_separate_masks(image, bbs, masks):
    image_size = image.shape[:2]
    argmax_map = np.zeros(image_size, dtype=np.uint8)
    cats_dict = {}
    separate_masks = []
    for bb, mask in zip(bbs, masks):
        if mask.shape != image_size:
            resize_mask = tf.image.resize(mask[..., None], image_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
            if not isinstance(resize_mask, np.ndarray):
                resize_mask = resize_mask.numpy()
        else:
            resize_mask = mask
        resize_mask = resize_mask.astype(bool)
        label = bb.label
        instance_number = cats_dict.get(label, 0)
        # update counter if reach max instances we treat the last objects as one
        cats_dict[label] = instance_number + 1 if instance_number < CONFIG[
            "MAX_INSTANCES_PER_CLASS"] else instance_number
        label_index = CONFIG["CATEGORIES"].index(label) * CONFIG["MAX_INSTANCES_PER_CLASS"] + cats_dict[label]
        if label == 'Tote':
            empty = argmax_map == 0
            tote = (argmax_map >= CONFIG["CATEGORIES"].index(label) * CONFIG["MAX_INSTANCES_PER_CLASS"]) & \
                   (argmax_map < CONFIG["CATEGORIES"].index(label) * (CONFIG["MAX_INSTANCES_PER_CLASS"] + 1))
            argmax_map[(empty | tote) & resize_mask] = label_index
        else:
            argmax_map[resize_mask] = label_index
        if bb.label == 'Object':
            separate_masks.append(resize_mask)
    argmax_map[argmax_map == 0] = len(CONFIG['INSTANCES']) + 1
    argmax_map -= 1
    return {"argmax_map": argmax_map, "separate_masks": separate_masks}


def extract_and_cache_bboxes(idx: int, data: Dict):
    x = data['samples'][idx]
    coco = data['cocofile']
    ann_ids = coco.getAnnIds(imgIds=x['id'])
    anns = coco.loadAnns(ann_ids)
    bboxes = np.zeros([CONFIG['MAX_BB_PER_IMAGE'], 5])
    max_anns = min(CONFIG['MAX_BB_PER_IMAGE'], len(anns))
    for i in range(max_anns):
        ann = anns[i]
        if isinstance(ann['segmentation'], list) and ann['category_id'] <= CONFIG['CLASSES']:
            img_size = (x['height'], x['width'])
            class_id = ann['category_id']
            bbox = polygon_to_bbox(ann['segmentation'][0])
            bbox /= np.array((img_size[1], img_size[0], img_size[1], img_size[0]))
            bboxes[i, :4] = bbox
            bboxes[i, 4] = class_id
    bboxes[max_anns:, 4] = CONFIG['BACKGROUND_LABEL']
    return bboxes
