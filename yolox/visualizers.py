from typing import List
import tensorflow as tf
import numpy as np
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.helpers.detection.utils import xyxy_to_xywh_format

from yolox.config import CONFIG
from yolox.utils.general_utils import bb_array_to_object


# Visualizers
# def pred_bb_decoder(image: np.ndarray, reg: tf.Tensor, cls: tf.Tensor) -> LeapImageWithBBox:
#     """
#     Overlays the BB predictions on the image
#     """
#     # x1,y1,x2,y2
#     reg_fixed = xyxy_to_xywh_format(reg) / image.shape[0]
#     bb_object = get_predict_bbox_list(reg_fixed, cls)
#     return LeapImageWithBBox(data=(image * 255).astype(np.float32), bounding_boxes=bb_object)

def pred_bb_visualizer(image: np.ndarray, y_pred: tf.Tensor) -> LeapImageWithBBox:
    bboxes = []
    for bbox in y_pred:
        only_bbox = bbox[:4]
        x, y = (only_bbox[0] + only_bbox[2]) / 2, (only_bbox[1] + only_bbox[3]) / 2
        w, h = np.abs(only_bbox[2] - only_bbox[0]), np.abs(only_bbox[3] - only_bbox[1])
        confidence = bbox[4] * np.max(bbox[5:])
        if confidence < CONFIG['CONF_THRESH']:
            continue
        bboxes.append(
            BoundingBox(
                x=x,
                y=y,
                width=w,
                height=h,
                confidence=confidence,
                label=''
            ))
    return LeapImageWithBBox((image * 255).astype(np.float32), bboxes)


def gt_bb_decoder(image: np.ndarray, bb_gt: np.ndarray) -> LeapImageWithBBox:
    """
    This function overlays ground truth bounding boxes (BBs) on the input image.

    Parameters:
    image (np.ndarray): The input image for which the ground truth bounding boxes need to be overlaid.
    bb_gt (np.ndarray): The ground truth bounding box array for the input image.

    Returns:
    An instance of LeapImageWithBBox containing the input image with ground truth bounding boxes overlaid.
    """
    bb_object: List[BoundingBox] = bb_array_to_object(bb_gt, iscornercoded=False, bg_label=CONFIG['BACKGROUND_LABEL'],
                                                      is_gt=True)
    return LeapImageWithBBox(data=(image * 255).astype(np.float32), bounding_boxes=bb_object)
