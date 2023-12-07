from typing import List
import tensorflow as tf
import numpy as np
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.helpers.detection.utils import xyxy_to_xywh_format

from yolox.config import CONFIG
from yolox.utils.general_utils import bb_array_to_object
from yolox.utils.yolox_loss import decode_outputs


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
    decoded_output = decode_outputs(y_pred[None, ...])[0].numpy()
    xywh = decoded_output[..., :4]
    predicted_classes = np.argmax(decoded_output[:, 5:], -1)
    xywh /= [*CONFIG['IMAGE_SIZE'][::-1], *CONFIG['IMAGE_SIZE'][::-1]]  # in absolute units
    for i in range(len(xywh)):
        confidence = decoded_output[i, 4] * np.max(decoded_output[i, 5:])
        if confidence < CONFIG['CONF_THRESH']:
            continue
        bbox = xywh[i]
        bboxes.append(
            BoundingBox(
                x=bbox[0],
                y=bbox[1],
                width=bbox[2],
                height=bbox[3],
                confidence=confidence,
                label=CONFIG['class_id_to_name'].get(predicted_classes[i].astype(int))
            ))
    image = (image - np.min(image)) / np.max(image - np.min(image)) * 255
    return LeapImageWithBBox(image.astype(np.uint8), bboxes)


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
    image = (image - np.min(image)) / np.max(image - np.min(image)) * 255
    return LeapImageWithBBox(data=image.astype(np.uint8), bounding_boxes=bb_object)
