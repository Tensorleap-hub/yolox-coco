import numpy as np
import tensorflow as tf
from code_loader.helpers.detection.utils import xywh_to_xyxy_format
from numpy._typing import NDArray

from yolox.config import CONFIG


def decode_outputs(pred):
    pred = tf.transpose(pred, (0, 2, 1))  # [batch, n_anchors, n_classes+5]
    grids = []
    strides = []
    for (hsize, wsize), stride in zip(CONFIG['FEATURE_MAPS'], CONFIG['STRIDES']):
        xv, yv = tf.meshgrid(tf.range(wsize), tf.range(hsize))  # Corrected order of ranges
        grid = tf.stack((xv, yv), 2)
        grid = tf.reshape(grid, (1, -1, 2))
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(tf.fill((shape[0], shape[1], 1), stride))

    grids = tf.concat(grids, axis=1)
    grids = tf.cast(grids, tf.float32)
    strides = tf.concat(strides, axis=1)
    strides = tf.cast(strides, tf.float32)

    if CONFIG['predict_log_wh']:
        decoded_outputs = tf.concat([
            (pred[..., 0:2] + grids) * strides,  # x, y
            tf.exp(pred[..., 2:4]) * strides,  # w, h
            pred[..., 4:]  # conf + classes
        ], axis=-1)
    else:
        decoded_outputs = tf.concat([
            (pred[..., 0:2] + grids) * strides,  # x, y
            pred[..., 2:4] * strides,  # w, h
            pred[..., 4:]  # conf + classes
        ], axis=-1)
    return decoded_outputs


def nms(y_pred: NDArray[np.float32], is_xyxy: bool = True) -> tf.Tensor:
    boxes = y_pred[..., :4]
    if not is_xyxy:
        boxes = xywh_to_xyxy_format(boxes)
    scores = y_pred[..., 4]
    selected_indices = tf.image.non_max_suppression(boxes=boxes,
                                                    scores=scores,
                                                    max_output_size=CONFIG['TOP_K'],
                                                    iou_threshold=CONFIG['NMS_THRESH'],
                                                    score_threshold=CONFIG['CONF_THRESH'])
    return selected_indices
