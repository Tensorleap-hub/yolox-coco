import tensorflow as tf
import numpy as np
from code_loader.helpers.detection.utils import xyxy_to_xywh_format

from yolonas.config import CONFIG
from yolonas.utils.yolo_utils import decoder


def custom_yolo_nas_loss(y_true, reg: tf.Tensor, cls: tf.Tensor):
    reg_fixed = xyxy_to_xywh_format(reg) / CONFIG['IMAGE_SIZE'][0]
    res = decoder(loc_data=[reg_fixed],
                  conf_data=[cls],
                  prior_data=[None],
                  from_logits=False,
                  decoded=True)
    if len([bb for bb in res if bb.size != 0]) == 0:
        return tf.convert_to_tensor([0.0])

    regression_loss = []
    y_pred_lst = [tf.convert_to_tensor(bb) for bb in res]
    for i, y_pred in enumerate(y_pred_lst):
        if y_pred.shape != 0:
            # Extract confidence, coordinates, and class predictions
            pred_confidence = y_pred[:, 0]
            pred_boxes = y_pred[:, 1:5]
            pred_class_probs = y_pred[:, 5:]

            true_boxes = y_true[i, :, :4]
            true_class_probs = y_true[i, :, 4:]

            # Calculate IoU for each pair of predicted and true bounding boxes
            pred_boxes = tf.cast(tf.expand_dims(pred_boxes, axis=1), tf.float32)  # Shape: (n_pred_boxes, 1, 4)
            true_boxes = tf.cast(tf.expand_dims(true_boxes, axis=0), tf.float32)  # Shape: (1, n_true_boxes, 4)

            mask = tf.cast(tf.expand_dims(y_true[i, :, 4], 0) != CONFIG['BACKGROUND_LABEL'], tf.float32)
            mask_expanded = tf.expand_dims(mask, axis=-1)

            regression_loss.append(tf.keras.losses.Huber()(true_boxes * mask_expanded, pred_boxes * mask_expanded))
        else:
            regression_loss.append(tf.convert_to_tensor(0.0))

    return tf.convert_to_tensor(regression_loss)


def placeholder_loss(y_true, reg: tf.Tensor, cls: tf.Tensor) -> tf.Tensor:  # return batch

    return tf.reduce_mean(y_true, axis=-1) * 0
