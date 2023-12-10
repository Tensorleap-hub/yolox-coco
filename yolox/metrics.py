import tensorflow as tf
from code_loader.helpers.detection.utils import xywh_to_xyxy_format

from yolox.config import CONFIG

from yolox.utils.yolo_utils import run_single_scale
from yolox.utils.yolox_loss import get_od_losses


def od_metrics_dict(y_true: tf.Tensor, y_pred: tf.Tensor):
    reg_loss, cls_loss, conf_loss = get_od_losses(y_true, y_pred)
    return {'regression_loss': reg_loss,
            'classification_loss': cls_loss,
            'objectness_loss': conf_loss}


def placeholder_loss(y_true, y_pred: tf.Tensor) -> tf.Tensor:  # return batch

    return tf.reduce_mean(y_true, axis=-1) * 0
