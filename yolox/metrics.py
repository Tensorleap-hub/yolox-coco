import tensorflow as tf

from yolox.utils.yolox_loss import get_yolox_od_losses


def od_metrics_dict(y_true: tf.Tensor, y_pred: tf.Tensor):
    conf_loss, class_loss, reg_loss, pose_loss, type_loss = get_yolox_od_losses(y_true, y_pred)
    return {'regression_loss': reg_loss,
            'classification_loss': class_loss,
            'objectness_loss': conf_loss,
            'pose_loss': pose_loss,
            'type_loss': type_loss
            }


def placeholder_loss(y_true, y_pred: tf.Tensor) -> tf.Tensor:  # return batch

    return tf.reduce_mean(y_true, axis=-1) * 0
