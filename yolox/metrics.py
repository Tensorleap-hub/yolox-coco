import tensorflow as tf
from code_loader.helpers.detection.utils import xywh_to_xyxy_format

from yolox.config import CONFIG

from yolox.utils.yolo_utils import run_single_scale


def compute_losses(y_true: tf.Tensor, y_pred: tf.Tensor):
    batch = y_true.shape[0]
    pred_scales = []
    for i, scale in enumerate(CONFIG['FEATURE_MAPS']):
        pred_scale = y_pred[:, :, i * scale[0] * scale[1]:(i + 1) * scale[0] * scale[1]]
        pred_scale = tf.reshape(pred_scale, [batch, CONFIG['CLASSES'] + 5, scale[0], scale[1]])
        pred_scales.append(pred_scale)

    batch_reg_loss = []
    batch_cls_loss = []
    batch_conf_loss = []
    for batch_i in range(batch):
        true_classes_all = []
        true_confidences_all = []
        reg_loss_all = []
        for i, pred_scale in enumerate(pred_scales):
            true_classes, reg_loss, true_confidences = run_single_scale(pred_scale, y_true)
            true_classes_all.append(true_classes)
            true_confidences_all.append(true_confidences)
            reg_loss_all.append(reg_loss)

        true_classes_all = tf.concat(true_classes_all, axis=0)
        true_confidences_all = tf.concat(true_confidences_all, axis=0)
        reg_loss_all = tf.concat(reg_loss_all, axis=0)

        predicted_classes_all = tf.transpose(y_pred[0, 5:, :])
        classes_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(true_classes_all,
                                                                                 predicted_classes_all)

        mean_reg_bbox_loss = tf.reduce_mean(reg_loss_all)
        mean_reg_bbox_loss = tf.cond(tf.math.is_nan(mean_reg_bbox_loss), lambda: 0.0, lambda: mean_reg_bbox_loss)

        predicted_conf_all = tf.expand_dims(tf.transpose(y_pred[0, 4, :]), axis=-1)
        reg_conf_loss = tf.keras.losses.MeanSquaredError()(true_confidences_all, predicted_conf_all)
        batch_reg_loss.append(mean_reg_bbox_loss)
        batch_cls_loss.append(classes_loss)
        batch_conf_loss.append(reg_conf_loss)

    return tf.transpose(tf.convert_to_tensor([batch_reg_loss])), \
        tf.transpose(tf.convert_to_tensor([batch_cls_loss])), \
        tf.transpose(tf.convert_to_tensor([batch_conf_loss]))


def custom_yolox_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    y_true = tf.cast(y_true, tf.float32)
    y_true_bboxes = xywh_to_xyxy_format(y_true[:, :, :4])
    y_true = tf.concat([y_true_bboxes, y_true[..., -2:-1]], axis=-1)
    reg_loss, cls_loss, conf_loss = compute_losses(y_true, y_pred)
    sum_loss = tf.reduce_sum([reg_loss, cls_loss, conf_loss], axis=0)
    non_nan_loss = tf.where(tf.math.is_nan(sum_loss), tf.zeros_like(sum_loss), sum_loss)
    return non_nan_loss


def od_metrics_dict(y_true: tf.Tensor, y_pred: tf.Tensor):
    y_true = tf.cast(y_true, tf.float32)
    y_true_bboxes = xywh_to_xyxy_format(y_true[:, :, :4])
    y_true = tf.concat([y_true_bboxes, y_true[..., -2:-1]], axis=-1)
    reg_loss, cls_loss, conf_loss = compute_losses(y_true, y_pred)
    return {'regression_loss': reg_loss,
            'classification_loss': cls_loss,
            'objectness_loss': conf_loss}


def placeholder_loss(y_true, reg: tf.Tensor, cls: tf.Tensor) -> tf.Tensor:  # return batch

    return tf.reduce_mean(y_true, axis=-1) * 0
