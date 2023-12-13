import tensorflow as tf
from code_loader.helpers.detection.utils import xywh_to_xyxy_format
from code_loader.helpers.detection.utils import jaccard

from yolox.config import CONFIG
from yolox.utils.yolo_utils import decode_outputs


def custom_yolox_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    conf_loss, class_loss, reg_loss_list = get_yolox_od_losses(y_true, y_pred)
    return conf_loss + class_loss + reg_loss_list


def get_yolox_od_losses(y_true: tf.Tensor, y_pred: tf.Tensor):
    # TODO: this modification is temporary and should be removed after updating the loss to match all output logits.
    #  This fix takes only the regression, confidence and class probabilities
    y_pred = y_pred[:, :5 + CONFIG['CLASSES'], :]
    decoded_preds = decode_outputs(y_pred)  # in absolute units
    bboxes = decoded_preds[:, :, :4]
    bboxes /= [*CONFIG['IMAGE_SIZE'][::-1], *CONFIG['IMAGE_SIZE'][::-1]]
    batch = y_true.shape[0]
    bbs_count = bboxes.shape[1]
    reg_loss_list = [tf.constant(0, dtype=tf.float32)] * batch
    class_loss_list = []
    conf_loss_list = []
    for i in range(batch):
        confidence_gt = tf.zeros(bbs_count)
        classes_gt = tf.zeros((bbs_count, CONFIG['CLASSES']), tf.float32)

        y_true_real = y_true[i][y_true[i][..., -1] != CONFIG['BACKGROUND_LABEL']]
        if len(y_true_real) > 0:
            xyxy_gt = xywh_to_xyxy_format(y_true_real[..., :-1])
            xyxy_pred = xywh_to_xyxy_format(bboxes[i, ...])
            ious = jaccard(xyxy_pred, xyxy_gt)  # BB, #GT
            max_iou = tf.reduce_max(ious, -1)
            all_gt_idx = tf.argmax(ious, 1)
            matched_bbs_idx = tf.where(max_iou > CONFIG['IOU_TH'])[..., 0]
            if matched_bbs_idx.shape[0] > 0:
                positive_gt_idx = tf.gather(all_gt_idx, matched_bbs_idx)
                matched_gts_bboxes = tf.gather(xyxy_gt, positive_gt_idx)
                matched_gts_classes = tf.gather(y_true_real[..., -1], positive_gt_idx)
                matched_bbs = tf.gather(xyxy_pred, matched_bbs_idx)
                reg_loss_list[i] = tf.keras.losses.Huber()(matched_bbs, matched_gts_bboxes)

                matched_classes = tf.one_hot(tf.cast(matched_gts_classes, tf.int32),
                                             CONFIG['CLASSES'])
                # compute reg loss
                positive_ious = tf.gather(max_iou, matched_bbs_idx)
                confidence_gt = tf.tensor_scatter_nd_update(
                    confidence_gt, matched_bbs_idx[..., None], positive_ious
                )
                classes_gt = tf.tensor_scatter_nd_update(
                    classes_gt, matched_bbs_idx[..., None], matched_classes
                )
            class_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(classes_gt,
                                                                                    decoded_preds[i, ..., 5:])
            confidence_loss = tf.keras.losses.MeanSquaredError()(confidence_gt, decoded_preds[i, ..., 4])

            conf_loss_list.append(confidence_loss)
            class_loss_list.append(class_loss)
        else:
            reg_loss_list[i] = tf.constant(0.0)
            conf_loss_list.append(tf.constant(0.0))
            class_loss_list.append(tf.constant(0.0))
    conf_loss = tf.stack(conf_loss_list, axis=0)
    class_loss = tf.stack(class_loss_list, axis=0)
    reg_loss = tf.stack(reg_loss_list, axis=0)
    return conf_loss, class_loss, reg_loss
