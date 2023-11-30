from typing import Union, List, Tuple, Dict

import tensorflow as tf
import numpy as np
from code_loader.helpers.detection.utils import xyxy_to_xywh_format, jaccard, xywh_to_xyxy_format

from yolonas.config import CONFIG
from yolonas.utils.general_utils import reshape_output_list, calculate_iou_batch, pad_bboxes_to_same_length, \
    calculate_iou_all_pairs
from yolonas.utils.yolo_utils import decoder, LOSS_FN


def compute_losses(obj_true: tf.Tensor, reg: tf.Tensor, cls: tf.Tensor) -> Union[
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]],
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]]:
    """
    Computes the sum of the classification (CE loss) and localization (regression) losses from all heads
    """
    decoded = False if CONFIG["MODEL_FORMAT"] != "inference" else True
    dummy = tf.zeros(shape=(cls.shape[0], cls.shape[1], 1))
    cls = tf.concat([cls, dummy], axis=2)
    reg = xyxy_to_xywh_format(reg)
    od_pred = tf.concat([reg, cls], axis=2)
    class_list_reshaped, loc_list_reshaped = reshape_output_list(od_pred, decoded=decoded,
                                                                 image_size=CONFIG["IMAGE_SIZE"])  # add batch
    for i in range(len(class_list_reshaped)):
        duplicated_tensors = [class_list_reshaped[i]] * 3
        # Concatenate along the second axis (axis=1)
        class_list_reshaped[i] = tf.concat(duplicated_tensors, axis=1)

    for i in range(len(loc_list_reshaped)):
        duplicated_tensors = [loc_list_reshaped[i]] * 3
        # Concatenate along the second axis (axis=1)
        loc_list_reshaped[i] = tf.concat(duplicated_tensors, axis=1)

    loss_l, loss_c, loss_o = LOSS_FN(y_true=obj_true, y_pred=(loc_list_reshaped, class_list_reshaped))
    return loss_l, loss_c, loss_o


def od_loss(bb_gt: tf.Tensor, reg: tf.Tensor, cls: tf.Tensor) -> tf.Tensor:  # return batch
    """
    Sums the classification and regression loss
    """
    loss_l, loss_c, loss_o = compute_losses(bb_gt, reg, cls)
    combined_losses = [l + c + o for l, c, o in zip(loss_l, loss_c, loss_o)]
    sum_loss = tf.reduce_sum(combined_losses, axis=0)
    non_nan_loss = tf.where(tf.math.is_nan(sum_loss), tf.zeros_like(sum_loss), sum_loss)  # LOSS 0 for NAN losses
    return non_nan_loss


def general_metrics_dict(bb_gt: tf.Tensor, reg: tf.Tensor, cls: tf.Tensor) -> Dict[str, tf.Tensor]:
    try:
        reg_met, class_met, obj_met = compute_losses(bb_gt, reg, cls)
    except Exception as e:
        print(e)
        batch_dim = bb_gt.shape[0]
        fault_res_tensor = [tf.convert_to_tensor(-np.ones((batch_dim, 1))) for _ in range(3)]
        reg_met, class_met, obj_met = (fault_res_tensor, fault_res_tensor, fault_res_tensor, fault_res_tensor)
    res = {
        "Regression_metric": tf.reduce_sum(reg_met, axis=0)[:, 0],
        "Classification_metric": tf.reduce_sum(class_met, axis=0)[:, 0],
        "Objectness_metric": tf.reduce_sum(obj_met, axis=0)[:, 0],
    }
    return res


def iou_metrics_dict(bb_gt: tf.Tensor, reg: tf.Tensor, cls: tf.Tensor) -> Dict[str, tf.Tensor]:
    id_to_name = CONFIG['class_id_to_name']
    threshold = CONFIG['CM_IOU_THRESH']
    reg_fixed = xyxy_to_xywh_format(reg) / CONFIG['IMAGE_SIZE'][0]
    outputs = decoder(loc_data=[reg_fixed], conf_data=[cls], prior_data=[None],
                      from_logits=False, decoded=True)
    batch_res = {f"{name}_mean_iou": [] for name in id_to_name.values()}
    batch_res['mean_all_iou'] = []
    for batch_i in range(len(outputs)):
        sample_res = {name: [] for name in id_to_name.values()}
        if len(outputs[batch_i]) != 0:
            ious = jaccard(outputs[batch_i][:, 1:5],
                           xywh_to_xyxy_format(
                               tf.cast(bb_gt[batch_i, :, :-1], tf.double))).numpy()  # (#bb_predicted,#gt)
            prediction_detected = np.any((ious > threshold), axis=1)
            max_iou_ind = np.argmax(ious, axis=1)
            max_iou = np.max(ious, axis=1)
            for i, prediction in enumerate(prediction_detected):
                gt_idx = int(bb_gt[batch_i, max_iou_ind[i], 4])
                class_name = id_to_name.get(gt_idx)

                if prediction:
                    sample_res[class_name].append(max_iou[i])
        for class_name in sample_res.keys():
            if np.isnan(np.mean(sample_res[class_name])):
                batch_res[f"{class_name}_mean_iou"].append(0)
            else:
                batch_res[f"{class_name}_mean_iou"].append(np.mean(sample_res[class_name]))
        all_ious = [np.mean(sample_res[class_name]) for class_name in sample_res.keys() if
                    len(sample_res[class_name]) > 0]
        if len(all_ious) > 0:
            batch_res['mean_all_iou'].append(
                np.mean([np.mean(sample_res[class_name]) for class_name in sample_res.keys() if
                         len(sample_res[class_name]) > 0]))
        else:
            batch_res['mean_all_iou'].append(0.0)
    return {k: tf.convert_to_tensor(v) for k, v in batch_res.items()}


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
