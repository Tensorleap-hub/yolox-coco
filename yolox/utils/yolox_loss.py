import numpy as np
import tensorflow as tf
from code_loader.helpers.detection.utils import xywh_to_xyxy_format
from code_loader.helpers.detection.utils import jaccard
from yolox.config import CONFIG

N_SAMPLES = 1100
N_TRAIN = 10
N_TEST = N_SAMPLES - N_TRAIN

PROJECT_ID = 'hp-dev-project'
BUCKET_NAME = 'hp-datasets'
DIR_NAME = "Test_MegaDB_BarakHS_HI4AI"
image_name_col = "fname"
LABELS = ['0', '11', '8', '1', '3', '9']


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

    decoded_outputs = tf.concat([
        (pred[..., 0:2] + grids) * strides,  # x, y
        tf.exp(pred[..., 2:4]) * strides,  # w, h
        pred[..., 4:]  # conf + classes
    ], axis=-1)

    return decoded_outputs


def simple_od_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    conf_loss, class_loss, reg_loss_list = get_od_losses(y_true, y_pred)
    return conf_loss + class_loss + reg_loss_list


def get_od_losses(y_true: tf.Tensor, y_pred: tf.Tensor):
    decoded_preds = decode_outputs(y_pred)  # in absolute units
    bboxes = decoded_preds[:, :, :4]
    bboxes /= [*CONFIG['IMAGE_SIZE'][::-1], *CONFIG['IMAGE_SIZE'][::-1]]
    batch = y_true.shape[0]
    bbs_count = bboxes.shape[1]
    reg_loss_list = [tf.constant(0, dtype=tf.float32)]*batch
    class_loss_list = []
    conf_loss_list = []
    for i in range(batch):
        confidence_gt = tf.zeros(bbs_count)
        classes_gt = tf.zeros((bbs_count, CONFIG['CLASSES'] - 1), tf.float32)
        classes_gt = tf.concat([tf.ones((bbs_count, 1), tf.float32), classes_gt], axis=-1)

        y_true_real = y_true[i][y_true[i][..., -1] != CONFIG['BACKGROUND_LABEL']]
        xyxy_gt = xywh_to_xyxy_format(y_true_real[...,:-1])
        xyxy_pred = xywh_to_xyxy_format(bboxes[i, ...])
        ious = jaccard(xyxy_pred, xyxy_gt) #BB, #GT
        max_iou = tf.reduce_max(ious, -1)
        all_gt_idx = tf.argmax(ious, 1)
        matched_bbs_idx = tf.where(max_iou > CONFIG['LOSS_IOU_TH'])[..., 0]
        if matched_bbs_idx.shape[0] > 0:
            positive_gt_idx = tf.gather(all_gt_idx, matched_bbs_idx)
            matched_gts = tf.gather(xyxy_gt, positive_gt_idx)
            matched_bbs = tf.gather(xyxy_pred, matched_bbs_idx)
            reg_loss_list[i] = tf.keras.losses.MeanSquaredError()(matched_bbs, matched_gts)
            matched_classes = tf.one_hot(tf.cast(matched_gts[...,-1], tf.int32), 80)
        #compute reg loss

            positive_ious = tf.gather(max_iou, matched_bbs_idx)
            confidence_gt = tf.tensor_scatter_nd_update(
                confidence_gt, matched_bbs_idx[..., None], positive_ious
            )
            classes_gt = tf.tensor_scatter_nd_update(
                classes_gt, matched_bbs_idx[..., None], matched_classes
            )
        class_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(classes_gt, decoded_preds[i, ..., 5:])
        confidence_loss = tf.keras.losses.MeanSquaredError()(confidence_gt, decoded_preds[i, ..., 4])
        conf_loss_list.append(confidence_loss)
        class_loss_list.append(class_loss)
    conf_loss = tf.stack(conf_loss_list, axis=0)
    class_loss = tf.stack(class_loss_list, axis=0)
    reg_loss_list = tf.stack(reg_loss_list, axis=0)
    return conf_loss, class_loss, reg_loss_list
