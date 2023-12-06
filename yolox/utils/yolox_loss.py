import tensorflow as tf
from code_loader.helpers.detection.utils import xywh_to_xyxy_format

from yolox.config import CONFIG

N_SAMPLES = 1100
N_TRAIN = 10
N_TEST = N_SAMPLES - N_TRAIN

PROJECT_ID = 'hp-dev-project'
BUCKET_NAME = 'hp-datasets'
DIR_NAME = "Test_MegaDB_BarakHS_HI4AI"
image_name_col = "fname"
LABELS = ['0', '11', '8', '1', '3', '9']


def calculate_iou(grid_size: int, bbox_gt: tf.Tensor) -> tf.Tensor:
    factor = 1 / grid_size
    grid_map = tf.range(0, 1, factor)

    xx, yy = tf.meshgrid(grid_map, grid_map, indexing='ij')
    grid_matrix_map_start = tf.stack([tf.reshape(xx, [-1]), tf.reshape(yy, [-1])], axis=1)
    grid_matrix_map_end = grid_matrix_map_start + factor

    box_gt_start = bbox_gt[:2]
    box_gt_end = bbox_gt[2:]

    max_start = tf.math.maximum(box_gt_start, grid_matrix_map_start)
    min_end = tf.math.minimum(box_gt_end, grid_matrix_map_end)

    interArea = tf.reduce_prod(tf.math.maximum(min_end - max_start, [0, 0]), axis=1)
    boxAArea = tf.reduce_prod(box_gt_end - box_gt_start)
    boxBArea = tf.reduce_prod(grid_matrix_map_end - grid_matrix_map_start, axis=1)

    iou = interArea / (boxAArea + boxBArea - interArea + tf.keras.backend.epsilon())

    return iou


def gather_positive_examples(pred_grids: tf.Tensor, bbox_gt: tf.Tensor):
    iou = calculate_iou(pred_grids.shape[-1], bbox_gt)

    iou_indices = tf.where(iou > 0.08)
    x_positive_indices = tf.math.floordiv(iou_indices, pred_grids.shape[-1])
    y_positive_indices = tf.math.floormod(iou_indices, pred_grids.shape[-1])

    pred_flat = tf.reshape(pred_grids, (CONFIG['CLASSES'] + 5, -1))

    confidence_flat_value = tf.reshape(tf.gather(iou, iou_indices), -1)
    positive_flat_indices = tf.reshape(y_positive_indices * pred_grids.shape[-1] + x_positive_indices, -1)

    positive_examples = tf.gather(pred_flat, positive_flat_indices, axis=-1)

    return positive_examples, positive_flat_indices, confidence_flat_value


def run_single_scale(pred_scale_reshaped: tf.Tensor, y_true):
    n_grids = pred_scale_reshaped.shape[-1] * pred_scale_reshaped.shape[-1]

    true_classes_per_grid = tf.zeros((n_grids, CONFIG['CLASSES'] - 1), tf.float32)
    true_classes_per_grid = tf.concat([tf.ones((n_grids, 1), tf.float32), true_classes_per_grid], axis=-1)
    reg_loss_all_boxes = tf.zeros((0, 1))
    true_confidences = tf.zeros((n_grids, 1), tf.float32)
    for single_gt in y_true[0, :]:
        bbox_gt = single_gt[:4]

        positive_examples, positive_flat_indices, confidence_flat_values = gather_positive_examples(
            pred_scale_reshaped, bbox_gt)

        class_index = tf.cast(single_gt[4], tf.int64)
        positive_flat_indices = tf.expand_dims(positive_flat_indices, axis=0)
        update_indices_class = tf.transpose(
            tf.concat([positive_flat_indices, tf.ones(positive_flat_indices.shape, tf.int64) * class_index], axis=0))
        update_indices_default = tf.transpose(
            tf.concat([positive_flat_indices, tf.zeros(positive_flat_indices.shape, tf.int64) * class_index], axis=0))
        update_indices = tf.stack([update_indices_class, update_indices_default], axis=1)

        if update_indices.shape[0] == 0:
            continue

        true_classes_per_grid = tf.tensor_scatter_nd_update(
            true_classes_per_grid, update_indices, [[1.0, 0.0]] * update_indices.shape[0]
        )

        true_confidences = tf.tensor_scatter_nd_update(
            true_confidences, update_indices_default, confidence_flat_values
        )

        reg_loss = tf.keras.losses.MeanSquaredError()(tf.transpose(positive_examples)[:, :4] / CONFIG['IMAGE_SIZE'][0],
                                                      bbox_gt)
        reg_loss_all_boxes = tf.concat([reg_loss_all_boxes, tf.reshape(reg_loss, (tf.size(reg_loss), 1))], axis=0)

    return true_classes_per_grid, reg_loss_all_boxes, true_confidences


def simple_od_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    pred_scale_80_80 = y_pred[:, :, :6400]
    pred_scale_40_40 = y_pred[:, :, 6400:8000]
    pred_scale_20_20 = y_pred[:, :, 8000:]
    pred_scale_80_80 = tf.reshape(pred_scale_80_80, [1, CONFIG['CLASSES'] + 5, 80, 80])
    pred_scale_40_40 = tf.reshape(pred_scale_40_40, [1, CONFIG['CLASSES'] + 5, 40, 40])
    pred_scale_20_20 = tf.reshape(pred_scale_20_20, [1, CONFIG['CLASSES'] + 5, 20, 20])

    y_true = tf.cast(y_true, tf.float32)
    y_true_bboxes = xywh_to_xyxy_format(y_true[:, :, :4])
    y_true = tf.concat([y_true_bboxes, y_true[..., -2:-1]], axis=-1)

    true_classes_80_80, reg_loss_80_80, true_confidences_80_80 = run_single_scale(pred_scale_80_80, y_true)
    true_classes_40_40, reg_loss_40_40, true_confidences_40_40 = run_single_scale(pred_scale_40_40, y_true)
    true_classes_20_20, reg_loss_20_20, true_confidences_20_20 = run_single_scale(pred_scale_20_20, y_true)

    true_classes_all = tf.concat([true_classes_80_80, true_classes_40_40, true_classes_20_20], axis=0)
    true_confidences_all = tf.concat([true_confidences_80_80, true_confidences_40_40, true_confidences_20_20], axis=0)
    reg_loss_all = tf.concat([reg_loss_80_80, reg_loss_40_40, reg_loss_20_20], axis=0)
    predicted_classes_all = tf.transpose(y_pred[0, 5:, :])
    classes_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(true_classes_all, predicted_classes_all)

    mean_reg_bbox_loss = tf.reduce_mean(reg_loss_all)
    mean_reg_bbox_loss = tf.cond(tf.math.is_nan(mean_reg_bbox_loss), lambda: 0.0, lambda: mean_reg_bbox_loss)

    predicted_conf_all = tf.expand_dims(tf.transpose(y_pred[0, 4, :]), axis=-1)
    reg_conf_loss = tf.keras.losses.MeanSquaredError()(true_confidences_all, predicted_conf_all)

    return mean_reg_bbox_loss + classes_loss + reg_conf_loss
