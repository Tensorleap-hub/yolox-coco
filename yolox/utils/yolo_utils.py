import tensorflow as tf

from yolox.config import CONFIG


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
