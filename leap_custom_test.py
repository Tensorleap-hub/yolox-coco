import tensorflow as tf
from leap_binder import (
    subset_images, input_image, get_bbs, metadata_dict, unlabeled_preprocessing_func
)
from yolox.config import CONFIG
from yolox.metrics import od_metrics_dict
from yolox.utils.confusion_matrix import confusion_matrix_metric
from yolox.utils.general_utils import draw_image_with_boxes
from yolox.visualizers import gt_bb_decoder, pred_bb_visualizer
from yolox.utils.yolox_loss import custom_yolox_loss


def check_integration():
    model_path = 'model/yolox_s.h5'
    model = tf.keras.models.load_model(model_path)
    batch = 1
    responses = subset_images()  # get dataset splits
    training_response = responses[1]
    unlabeled_response = unlabeled_preprocessing_func()
    unlabeled_image = input_image(0, unlabeled_response)
    unlabeled_metadata = metadata_dict(0, unlabeled_response)
    images = []
    bbs_gt = []
    for idx in range(batch):
        print(idx)
        image = input_image(idx, training_response)
        bb_gt = get_bbs(idx, training_response)
        images.append(image)
        bbs_gt.append(bb_gt)

    metadata = metadata_dict(0, training_response)
    y_true_bbs = tf.convert_to_tensor(bbs_gt)  # convert ground truth bbs to tensor

    input_img_tf = tf.convert_to_tensor(images, dtype=tf.float32)
    y_pred = model([input_img_tf])  # infer and get model prediction
    dummy_y = tf.random.uniform((batch, 10, y_pred.shape[-1]), 0, 1)
    y_pred = tf.concat([y_pred, dummy_y], 1)
    loss = custom_yolox_loss(y_true_bbs, y_pred)
    od_metrics = od_metrics_dict(y_true_bbs, y_pred)

    conf_mat = confusion_matrix_metric(y_true_bbs, y_pred)

    for i in range(batch):
        pred_bb_vis = pred_bb_visualizer(images[i], y_pred[i, ...].numpy())
        draw_image_with_boxes(pred_bb_vis.data, pred_bb_vis.bounding_boxes)
    for i in range(batch):
        gt_bb_vis = gt_bb_decoder(images[i], bbs_gt[i])
        draw_image_with_boxes(gt_bb_vis.data, gt_bb_vis.bounding_boxes)


if __name__ == '__main__':
    check_integration()
