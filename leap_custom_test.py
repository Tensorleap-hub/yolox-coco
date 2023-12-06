import tensorflow as tf
from leap_binder import (
    subset_images, input_image, get_bbs, metadata_dict, unlabeled_preprocessing_func
)
from yolox.metrics import od_metrics_dict
from yolox.utils.general_utils import draw_image_with_boxes
from yolox.visualizers import gt_bb_decoder, pred_bb_visualizer
from yolox.utils.yolox_loss import simple_od_loss


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

    # metadata = metadata_dict(idx, training_response)
    y_true_bbs = tf.convert_to_tensor(bbs_gt)  # convert ground truth bbs to tensor

    input_img_tf = tf.convert_to_tensor(images, dtype=tf.float32)
    y_pred = model([input_img_tf])  # infer and get model prediction
    # output = tf.transpose(output, (0, 2, 1))
    s_loss = simple_od_loss(y_true_bbs, y_pred)
    # loss = custom_yolox_loss(y_true_bbs, y_pred)
    od_metrics = od_metrics_dict(y_true_bbs, y_pred)

    # iou_metrics = iou_metrics_dict(y_true_bbs, reg, cls)
    # conf_mat = confusion_matrix_metric(y_true_bbs, cls, reg, input_img_tf)

    pred_bb_vis = pred_bb_visualizer(images[2], y_pred[2, ...].numpy())
    draw_image_with_boxes(pred_bb_vis.data / 255., pred_bb_vis.bounding_boxes)
    gt_bb_vis = gt_bb_decoder(images[0], bbs_gt[0])
    draw_image_with_boxes(gt_bb_vis.data / 255., gt_bb_vis.bounding_boxes)


if __name__ == '__main__':
    check_integration()
