import tensorflow as tf
from leap_binder import (
    subset_images, input_image, get_bbs, confusion_matrix_metric, metadata_dict, unlabeled_preprocessing_func
)
from yolonas.metrics import od_loss, iou_metrics_dict
from yolonas.utils.general_utils import draw_image_with_boxes
from yolonas.visualizers import pred_bb_decoder, gt_bb_decoder


def check_integration():
    model_path = 'model/yolo_nas_s_permuted_output.h5'
    model = tf.keras.models.load_model(model_path)
    batch = 4
    responses = subset_images()  # get dataset splits
    training_response = responses[0]
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

        input_img_tf = tf.convert_to_tensor(images)
        reg, cls = model([input_img_tf])  # infer and get model prediction
        # loss = od_loss(y_true_bbs, reg, cls)
        iou_metrics = iou_metrics_dict(y_true_bbs, reg, cls)
        conf_mat = confusion_matrix_metric(y_true_bbs, cls, reg, input_img_tf)

        # pred_bb_vis = pred_bb_decoder(image, reg[0, ...], cls[0, ...])
        # draw_image_with_boxes(pred_bb_vis.data / 255., pred_bb_vis.bounding_boxes)
        # gt_bb_vis = gt_bb_decoder(image, bb_gt)
        # draw_image_with_boxes(gt_bb_vis.data / 255., gt_bb_vis.bounding_boxes)


if __name__ == '__main__':
    check_integration()
