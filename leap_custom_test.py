import tensorflow as tf
import numpy as np
from code_loader.helpers.detection.utils import xyxy_to_xywh_format

from leap_binder import (
    subset_images, input_image, get_bbs, confusion_matrix_metric
)
from yolonas.config import CONFIG
from yolonas.custom_layers import MockOneClass
from yolonas.metrics import custom_yolo_nas_loss, huber_metric
from yolonas.utils.general_utils import draw_image_with_boxes
from yolonas.visualizers import pred_bb_decoder, gt_bb_decoder
import matplotlib.pyplot as plt

def check_integration():
    model_path = 'model/yolo_nas_s.h5'
    model = tf.keras.models.load_model(model_path)
    batch = 8
    responses = subset_images()  # get dataset splits
    training_response = responses[0]  # [training, validation, test]

    for idx in range(batch):
        images = []
        bb_gt = []
        images.append(input_image(idx, training_response))
        bb_gt.append(get_bbs(idx, training_response))
        y_true_bbs = tf.convert_to_tensor(bb_gt)  # convert ground truth bbs to tensor

        input_img_tf = tf.convert_to_tensor(images)
        reg, cls = model([input_img_tf])  # infer and get model prediction

        # mock one class
        mock_layer = MockOneClass()
        cls = mock_layer(cls)
        loss = custom_yolo_nas_loss(y_true=y_true_bbs, reg=reg, cls=cls)
        metric = huber_metric(y_true=y_true_bbs, reg=reg, cls=cls)
        pred_bb_vis = pred_bb_decoder(images[0], reg[0], cls[0])
        draw_image_with_boxes(pred_bb_vis.data / 255., pred_bb_vis.bounding_boxes)
        gt_bb_vis = gt_bb_decoder(input_image(idx, training_response, False), bb_gt[0])
        conf_mat = confusion_matrix_metric(y_true_bbs, cls, reg, input_img_tf)
        # pred_bb_vis = pred_bb_decoder(images[0], reg[0], cls[0])
        # draw_image_with_boxes(pred_bb_vis.data / 255., pred_bb_vis.bounding_boxes)
        gt_bb_vis = gt_bb_decoder(images[0], bb_gt[0])
        draw_image_with_boxes(gt_bb_vis.data / 255., gt_bb_vis.bounding_boxes)


if __name__ == '__main__':
    check_integration()
