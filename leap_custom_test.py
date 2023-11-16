import tensorflow as tf
from code_loader.helpers.detection.utils import xyxy_to_xywh_format

from leap_binder import (
    subset_images, input_image, get_bbs
)
from yolonas.metrics import custom_yolo_nas_loss
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

        loss = custom_yolo_nas_loss(y_true=y_true_bbs, reg=reg, cls=cls)

        pred_bb_vis = pred_bb_decoder(images[0], reg[0], cls[0])
        plt.imshow(pred_bb_vis.data / 255.)
        gt_bb_vis = gt_bb_decoder(images[0], bb_gt[0])
        plt.imshow(gt_bb_vis.data / 255.)


if __name__ == '__main__':
    check_integration()
