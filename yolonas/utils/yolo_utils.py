import torch
import numpy as np
import tensorflow as tf
from PIL import Image

from code_loader.helpers.detection.utils import xyxy_to_xywh_format

from yolonas.utils.decoder import Decoder

decoder = Decoder(num_classes=80,
                  background_label=81,
                  top_k=300,
                  conf_thresh=0.25,
                  nms_thresh=0.7,
                  max_bb_per_layer=1000,
                  max_bb=1000,
                  semantic_instance=False,
                  class_agnostic_nms=True,
                  has_object_logit=False)


# def y_pred_from_model_outputs(reg: tf.Tensor, cls: tf.Tensor):
#
#     return y_pred
