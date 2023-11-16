import torch
import numpy as np
import tensorflow as tf
from PIL import Image

from code_loader.helpers.detection.utils import xyxy_to_xywh_format

from yolonas.config import CONFIG
from yolonas.utils.decoder import Decoder

decoder = Decoder(num_classes=CONFIG['CLASSES'],
                  background_label=CONFIG['BACKGROUND_LABEL'],
                  top_k=300,
                  conf_thresh=0.25,
                  nms_thresh=0.7,
                  max_bb_per_layer=1000,
                  max_bb=1000,
                  semantic_instance=False,
                  class_agnostic_nms=True,
                  has_object_logit=False)


