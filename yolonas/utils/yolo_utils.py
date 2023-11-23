from yolonas.config import CONFIG
from yolonas.utils.decoder import Decoder
import numpy as np
from code_loader.helpers.detection.yolo.loss import YoloLoss
from code_loader.helpers.detection.yolo.grid import Grid

BOXES_GENERATOR = Grid(image_size=CONFIG['IMAGE_SIZE'],
                       feature_maps=CONFIG['FEATURE_MAPS'],
                       box_sizes=CONFIG['BOX_SIZES'],
                       strides=CONFIG['STRIDES'],
                       offset=CONFIG['OFFSET'])
DEFAULT_BOXES = BOXES_GENERATOR.generate_anchors()

LOSS_FN = YoloLoss(num_classes=CONFIG['CLASSES'],
                   overlap_thresh=CONFIG['OVERLAP_THRESH'],
                   features=CONFIG['FEATURE_MAPS'],
                   anchors=np.array(CONFIG['BOX_SIZES']),
                   default_boxes=DEFAULT_BOXES,
                   background_label=CONFIG['BACKGROUND_LABEL'],
                   from_logits=False if CONFIG['MODEL_FORMAT'] == "inference" else True,
                   image_size=CONFIG['IMAGE_SIZE'],
                   yolo_match=True,
                   semantic_instance=False)

decoder = Decoder(num_classes=CONFIG['CLASSES'],
                  background_label=CONFIG['BACKGROUND_LABEL'],
                  top_k=CONFIG['TOP_K'],
                  conf_thresh=CONFIG['CONF_THRESH'],
                  nms_thresh=CONFIG['NMS_THRESH'],
                  max_bb_per_layer=1000,
                  max_bb=1000,
                  semantic_instance=False,
                  class_agnostic_nms=True,
                  has_object_logit=False)

DECODER = Decoder(CONFIG['CLASSES'],
                  background_label=CONFIG['BACKGROUND_LABEL'],
                  top_k=50,
                  conf_thresh=CONFIG['CONF_THRESH'],
                  nms_thresh=CONFIG['NMS_THRESH'],
                  semantic_instance=False,
                  max_bb=50,
                  max_bb_per_layer=50)
