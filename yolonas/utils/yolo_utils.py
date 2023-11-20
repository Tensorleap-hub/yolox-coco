from yolonas.config import CONFIG
from yolonas.utils.decoder import Decoder

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


