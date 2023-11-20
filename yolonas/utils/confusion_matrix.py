from code_loader.helpers.detection.utils import xyxy_to_xywh_format
from code_loader.helpers.detection.yolo.utils import jaccard, xywh_to_xyxy_format
from yolonas.utils.yolo_utils import decoder
from code_loader.contract.datasetclasses import ConfusionMatrixElement
from code_loader.contract.enums import ConfusionMatrixValue
import tensorflow as tf
import numpy as np
from yolonas.config import CONFIG


def confusion_matrix_metric(gt, cls, reg, image):
    #assumes we get predictions in xyxy format in gt AND reg
    #assumes gt is in xywh form
    threshold = 0.5
    reg_fixed = xyxy_to_xywh_format(reg) / image.shape[1]
    outputs = decoder(loc_data=[reg_fixed], conf_data=[cls], prior_data=[None],
                      from_logits=False, decoded=True)
    ret = []
    for batch_i in range(len(outputs)):
        confusion_matrix_elements = []
        if len(outputs[batch_i]) != 0:
            ious = jaccard(outputs[batch_i][:, 1:5],
                        xywh_to_xyxy_format(tf.cast(gt[batch_i, :, :-1], tf.double))).numpy()  # (#bb_predicted,#gt)
            prediction_detected = np.any((ious > threshold), axis=1)
            max_iou_ind = np.argmax(ious, axis=1)
            for i, prediction in enumerate(prediction_detected):
                gt_idx = int(gt[batch_i, max_iou_ind[i], 4])
                gt_label = f"class_{gt_idx}"
                confidence = outputs[batch_i][i, 0]
                if prediction: #TP
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(gt_label),
                        ConfusionMatrixValue.Positive,
                        float(confidence)
                    ))
                else: #FP
                    pred_label = f"class_{int(outputs[batch_i][i, -1])}"
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(pred_label),
                        ConfusionMatrixValue.Negative,
                        float(confidence)
                    ))
        else: #No prediction
            ious = np.zeros((1, gt[batch_i, ...].shape[0]))
        gts_detected = np.any((ious > threshold), axis=0)
        for k, gt_detection in enumerate(gts_detected):
            label_idx = gt[batch_i, k, -1]
            if not gt_detection and label_idx != CONFIG['BACKGROUND_LABEL']: #FN
                confusion_matrix_elements.append(ConfusionMatrixElement(
                    f"class_{int(gt[batch_i, k, -2])}",
                    ConfusionMatrixValue.Positive,
                    float(0)
                ))
        ret.append(confusion_matrix_elements)
    return ret
