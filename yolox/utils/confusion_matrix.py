from code_loader.helpers.detection.yolo.utils import jaccard, xywh_to_xyxy_format

from code_loader.contract.datasetclasses import ConfusionMatrixElement
from code_loader.contract.enums import ConfusionMatrixValue
import numpy as np
from yolox.config import CONFIG
from yolox.utils.yolo_utils import nms, decode_outputs


def confusion_matrix_metric(gt, y_pred):
    # assumes we get predictions in xyxy format in gt AND reg
    # assumes gt is in xywh form
    id_to_name = CONFIG['class_id_to_name']
    threshold = CONFIG['IOU_TH']
    outputs = decode_outputs(y_pred).numpy()
    ret = []
    for batch_i in range(len(outputs)):
        above_conf_indices = outputs[batch_i, :, 4] > CONFIG['CONF_THRESH']
        batch_outputs = outputs[batch_i, above_conf_indices, :]
        nms_selected = nms(batch_outputs[:, :5], is_xyxy=False).numpy()
        batch_outputs = batch_outputs[nms_selected, :]
        confusion_matrix_elements = []
        if len(batch_outputs) != 0:
            ious = jaccard(xywh_to_xyxy_format(batch_outputs[:, :4]) / CONFIG['IMAGE_SIZE'][0],
                           xywh_to_xyxy_format(gt[batch_i, :, :-1])).numpy()  # (#bb_predicted,#gt)
            prediction_detected = np.any((ious > threshold), axis=1)
            max_iou_ind = np.argmax(ious, axis=1)
            for i, prediction in enumerate(prediction_detected):
                gt_idx = int(gt[batch_i, max_iou_ind[i], 4])
                class_name = id_to_name.get(gt_idx)
                gt_label = f"{class_name}"
                confidence = batch_outputs[i, 4]
                if prediction:  # TP
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(gt_label),
                        ConfusionMatrixValue.Positive,
                        float(confidence)
                    ))
                else:  # FP
                    class_name = id_to_name.get(int(batch_outputs[i, -1]))
                    pred_label = f"{class_name}"
                    confusion_matrix_elements.append(ConfusionMatrixElement(
                        str(pred_label),
                        ConfusionMatrixValue.Negative,
                        float(confidence)
                    ))
        else:  # No prediction
            ious = np.zeros((1, gt[batch_i, ...].shape[0]))
        gts_detected = np.any((ious > threshold), axis=0)
        for k, gt_detection in enumerate(gts_detected):
            label_idx = gt[batch_i, k, -1]
            if not gt_detection and label_idx != CONFIG['BACKGROUND_LABEL']:  # FN
                class_name = id_to_name.get(int(gt[batch_i, k, -2]))
                confusion_matrix_elements.append(ConfusionMatrixElement(
                    f"{class_name}",
                    ConfusionMatrixValue.Positive,
                    float(0)
                ))
        if all(~ gts_detected):
            confusion_matrix_elements.append(ConfusionMatrixElement(
                "background",
                ConfusionMatrixValue.Positive,
                float(0)
            ))
        ret.append(confusion_matrix_elements)
    return ret
