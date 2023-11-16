from typing import List

import numpy as np
import tensorflow as tf  # type: ignore
from numpy.typing import NDArray

from code_loader.helpers.detection.utils import xywh_to_xyxy_format
from code_loader.helpers.detection.yolo.utils import decode_bboxes


class Decoder:
    """At test time, Detect is the final layer of SSD.
    Consists of 4 steps:
    Bounding Boxes Decoding,
    Confidence Thresholding,
    Non-Max Suppression,
    Top-K Filtering.
    """

    def __init__(self, num_classes: int, background_label: int, top_k: int, conf_thresh: float, nms_thresh: float,
                 max_bb_per_layer: int = 20, max_bb: int = 20, semantic_instance: bool = False,
                 class_agnostic_nms: bool = False, has_object_logit=True):
        if not has_object_logit and semantic_instance:
            raise NotImplementedError("Semantic Instance Segmentation without an object logit needs not implemented")
        self.num_classes = num_classes
        self.background_label = background_label
        self.top_k = top_k
        self.max_bb_per_layer = max_bb_per_layer
        self.max_bb = max_bb
        # Parameters used in nms.
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.semantic_instance = semantic_instance
        self.class_agnostic_nms = class_agnostic_nms
        self.has_object_logit = has_object_logit

    def __call__(self, loc_data: List[tf.Tensor], conf_data: List[tf.Tensor], prior_data: List[NDArray[np.float32]],
                 from_logits: bool = True, decoded: bool = False) -> List[NDArray[np.float32]]:
        """
        Args:
            loc_data: (tensor) Location preds from loc layers
                Shape: [batch, num_priors*4]
            conf_data: (tensor) Shape: Confidence preds from confidence layers
                Shape: [batch, num_priors, num_classes]
            prior_data: (tensor) Prior boxes and variances? from priorbox layers
                Shape: [1, num_priors, 4]
        """
        # loc_data: (batch_size, num_priors, 4)
        # conf_data: (batch_size, num_priors, num_classes)
        if not self.semantic_instance:
            classes_num = conf_data[0].shape[-1]
            if self.has_object_logit:
                classes_num -= 1
        else:
            classes_num = self.num_classes
        conf_preds = [tf.transpose(a=layer_conf, perm=[0, 2, 1]) for layer_conf in
                      conf_data]  # (batch_size, num_classes, num_priors)
        outputs = []
        for i in range(tf.shape(loc_data[0])[0]):
            loc = [loc_e[i, ...] for loc_e in loc_data]
            conf = [conf_e[i, :self.num_classes + 1, :] for conf_e in conf_preds]
            if from_logits:
                conf = [tf.math.sigmoid(layer_conf) for layer_conf in conf]
            class_selections: List[List[tf.Tensor]] = [[] for j in range(classes_num)]
            for m, (l_loc, l_conf, l_prior) in enumerate(zip(loc, conf, prior_data)):
                l_conf = l_conf.numpy()
                if self.has_object_logit:
                    mask_confidence = l_conf[0, ...]
                    if classes_num > 1:
                        l_conf = l_conf[1:, ...] * mask_confidence
                    else:
                        l_conf = mask_confidence[np.newaxis, ...]
                else:
                    mask_confidence = np.max(l_conf, axis=0)
                mask = mask_confidence > self.conf_thresh
                classes = np.argmax(l_conf, axis=0)
                max_scores = np.max(l_conf, axis=0)
                # mask = max_scores > self.conf_thresh
                non_zero_indices = np.where(mask)[0]
                if len(non_zero_indices) != 0:
                    scores_masked = max_scores[mask]
                    if len(scores_masked) > self.max_bb_per_layer:
                        best_scores, best_indices = tf.math.top_k(scores_masked, k=self.max_bb_per_layer)
                    else:
                        best_scores = scores_masked
                        best_indices = np.arange(len(scores_masked))
                    topk_indices = np.take(non_zero_indices, best_indices)
                    selected_loc = tf.gather(l_loc, topk_indices, 1).numpy()
                    selected_scores = best_scores
                    if self.semantic_instance:
                        selected_mask_indices = tf.gather(conf_data[m][0, :, self.num_classes + 1:],
                                                          topk_indices, 1).numpy()
                    if not decoded:
                        selected_prior = tf.gather(l_prior, topk_indices, 1).numpy()
                        selected_decoded = decode_bboxes(selected_loc,
                                                         selected_prior)  # (num_priors, 4)  (xmin, ymin, xmax, ymax) - THIS WORKS
                    else:
                        selected_decoded = xywh_to_xyxy_format(selected_loc)
                    selected_classes = tf.gather(classes, topk_indices).numpy()
                    for k in range(len(selected_classes)):
                        if not self.semantic_instance:
                            class_selections[selected_classes[k]].append(
                                (selected_scores[k], *selected_decoded[k, :], selected_classes[k]))
                        else:
                            class_selections[selected_classes[k]].append(
                                (selected_scores[k], *selected_decoded[k, :], selected_classes[k],
                                 *selected_mask_indices[k, :]))
            final_preds = []
            if not self.class_agnostic_nms:
                for j in range(classes_num):
                    if len(class_selections[j]) > 0:
                        np_selection: NDArray[np.float32] = np.array(class_selections[j])
                        selected_indices = self.nms(np_selection)
                        final_preds.append(np_selection[selected_indices, :])
                if any(class_selections):
                    # choose best MAX_RETURNS/#detected class
                    chosen_list = []
                    remainder_list = []
                    max_per_class = max(int(self.max_bb / len(final_preds)), 1)
                    for l in range(len(final_preds)):
                        chosen_list.append(final_preds[l][:max_per_class, ...])
                        remains = final_preds[l][max_per_class:, ...]
                        if remains.shape[0] > 0:
                            remainder_list.append(remains)
                    predictions: NDArray[np.float32] = np.concatenate(chosen_list, axis=0)
                    predictions = predictions[:self.max_bb, ...]  # this selects top 20 objects
                    matched_obj = predictions.shape[0]
                    missing_amount = self.max_bb - matched_obj
                    if len(remainder_list) > 0 and predictions.shape[0] < self.max_bb:
                        remainder_np: NDArray[np.float32] = np.concatenate(remainder_list, axis=0)
                        if remainder_np.shape[0] > missing_amount:
                            top_indices = np.argpartition(remainder_np[:, 0], -missing_amount)[-missing_amount:]
                            remainder_np = remainder_np[top_indices, ...]
                        predictions = np.concatenate([predictions, remainder_np], axis=0)
                    outputs.append(predictions)
                else:
                    outputs.append(np.empty(0, dtype=float))
            else:  # class agnostic
                all_selection = []
                for j in range(len(class_selections)):
                    all_selection += class_selections[j]
                np_all_selected: NDArray[np.float32] = np.array(all_selection)
                selected_indices = self.nms(np_all_selected)
                outputs.append(np_all_selected[selected_indices, :])
        return outputs

    def nms(self, np_selection: NDArray[np.float32]) -> tf.Tensor:
        boxes = np_selection[:, 1:5]
        scores = np_selection[:, 0]
        selected_indices = tf.image.non_max_suppression(boxes=boxes,
                                                        scores=scores,
                                                        max_output_size=self.top_k,
                                                        iou_threshold=self.nms_thresh)
        return selected_indices
