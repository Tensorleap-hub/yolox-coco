from typing import Tuple

import numpy as np

from src.utils.dataclasses import RescaleMetadata, PaddingCoordinates


def xyxy2cxcywh(bboxes):
    """
    Transforms bboxes from xyxy format to centerized xy wh format
    :param bboxes: array, shaped (nboxes, 4)
    :return: modified bboxes
    """
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def cxcywh2xyxy(bboxes):
    """
    Transforms bboxes from centerized xy wh format to xyxy format
    :param bboxes: array, shaped (nboxes, 4)
    :return: modified bboxes
    """
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
    return bboxes


def shift_bboxes(targets: np.array, shift_w: float, shift_h: float) -> np.array:
    """Shift bboxes with respect to padding values.

    :param targets:  Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    :param shift_w:  shift width.
    :param shift_h:  shift height.
    :return:         Bboxes transformed of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    """
    boxes, labels = targets[:, :4], targets[:, 4:]
    boxes[:, [0, 2]] += shift_w
    boxes[:, [1, 3]] += shift_h
    return np.concatenate((boxes, labels), 1)


def rescale_xyxy_bboxes(targets: np.array, r: float) -> np.array:
    """Scale targets to given scale factors.

    :param targets:  Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    :param r:        DetectionRescale coefficient that was applied to the image
    :return:         Rescaled Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    """
    targets = targets.copy()
    boxes, targets = targets[:, :4], targets[:, 4:]
    boxes = xyxy2cxcywh(boxes)
    boxes *= r
    boxes = cxcywh2xyxy(boxes)
    return np.concatenate((boxes, targets), 1)


def rescale_bboxes(targets: np.ndarray, scale_factors: Tuple[float, float]) -> np.ndarray:
    """Rescale bboxes to given scale factors, without preserving aspect ratio.

    :param targets:         Targets to rescale (N, 4+), where target[:, :4] is the bounding box coordinates.
    :param scale_factors:   Tuple of (scale_factor_h, scale_factor_w) scale factors to rescale to.
    :return:                Rescaled targets.
    """

    targets = targets.astype(np.float32, copy=True)

    sy, sx = scale_factors
    targets[:, :4] *= np.array([[sx, sy, sx, sy]], dtype=targets.dtype)
    return targets


def postprocess_bboxes(bboxes, pad_params: PaddingCoordinates, rescale_params: RescaleMetadata):
    shifted_bboxes = shift_bboxes(bboxes, pad_params.top, pad_params.left)
    rescaled_bboxes = rescale_bboxes(shifted_bboxes, scale_factors=(
        1 / rescale_params.scale_factor_h, 1 / rescale_params.scale_factor_w))
    return rescaled_bboxes
