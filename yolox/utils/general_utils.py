from typing import List, Union, Dict

import numpy as np
import tensorflow as tf
from code_loader.contract.responsedataclasses import BoundingBox
from matplotlib import patches
import matplotlib.pyplot as plt
from numpy._typing import NDArray
from yolox.config import CONFIG
from code_loader.helpers.detection.utils import xyxy_to_xywh_format


def bb_array_to_object(bb_array: Union[NDArray[float], tf.Tensor], iscornercoded: bool = True, bg_label: int = 0,
                       is_gt=False) -> List[BoundingBox]:
    """
    Assumes a (X,Y,W,H) Format for the BB text
    bb_array is (CLASSES,TOP_K,PROPERTIES) WHERE PROPERTIES =(conf,xmin,ymin,xmax,ymax)
    """
    bb_list = []
    original_label_to_name = CONFIG['class_id_to_name']
    if not isinstance(bb_array, np.ndarray):
        bb_array = np.array(bb_array)
    if len(bb_array.shape) == 3:
        bb_array = bb_array.reshape(-1, bb_array.shape[-1])
    for i in range(bb_array.shape[0]):
        if bb_array[i][CONFIG['GT_CLS_IDX']] != bg_label:
            if iscornercoded:
                x, y, w, h = xyxy_to_xywh_format(bb_array[i][:4])
                # unormalize to image dimensions
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            conf = 1 if is_gt else bb_array[i][0]
            label_name = original_label_to_name.get(bb_array[i][CONFIG['GT_CLS_IDX']])
            curr_bb = BoundingBox(x=x, y=y, width=w, height=h, confidence=conf,
                                  label=str(label_name))

            bb_list.append(curr_bb)
    return bb_list


def extract_and_cache_bboxes(idx: int, data: Dict) -> np.ndarray:
    x = data['samples'][idx]
    coco = data['cocofile']
    ann_ids = coco.getAnnIds(imgIds=x['id'])
    anns = coco.loadAnns(ann_ids)
    bboxes = np.zeros([CONFIG['MAX_BB_PER_IMAGE'], 7], dtype=np.float32)
    max_anns = min(CONFIG['MAX_BB_PER_IMAGE'], len(anns))
    for i in range(max_anns):
        ann = anns[i]
        if isinstance(ann['bbox'], list):
            img_size = (x['height'], x['width'])
            class_id = ann['category_id']
            veh_type = 0
            veh_pose = 0
            if ann['veh_type'] is not None:
                veh_type = int(ann['veh_type'])
            if ann['veh_pose'] is not None:
                veh_pose = int(ann['veh_pose'])
            bbox = np.expand_dims(ann['bbox'], 0)[0].astype(np.float32)
            if not CONFIG['gt_xy_center']:
                bbox[0] += bbox[2] / 2.
                bbox[1] += bbox[3] / 2.
            bbox /= np.array((img_size[1], img_size[0], img_size[1], img_size[0])).astype(np.float32)
            bboxes[i, :4] = bbox
            bboxes[i, 4] = class_id
            bboxes[i, 5] = veh_pose
            bboxes[i, 6] = veh_type
    bboxes[max_anns:, 4] = CONFIG['BACKGROUND_LABEL']
    return bboxes  # [x, y, w, h, cls, pose, type]


def map_class_ids(bboxes: np.ndarray) -> np.ndarray:
    mapping_dict = CONFIG.get('labels_original_to_consecutive', None)
    if mapping_dict is None:
        return bboxes
    else:
        mapped_bboxes_ids = np.vectorize(mapping_dict.get)(bboxes[:, CONFIG['GT_CLS_IDX']].astype(int))
        bboxes[:, CONFIG['GT_CLS_IDX']] = mapped_bboxes_ids
        return bboxes


def draw_image_with_boxes(image, bounding_boxes):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Draw bounding boxes on the image
    for bbox in bounding_boxes:
        x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
        confidence, label = bbox.confidence, bbox.label

        # Convert relative coordinates to absolute coordinates
        abs_x = x * image.shape[1]
        abs_y = y * image.shape[0]
        abs_width = width * image.shape[1]
        abs_height = height * image.shape[0]

        # Create a rectangle patch
        rect = patches.Rectangle(
            (abs_x - abs_width / 2, abs_y - abs_height / 2),
            abs_width, abs_height,
            linewidth=2, edgecolor='r', facecolor='none'
        )

        # Add the rectangle to the axes
        ax.add_patch(rect)

        # Display label and confidence
        ax.text(abs_x - abs_width / 2, abs_y - abs_height / 2 - 5,
                f"{label} {confidence:.2f}", color='r', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # Show the image with bounding boxes
    plt.show()
