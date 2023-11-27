import os
from typing import List, Dict, Union
import numpy as np
from PIL import Image

from code_loader import leap_binder

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import LeapDataType
from pycocotools.coco import COCO

from yolonas.config import dataset_path, CONFIG
from yolonas.custom_layers import MockOneClass
from yolonas.metrics import custom_yolo_nas_loss, placeholder_loss, general_metrics_dict, od_loss
from yolonas.utils.general_utils import extract_and_cache_bboxes, map_class_ids
from yolonas.visualizers import pred_bb_decoder, gt_bb_decoder
from yolonas.utils.confusion_matrix import confusion_matrix_metric


# ----------------------------------------------------data processing--------------------------------------------------
def subset_images() -> List[PreprocessResponse]:
    """
    This function returns the training and validation datasets in the format expected by tensorleap
    """
    # initialize COCO api for instance annotations
    train_coco = COCO(os.path.join(dataset_path, CONFIG['train_file']))
    imgIds = train_coco.getImgIds()
    imgs = train_coco.loadImgs(imgIds)
    existing_images = set(train_coco.imgs.keys())
    x_train_raw = train_coco.loadImgs(set(imgIds).intersection(existing_images))

    val_coco = COCO(os.path.join(dataset_path, CONFIG['val_file']))
    imgIds = val_coco.getImgIds()
    imgs = val_coco.loadImgs(imgIds)
    existing_images = set(val_coco.imgs.keys())
    x_val_raw = val_coco.loadImgs(set(imgIds).intersection(existing_images))

    train_size = min(len(x_train_raw), CONFIG['TRAIN_SIZE'])
    val_size = min(len(x_val_raw), CONFIG['VAL_SIZE'])
    training_subset = PreprocessResponse(length=train_size, data={'cocofile': train_coco,
                                                                  'samples': x_train_raw,
                                                                  'subdir': 'train'})

    validation_subset = PreprocessResponse(length=val_size, data={'cocofile': val_coco,
                                                                  'samples': x_val_raw,
                                                                  'subdir': 'val'})
    return [training_subset, validation_subset]


def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    """
    Returns a BGR image normalized and padded
    """
    data = data.data
    x = data['samples'][idx]
    path = os.path.join(dataset_path, f"images/{x['file_name']}")
    image = Image.open(path)
    image = image.resize((CONFIG['IMAGE_SIZE'][0], CONFIG['IMAGE_SIZE'][1]), Image.BILINEAR)
    return np.asarray(image) / 255.


def get_annotation_coco(idx: int, data: PreprocessResponse) -> np.ndarray:
    x = data['samples'][idx]
    coco = data['cocofile']
    ann_ids = coco.getAnnIds(imgIds=x['id'])
    anns = coco.loadAnns(ann_ids)
    return anns


def get_bbs(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    bboxes = extract_and_cache_bboxes(idx, data)
    bboxes = map_class_ids(bboxes)
    return bboxes


def get_dummy_gt(idx: int, data: PreprocessResponse) -> np.ndarray:
    return np.array([0])


# ----------------------------------------------------------metadata----------------------------------------------------
def get_fname(index: int, subset: PreprocessResponse) -> str:
    data = subset.data
    x = data['samples'][index]
    return str(x['file_name'])


def get_original_width(index: int, subset: PreprocessResponse) -> int:
    data = subset.data
    x = data['samples'][index]
    return int(x['width'])


def get_original_height(index: int, subset: PreprocessResponse) -> int:
    data = subset.data
    x = data['samples'][index]
    return int(x['height'])


def bbox_num(bbs: np.ndarray) -> int:
    number_of_bb = np.count_nonzero(bbs[..., -1] != CONFIG['BACKGROUND_LABEL'])
    return int(number_of_bb)


def get_avg_bb_area(bbs: np.ndarray) -> float:
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    if len(valid_bbs) > 0:
        areas = valid_bbs[:, 2] * valid_bbs[:, 3]
        return float(round(areas.mean(), 3))
    else:
        return float(0.0)


def get_avg_bb_aspect_ratio(bbs: np.ndarray) -> float:
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    if len(valid_bbs) > 0:
        aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
        return float(round(aspect_ratios.mean(), 3))
    else:
        return float(0.0)


def get_instances_num(bbs: np.ndarray) -> float:
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    return float(round(valid_bbs.shape[0], 3))


# def get_obj_bbox_occlusions_count(img: np.ndarray, bboxes: np.ndarray, calc_avg_flag=False) -> float:
#     occlusion_threshold = 0.2  # Example threshold value
#     occlusions_count = count_obj_bbox_occlusions(img, bboxes, occlusion_threshold, calc_avg_flag)
#     return occlusions_count
#
#
# def get_obj_bbox_occlusions_avg(img: np.ndarray, bboxes: np.ndarray) -> float:
#     return get_obj_bbox_occlusions_count(img, bboxes, calc_avg_flag=True)


def count_duplicate_bbs(bbs_gt: np.ndarray) -> int:
    real_gt = bbs_gt[bbs_gt[..., 4] != CONFIG['BACKGROUND_LABEL']]
    return int(real_gt.shape[0] != np.unique(real_gt, axis=0).shape[0])


def count_small_bbs(bboxes: np.ndarray) -> float:
    obj_boxes = bboxes[bboxes[..., -1] == 0]
    areas = obj_boxes[..., 2] * obj_boxes[..., 3]
    return float(round(len(areas[areas < CONFIG['SMALL_BBS_TH']]), 3))


def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    bbs = get_bbs(idx, data)
    img = input_image(idx, data)

    metadatas = {
        "idx": idx,
        "fname": get_fname(idx, data),
        "origin_width": get_original_width(idx, data),
        "origin_height": get_original_height(idx, data),
        "instances_number": get_instances_num(bbs),
        "bbox_number": bbox_num(bbs),
        "bbox_area": get_avg_bb_area(bbs),
        "bbox_aspect_ratio": get_avg_bb_aspect_ratio(bbs),
        "duplicate_bb": count_duplicate_bbs(bbs),
        "small_bbs_number": count_small_bbs(bbs),
        "image_mean": float(img.mean()),
        "image_std": float(img.std()),
        "image_min": float(img.min()),
        "image_max": float(img.max())
        # "count_total_obj_bbox_occlusions": get_obj_bbox_occlusions_count(img, bbs),
        # "avg_obj_bbox_occlusions": get_obj_bbox_occlusions_avg(img, bbs),
    }
    return metadatas


# ---------------------------------------------------------binding------------------------------------------------------
# preprocess function
leap_binder.set_preprocess(subset_images)

# unlabeled data preprocess
# set input and gt
leap_binder.set_input(input_image, 'images')
leap_binder.set_ground_truth(get_bbs, 'bbs')
# set prediction (object)
leap_binder.add_prediction('bbox coordinates', ["x1", "y1", "x2", "y2"])
leap_binder.add_prediction('classes', [f"class_{i}" for i in range(CONFIG['CLASSES'])])
# set custom loss
leap_binder.add_custom_loss(placeholder_loss, 'zero_loss')
leap_binder.add_custom_loss(custom_yolo_nas_loss, 'custom_yolo_nas_loss')
leap_binder.add_custom_loss(od_loss, 'od_loss')

# set visualizers
leap_binder.set_visualizer(gt_bb_decoder, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(pred_bb_decoder, 'pred_bb_decoder', LeapDataType.ImageWithBBox)
leap_binder.add_custom_metric(confusion_matrix_metric, "Confusion metric")

#
leap_binder.set_custom_layer(MockOneClass, 'reduce_to_one_class')
# set metadata
leap_binder.set_metadata(metadata_dict, name='metadata')
# metric
leap_binder.add_custom_metric(general_metrics_dict, 'od_metrics')
if __name__ == '__main__':
    leap_binder.check()
