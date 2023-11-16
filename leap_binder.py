import os
from typing import Union, List, Dict
import numpy as np
import tensorflow as tf
from PIL import Image

from code_loader import leap_binder

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import LeapDataType
from pycocotools.coco import COCO

from yolonas.config import dataset_path, CONFIG
from yolonas.data.preprocessing import load_set, preprocess_image
from yolonas.metrics import custom_yolo_nas_loss
from yolonas.utils.general_utils import extract_and_cache_bboxes
from yolonas.visualizers import pred_bb_decoder, gt_bb_decoder


# ----------------------------------------------------data processing--------------------------------------------------
def subset_images() -> List[PreprocessResponse]:
    """
    This function returns the training and validation datasets in the format expected by tensorleap
    """
    # initialize COCO api for instance annotations
    train = COCO(os.path.join(dataset_path, 'train.json'))
    x_train_raw = load_set(coco=train, load_union=CONFIG['LOAD_UNION_CATEGORIES_IMAGES'], local_filepath=dataset_path)

    val = COCO(os.path.join(dataset_path, 'val.json'))
    x_val_raw = load_set(coco=val, load_union=CONFIG['LOAD_UNION_CATEGORIES_IMAGES'], local_filepath=dataset_path)

    test = COCO(os.path.join(dataset_path, 'test.json'))
    x_test_raw = load_set(coco=test, load_union=CONFIG['LOAD_UNION_CATEGORIES_IMAGES'], local_filepath=dataset_path)

    train_size = min(len(x_train_raw), CONFIG['TRAIN_SIZE'])
    val_size = min(len(x_val_raw), CONFIG['VAL_SIZE'])
    test_size = min(len(x_val_raw), CONFIG['TEST_SIZE'])

    np.random.seed(0)
    train_idx, val_idx, test_idx = (np.random.choice(len(x_train_raw), train_size, replace=False),
                                    np.random.choice(len(x_val_raw), val_size, replace=False),
                                    np.random.choice(len(x_test_raw), test_size, replace=False))
    training_subset = PreprocessResponse(length=train_size, data={'cocofile': train,
                                                                  'samples': np.take(x_train_raw, train_idx),
                                                                  'subdir': 'train'})
    validation_subset = PreprocessResponse(length=val_size, data={'cocofile': val,
                                                                  'samples': np.take(x_val_raw, val_idx),
                                                                  'subdir': 'val'})
    test_subset = PreprocessResponse(length=test_size, data={'cocofile': test,
                                                             'samples': np.take(x_test_raw, test_idx),
                                                             'subdir': 'test'})
    return [training_subset, validation_subset, test_subset]


def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    """
    Returns a BGR image normalized and padded
    """
    data = data.data
    x = data['samples'][idx]
    path = os.path.join(dataset_path, f"images/{x['file_name']}")
    # rescale
    image = np.array(
        Image.open(path).resize((CONFIG['IMAGE_SIZE'][0], CONFIG['IMAGE_SIZE'][1]), Image.BILINEAR)) / 255.
    return image


def get_annotation_coco(idx: int, data: PreprocessResponse) -> np.ndarray:
    x = data['samples'][idx]
    coco = data['cocofile']
    ann_ids = coco.getAnnIds(imgIds=x['id'])
    anns = coco.loadAnns(ann_ids)
    return anns


def get_bbs(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    bboxes = extract_and_cache_bboxes(idx, data)
    return bboxes


def get_dummy_gt(idx: int, data: PreprocessResponse) -> np.ndarray:
    return np.array([0])


# ----------------------------------------------------------metadata----------------------------------------------------
def get_fname(index: int, subset: PreprocessResponse) -> str:
    data = subset.data
    x = data['samples'][index]
    return x['file_name']


def get_original_width(index: int, subset: PreprocessResponse) -> int:
    data = subset.data
    x = data['samples'][index]
    return x['width']


def get_original_height(index: int, subset: PreprocessResponse) -> int:
    data = subset.data
    x = data['samples'][index]
    return x['height']


# def bbox_num(bbs: np.ndarray) -> int:
#     number_of_bb = np.count_nonzero(bbs[..., -1] != CONFIG['BACKGROUND_LABEL'])
#     return number_of_bb
#
#
# def get_avg_bb_area(bbs: np.ndarray) -> float:
#     valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
#     areas = valid_bbs[:, 2] * valid_bbs[:, 3]
#     return areas.mean()
#
#
# def get_avg_bb_aspect_ratio(bbs: np.ndarray) -> float:
#     valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
#     aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
#     return aspect_ratios.mean()
#
#
# def get_instances_num(bbs: np.ndarray) -> float:
#     valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
#     return float(valid_bbs.shape[0])


# def get_object_instances_num(bbs: np.ndarray) -> float:
#     label = CONFIG['CATEGORIES'].index('Object')
#     valid_bbs = bbs[bbs[..., -1] == label]
#     return float(valid_bbs.shape[0])


# def get_obj_bbox_occlusions_count(img: np.ndarray, bboxes: np.ndarray, calc_avg_flag=False) -> float:
#     occlusion_threshold = 0.2  # Example threshold value
#     occlusions_count = count_obj_bbox_occlusions(img, bboxes, occlusion_threshold, calc_avg_flag)
#     return occlusions_count


# def get_obj_bbox_occlusions_avg(img: np.ndarray, bboxes: np.ndarray) -> float:
#     return get_obj_bbox_occlusions_count(img, bboxes, calc_avg_flag=True)


# def count_duplicate_bbs(bbs_gt: np.ndarray) -> int:
#     real_gt = bbs_gt[bbs_gt[..., 4] != CONFIG['BACKGROUND_LABEL']]
#     return int(real_gt.shape[0] != np.unique(real_gt, axis=0).shape[0])
#
#
# def count_small_bbs(bboxes: np.ndarray) -> float:
#     obj_boxes = bboxes[bboxes[..., -1] == 0]
#     areas = obj_boxes[..., 2] * obj_boxes[..., 3]
#     return float(len(areas[areas < CONFIG['SMALL_BBS_TH']]))
#
#
# def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
#     bbs = get_bbs(idx, data)
#     img = input_image(idx, data)
#
#     metadatas = {
#         "idx": idx,
#         "fname": get_fname(idx, data),
#         "origin_width": get_original_width(idx, data),
#         "origin_height": get_original_height(idx, data),
#         "instances_number": get_instances_num(bbs),
#         # "object_number": get_object_instances_num(bbs),
#         "bbox_number": bbox_num(bbs),
#         "bbox_area": get_avg_bb_area(bbs),
#         "bbox_aspect_ratio": get_avg_bb_aspect_ratio(bbs),
#         "duplicate_bb": count_duplicate_bbs(bbs),
#         "small_bbs_number": count_small_bbs(bbs),
#         # "count_total_obj_bbox_occlusions": get_obj_bbox_occlusions_count(img, bbs),
#         # "avg_obj_bbox_occlusions": get_obj_bbox_occlusions_avg(img, bbs),
#     }
#
#     return metadatas

def placeholder_loss(gt: tf.Tensor, y_pred_1: tf.Tensor, y_pred_2: tf.Tensor) -> tf.Tensor:  # return batch
    """
    Sums the classification and regression loss
    """
    return tf.reduce_mean(y_pred_1, axis=-1) * 0


# ---------------------------------------------------------binding------------------------------------------------------
# preprocess function
leap_binder.set_preprocess(subset_images)

# unlabeled data preprocess
# set input and gt
leap_binder.set_input(input_image, 'images')
leap_binder.set_ground_truth(get_bbs, 'bbs')
# set prediction (object)
leap_binder.add_prediction('object detection',
                           ["x", "y", "x", "y"] +
                           [f"class_{i}" for i in range(CONFIG['CLASSES'])])

# set custom loss
leap_binder.add_custom_loss(placeholder_loss, 'zero_loss')
leap_binder.add_custom_loss(custom_yolo_nas_loss, 'custom_yolo_nas_loss')
# # set visualizers
leap_binder.set_visualizer(gt_bb_decoder, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(pred_bb_decoder, 'pred_bb_decoder', LeapDataType.ImageWithBBox)
#
# # set custom metrics
leap_binder.add_custom_metric(custom_yolo_nas_loss, 'huber')
#
# # set metadata
# leap_binder.set_metadata(metadata_dict, name='metadata')

if __name__ == '__main__':
    leap_binder.check()
