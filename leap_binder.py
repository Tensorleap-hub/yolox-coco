import os
from typing import List, Dict, Union
import numpy as np
from PIL import Image

from code_loader import leap_binder

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import LeapDataType
from pycocotools.coco import COCO

from yolox.config import root_dataset_path, unlabeled_dataset_path, CONFIG, annotation_files_dir
from yolox.metrics import placeholder_loss, od_metrics_dict
from yolox.utils.custom_layers import MockNClass
from yolox.utils.general_utils import extract_and_cache_bboxes, map_class_ids
from yolox.utils.yolox_loss import custom_yolox_loss
from yolox.visualizers import gt_bb_decoder, pred_bb_visualizer
from yolox.utils.confusion_matrix import confusion_matrix_metric


# ----------------------------------------------------data processing--------------------------------------------------
def subset_images() -> List[PreprocessResponse]:
    """
    This function returns the training and validation datasets in the format expected by tensorleap
    """
    # initialize COCO api for instance annotations
    train_coco = COCO(os.path.join(annotation_files_dir, CONFIG['train_file']))
    imgIds = train_coco.getImgIds()
    imgs = train_coco.loadImgs(imgIds)
    existing_images = set(train_coco.imgs.keys())
    x_train_raw = train_coco.loadImgs(set(imgIds).intersection(existing_images))

    val_coco = COCO(os.path.join(annotation_files_dir, CONFIG['val_file']))
    imgIds = val_coco.getImgIds()
    imgs = val_coco.loadImgs(imgIds)
    existing_images = set(val_coco.imgs.keys())
    x_val_raw = val_coco.loadImgs(set(imgIds).intersection(existing_images))

    train_size = min(len(x_train_raw), CONFIG['TRAIN_SIZE'])
    val_size = min(len(x_val_raw), CONFIG['VAL_SIZE'])
    training_subset = PreprocessResponse(length=train_size, data={'cocofile': train_coco,
                                                                  'dataset_path': root_dataset_path,
                                                                  'samples': x_train_raw,
                                                                  'subdir': 'train'})

    validation_subset = PreprocessResponse(length=val_size, data={'cocofile': val_coco,
                                                                  'dataset_path': root_dataset_path,
                                                                  'samples': x_val_raw,
                                                                  'subdir': 'val'})
    return [training_subset, validation_subset]


def unlabeled_preprocessing_func() -> PreprocessResponse:
    """
    This function returns the unlabeled data split in the format expected by tensorleap
    """
    unlabeled_files = os.listdir(unlabeled_dataset_path)[:CONFIG['UNLABELED_SIZE']]
    unlabeled_size = len(unlabeled_files)
    print(unlabeled_size)
    unlabeled_subset = PreprocessResponse(length=unlabeled_size, data={'unlabeled_files': unlabeled_files,
                                                                       'dataset_path': unlabeled_dataset_path,
                                                                       'subdir': 'unlabeled'})
    return unlabeled_subset


def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    """
    Returns a BGR image normalized and padded
    """
    data = data.data
    if data['subdir'] == 'unlabeled':
        path = os.path.join(data['dataset_path'], data['unlabeled_files'][idx])
    else:
        x = data['samples'][idx]
        file_name: str = x['file_name']
        if '../' in file_name:
            file_name = file_name.replace("../", "")
            path = os.path.join(data['dataset_path'].replace('images', ''), f"{file_name}")
        else:
            path = os.path.join(data['dataset_path'], f"{file_name}")
    image = Image.open(path)
    image = image.resize((CONFIG['IMAGE_SIZE'][0], CONFIG['IMAGE_SIZE'][1]), Image.BILINEAR)
    image_array = np.asarray(image)
    if image_array.shape[-1] != 3:
        image_array = np.repeat(image_array[..., np.newaxis], 3, axis=-1)
    return image_array


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
    number_of_bb = np.count_nonzero(bbs[..., CONFIG['GT_CLS_IDX']] != CONFIG['BACKGROUND_LABEL'])
    return int(number_of_bb)


def get_avg_bb_area(bbs: np.ndarray) -> float:
    valid_bbs = bbs[bbs[..., CONFIG['GT_CLS_IDX']] != CONFIG['BACKGROUND_LABEL']]
    if len(valid_bbs) > 0:
        areas = valid_bbs[:, 2] * valid_bbs[:, 3]
        return float(round(areas.mean(), 3))
    else:
        return float(0.0)


def get_avg_bb_aspect_ratio(bbs: np.ndarray) -> float:
    valid_bbs = bbs[bbs[..., CONFIG['GT_CLS_IDX']] != CONFIG['BACKGROUND_LABEL']]
    if len(valid_bbs) > 0:
        aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
        return float(round(aspect_ratios.mean(), 3))
    else:
        return float(0.0)


def get_instances_num(bbs: np.ndarray) -> int:
    valid_bbs = bbs[bbs[..., CONFIG['GT_CLS_IDX']] != CONFIG['BACKGROUND_LABEL']]
    return int(valid_bbs.shape[0])


def count_duplicate_bbs(bbs_gt: np.ndarray) -> int:
    real_gt = bbs_gt[bbs_gt[..., CONFIG['GT_CLS_IDX']] != CONFIG['BACKGROUND_LABEL']]
    return int(real_gt.shape[0] != np.unique(real_gt, axis=0).shape[0])


# def count_persons(bbs_gt: np.ndarray) -> int:
#     person_gt = bbs_gt[bbs_gt[..., 4] == CONFIG['class_name_to_id']['person']]
#     return int(person_gt.shape[0])


def count_small_bbs(bboxes: np.ndarray) -> int:
    obj_boxes = bboxes[bboxes[..., CONFIG['GT_CLS_IDX']] != CONFIG['BACKGROUND_LABEL']]
    areas = obj_boxes[..., 2] * obj_boxes[..., 3]
    return len(areas[areas < CONFIG['SMALL_BBS_TH']])


def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    img = input_image(idx, data)

    if data.data['subdir'] == 'unlabeled':
        metadatas = {
            "idx": idx,
            "fname": data.data["unlabeled_files"][idx],
            "origin_width": 0,
            "origin_height": 0,
            "instances_number": 0,
            "bbox_number": 0,
            "bbox_area": 0,
            "bbox_aspect_ratio": 0,
            "duplicate_bb": 0,
            "small_bbs_number": 0,
            "image_mean": float(img.mean()),
            "image_std": float(img.std()),
            "image_min": float(img.min()),
            "image_max": float(img.max()),
            # "number_of_persons": 0,
            'veh_pose': -1,
            'veh_type': -1,
            'plate_state': 'null',
            'plate_number': 'null'
        }
        return metadatas

    bbs = get_bbs(idx, data)
    metadatas = {
        "idx": idx,
        "fname": str(get_fname(idx, data)),
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
        "image_max": float(img.max()),
        # "number_of_persons": count_persons(bbs),
    }

    # add extra annotations as metadata
    anns = get_annotation_coco(idx, data.data)

    if len(anns) == 1:
        anns = anns[0]
        if anns['veh_pose'] is not None:  # assuming coco loads `null` values as None
            veh_pose = int(anns['veh_pose'])
        else:
            veh_pose = -1
        if anns['veh_type'] is not None:
            veh_type = int(anns['veh_type'])
        else:
            veh_type = -1
        if anns['plate_state'] is not None:
            plate_state = str(anns['plate_state'])
        else:
            plate_state = 'null'
        if anns['plate_number'] is not None:
            plate_number = str(anns['plate_number'])
        else:
            plate_number = 'null'
        metadatas['veh_pose'] = veh_pose
        metadatas['veh_type'] = veh_type
        metadatas['plate_state'] = plate_state
        metadatas['plate_number'] = plate_number
    else:
        metadatas['veh_pose'] = -1
        metadatas['veh_type'] = -1
        metadatas['plate_state'] = 'null'
        metadatas['plate_number'] = 'null'

    return metadatas


# ---------------------------------------------------------binding------------------------------------------------------
# preprocess function
leap_binder.set_preprocess(subset_images)
leap_binder.set_unlabeled_data_preprocess(unlabeled_preprocessing_func)
# unlabeled data preprocess
# set input and gt
leap_binder.set_input(input_image, 'images')
leap_binder.set_ground_truth(get_bbs, 'bbs')
# set prediction (object)
leap_binder.add_prediction(name='predictions',
                           labels=["xc", "yc", "w", "h"] +  # bounding box coords
                                  ["obj"] +  # object confidence
                                  list(CONFIG['class_id_to_name'].values()) +  # classes confidence
                                  ['0', 'Front', 'Back'] +  # orientation
                                  [str(i) for i in range(8)],
                           channel_dim=1)
# set custom loss
leap_binder.add_custom_loss(placeholder_loss, 'zero_loss')
leap_binder.add_custom_loss(custom_yolox_loss, 'custom_yolox_loss')
# set visualizers
leap_binder.set_visualizer(gt_bb_decoder, 'gt_bb_visualizer', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(pred_bb_visualizer, 'pred_bb_visualizer', LeapDataType.ImageWithBBox)
leap_binder.add_custom_metric(confusion_matrix_metric, "confusion matrix")
# set metadata
leap_binder.set_metadata(metadata_dict, name='metadata')
# custom metrics
leap_binder.add_custom_metric(od_metrics_dict, 'od_metrics')

leap_binder.set_custom_layer(MockNClass, 'slice_n_classes')
if __name__ == '__main__':
    leap_binder.check()
