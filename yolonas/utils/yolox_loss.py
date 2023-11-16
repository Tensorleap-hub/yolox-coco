import os
from typing import List, Optional
import re
from functools import lru_cache

import pandas as pd
import cv2
from google.cloud import storage
from google.cloud.storage import Bucket
from google.oauth2 import service_account
import json
import numpy as np
from code_loader import leap_binder
import tensorflow as tf
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.contract.enums import DatasetMetadataType

N_SAMPLES = 1100
N_TRAIN = 10
N_TEST = N_SAMPLES - N_TRAIN

PROJECT_ID = 'hp-dev-project'
BUCKET_NAME = 'hp-datasets'
DIR_NAME = "Test_MegaDB_BarakHS_HI4AI"
image_name_col = "fname"
LABELS = ['0', '11', '8', '1', '3', '9']


def calculate_iou(grid_size: int, bbox_gt: tf.Tensor) -> tf.Tensor:
    factor = 1 / grid_size
    grid_map = tf.range(0, 1, factor)

    xx, yy = tf.meshgrid(grid_map, grid_map, indexing='ij')
    grid_matrix_map_start = tf.stack([tf.reshape(xx, [-1]), tf.reshape(yy, [-1])], axis=1)
    grid_matrix_map_end = grid_matrix_map_start + factor

    box_gt_start = bbox_gt[:2]
    box_gt_end = bbox_gt[2:]

    max_start = tf.math.maximum(box_gt_start, grid_matrix_map_start)
    min_end = tf.math.minimum(box_gt_end, grid_matrix_map_end)

    interArea = tf.reduce_prod(tf.math.maximum(min_end - max_start, [0, 0]), axis=1)
    boxAArea = tf.reduce_prod(box_gt_end - box_gt_start)
    boxBArea = tf.reduce_prod(grid_matrix_map_end - grid_matrix_map_start, axis=1)

    iou = interArea / (boxAArea + boxBArea - interArea + tf.keras.backend.epsilon())

    return iou


def meta_data_min_iou(idx, samples) -> float:
    scales = [80, 40, 20]
    gt = tf.convert_to_tensor(get_gt(idx, samples))
    ratio = tf.cast(gt[0][5], tf.float32)
    img_origin_size = tf.cast(gt[0][6:], tf.float32)

    all_bbox_max_iou = []

    for single_gt in gt:
        bbox_gt = single_gt[:4]
        bbox_gt = to_x1y1x2y2_in_percentage(bbox_gt, img_origin_size, ratio)
        all_scales_iou = tf.zeros((0,))
        for grid_size in scales:
            iou = calculate_iou(grid_size, bbox_gt)
            all_scales_iou = tf.concat([all_scales_iou, iou], axis=0)

        all_bbox_max_iou.append(all_scales_iou.numpy().max())

    return float(min(all_bbox_max_iou))


def gather_positive_examples(pred_grids: tf.Tensor, bbox_gt: tf.Tensor):
    iou = calculate_iou(pred_grids.shape[-1], bbox_gt)
    iou_indices = tf.where(iou > 0.08)
    x_positive_indices = tf.math.floordiv(iou_indices, pred_grids.shape[-1])
    y_positive_indices = tf.math.floormod(iou_indices, pred_grids.shape[-1])

    pred_flat = tf.reshape(pred_grids, (11, -1))

    confidence_flat_value = tf.reshape(tf.gather(iou, iou_indices), -1)
    positive_flat_indices = tf.reshape(y_positive_indices * pred_grids.shape[-1] + x_positive_indices, -1)

    positive_examples = tf.gather(pred_flat, positive_flat_indices, axis=-1)

    return positive_examples, positive_flat_indices, confidence_flat_value


def to_x1y1x2y2_in_percentage(bbox_gt, img_origin_size, ratio) -> tf.Tensor:
    bbox_gt = tf.cast(bbox_gt, tf.float32)
    center = (bbox_gt[:2] + bbox_gt[2:]) / 2
    widths = bbox_gt[2:] - bbox_gt[:2]
    xy_r = center / img_origin_size
    wh_r = widths / img_origin_size

    x1y1 = xy_r * img_origin_size * ratio
    w1h1 = wh_r * img_origin_size * ratio

    x2y2 = x1y1 + w1h1

    bbox_gt = tf.concat([x1y1, x2y2], 0)
    return bbox_gt / 640.0


def run_single_scale(pred_scale_reshaped: tf.Tensor, ratio, img_origin_size, y_true):
    n_grids = pred_scale_reshaped.shape[-1] * pred_scale_reshaped.shape[-1]

    true_classes_per_grid = tf.zeros((n_grids, 5), tf.float32)
    true_classes_per_grid = tf.concat([tf.ones((n_grids, 1), tf.float32), true_classes_per_grid], axis=-1)
    reg_loss_all_boxes = tf.zeros((0, 1))
    true_confidences = tf.zeros((n_grids, 1), tf.float32)
    for single_gt in y_true[0, :]:
        bbox_gt = single_gt[:4]
        bbox_gt = to_x1y1x2y2_in_percentage(bbox_gt, img_origin_size, ratio)

        positive_examples, positive_flat_indices, confidence_flat_values = gather_positive_examples(
            pred_scale_reshaped, bbox_gt)

        class_index = tf.cast(single_gt[4], tf.int64)
        positive_flat_indices = tf.expand_dims(positive_flat_indices, axis=0)
        update_indices_class = tf.transpose(
            tf.concat([positive_flat_indices, tf.ones(positive_flat_indices.shape, tf.int64) * class_index], axis=0))
        update_indices_default = tf.transpose(
            tf.concat([positive_flat_indices, tf.zeros(positive_flat_indices.shape, tf.int64) * class_index], axis=0))
        update_indices = tf.stack([update_indices_class, update_indices_default], axis=1)

        if update_indices.shape[0] == 0:
            continue

        true_classes_per_grid = tf.tensor_scatter_nd_update(
            true_classes_per_grid, update_indices, [[1.0, 0.0]] * update_indices.shape[0]
        )

        true_confidences = tf.tensor_scatter_nd_update(
            true_confidences, update_indices_default, confidence_flat_values
        )

        reg_loss = tf.keras.losses.MeanSquaredError()(tf.transpose(positive_examples)[:, :4] / 640.0, bbox_gt)
        reg_loss_all_boxes = tf.concat([reg_loss_all_boxes, tf.reshape(reg_loss, (tf.size(reg_loss), 1))], axis=0)

    return true_classes_per_grid, reg_loss_all_boxes, true_confidences


def simple_od_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    pred_scale_80_80 = y_pred[:, :, :6400]
    pred_scale_40_40 = y_pred[:, :, 6400:8000]
    pred_scale_20_20 = y_pred[:, :, 8000:]
    pred_scale_80_80 = tf.reshape(pred_scale_80_80, [1, 11, 80, 80])
    pred_scale_40_40 = tf.reshape(pred_scale_40_40, [1, 11, 40, 40])
    pred_scale_20_20 = tf.reshape(pred_scale_20_20, [1, 11, 20, 20])

    ratio = tf.cast(y_true[0][0][5], tf.float32)
    img_origin_size = tf.cast(y_true[0][0][6:], tf.float32)

    true_classes_80_80, reg_loss_80_80, true_confidences_80_80 = run_single_scale(
        pred_scale_80_80, ratio, img_origin_size, y_true)
    true_classes_40_40, reg_loss_40_40, true_confidences_40_40 = run_single_scale(
        pred_scale_40_40, ratio, img_origin_size, y_true)
    true_classes_20_20, reg_loss_20_20, true_confidences_20_20 = run_single_scale(
        pred_scale_20_20, ratio, img_origin_size, y_true)

    true_classes_all = tf.concat([true_classes_80_80, true_classes_40_40, true_classes_20_20], axis=0)
    true_confidences_all = tf.concat([true_confidences_80_80, true_confidences_40_40, true_confidences_20_20], axis=0)
    reg_loss_all = tf.concat([reg_loss_80_80, reg_loss_40_40, reg_loss_20_20], axis=0)
    predicted_classes_all = tf.transpose(y_pred[0, -6:, :])
    classes_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(true_classes_all, predicted_classes_all)

    mean_reg_bbox_loss = tf.reduce_mean(reg_loss_all)
    mean_reg_bbox_loss = tf.cond(tf.math.is_nan(mean_reg_bbox_loss), lambda: 0.0, lambda: mean_reg_bbox_loss)

    predicted_conf_all = tf.transpose(y_pred[0, 4, :])
    reg_conf_loss = tf.keras.losses.MeanSquaredError()(true_confidences_all, predicted_conf_all)

    return mean_reg_bbox_loss + classes_loss + reg_conf_loss


@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    auth_secret_string = os.environ['AUTH_SECRET']
    auth_secret = json.loads(auth_secret_string)
    if type(auth_secret) is dict:
        # getting credentials from dictionary account info
        credentials = service_account.Credentials.from_service_account_info(auth_secret)
    else:
        # getting credentials from path
        credentials = service_account.Credentials.from_service_account_file(auth_secret)
    project = credentials.project_id
    gcs_client = storage.Client(project=project, credentials=credentials)

    return gcs_client.bucket(bucket_name)


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "Tensorleap_data", BUCKET_NAME) + cloud_file_path

    # check if file is already exists
    if os.path.exists(local_file_path):
        return local_file_path

    bucket = _connect_to_gcs_and_return_bucket(BUCKET_NAME)
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path


def preprocess() -> List[PreprocessResponse]:
    csv_local_file_path = _download(f"{DIR_NAME}/testmegadbbarakhshi4ai-v3.0 CSV.csv")
    df = pd.read_csv(csv_local_file_path)
    df = df[df[image_name_col].str.contains('Def')]
    df = df[df['class_id'].astype(str).isin(LABELS)]
    input_images = list(df.fname.unique())
    input_images = input_images[:N_SAMPLES]
    train_images, test_images = input_images[:N_TRAIN], input_images[N_TRAIN: N_TRAIN + N_TEST]
    return [
        PreprocessResponse(length=len(train_images), data={"df": df, image_name_col: train_images,
                                                           'ratios': np.ones(len(train_images)),
                                                           'img_original_sizes': np.zeros((len(train_images), 2))}),
        PreprocessResponse(length=len(test_images), data={"df": df, image_name_col: test_images,
                                                          'ratios': np.ones(len(test_images)),
                                                          'img_original_sizes': np.zeros((len(test_images), 2))})]


def get_image(idx, samples, to_swap: bool = False):
    image_names = samples.data[image_name_col]
    image_file_name = image_names[idx]
    image_file_path = f'{DIR_NAME}/{image_file_name}'
    image_local_file_path_def = _download(image_file_path)
    image_file_path_ref = to_ref_fpath(image_file_path)
    image_local_file_path_ref = _download(image_file_path_ref)
    image = ps_color(image_local_file_path_def, image_local_file_path_ref)
    input_size = (640, 640)
    padded_img, r = preproc(image, input_size, to_swap=to_swap)
    samples.data['ratios'][idx] = r  # update ratio
    samples.data['img_original_sizes'][idx] = (image.shape[:2])
    return padded_img


def get_ratio(idx, samples):
    return samples.data['ratios'][idx]


def get_original_size(idx, samples):
    return samples.data['img_original_sizes'][idx]


def get_original_size_w(idx, samples):
    return samples.data['img_original_sizes'][idx][1]


def get_original_size_h(idx, samples):
    return samples.data['img_original_sizes'][idx][0]


def get_fname(idx, samples):
    image_names = samples.data[image_name_col]
    image_file_name = image_names[idx]
    return str(image_file_name)


def get_sample_idx(idx, samples):
    return idx


def count_gts(idx, samples):
    image_names = samples.data[image_name_col]
    image_file_name = image_names[idx]
    df = samples.data['df']
    gt_df = df[df[image_name_col] == image_file_name]
    n_labels = len(gt_df[gt_df['class_id'] != 0])
    return float(n_labels)


def get_label_bbox_count_callable(label_key: str):
    def func(idx: int, samples: PreprocessResponse) -> float:
        image_names = samples.data[image_name_col]
        image_file_name = image_names[idx]
        df = samples.data['df']
        gt_df = df[df[image_name_col] == image_file_name]
        return float(len(gt_df[gt_df['class_id'] == int(label_key)]))

    func.__name__ = f'{label_key}_bbox_count'
    return func


def get_gt(idx, samples):
    image_names = samples.data[image_name_col]
    image_file_name = image_names[idx]
    df = samples.data['df']
    gt_df = df.loc[df[image_name_col] == image_file_name]
    ratio = get_ratio(idx, samples)

    original_size = get_original_size(idx, samples)

    bbox_matrix = gt_df[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
    labels = gt_df['class_id'].to_numpy()
    label_indices = np.array([[LABELS.index(str(int(label)))] for label in labels], np.int32)

    concat_with_image_data = np.concatenate([bbox_matrix,
                                             label_indices,
                                             np.ones((bbox_matrix.shape[0], 1)) * ratio,
                                             np.ones((bbox_matrix.shape[0], 1)) * original_size[1],
                                             np.ones((bbox_matrix.shape[0], 1)) * original_size[0]], axis=1)

    return concat_with_image_data


def avg_bbox_per_img_origin_size(idx, samples):
    # todo change to scaled size
    gt = get_gt(idx, samples)
    bboxes = gt[:, :4]
    w = np.abs(bboxes[:, 3] - bboxes[:, 1])
    h = np.abs(bboxes[:, 2] - bboxes[:, 0])
    res = np.mean(w * h)
    if res is None:
        return 0.
    return res


def temp_loss(y_true, y_pred):
    return tf.reshape(y_pred, [-1])[0] - tf.reshape(y_pred, [-1])[0]


def viz_gt(image, gt) -> LeapImageWithBBox:
    bboxes = []
    for bbox in gt:
        img_origin_size = bbox[-2:]
        ratio = bbox[-3]
        x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2  # center
        w, h = np.abs((bbox[2] - bbox[0])), np.abs((bbox[3] - bbox[1]))
        x_r, y_r = x / img_origin_size[1], y / img_origin_size[0]
        w_r, h_r = w / img_origin_size[1], h / img_origin_size[0]
        x1, y1 = x_r * int(img_origin_size[1] * ratio), y_r * int(img_origin_size[0] * ratio)
        w1, h1 = w_r * int(img_origin_size[1] * ratio), h_r * int(img_origin_size[0] * ratio)
        only_bbox = np.array([x1, y1, w1, h1])
        only_bbox /= 640
        bboxes.append(BoundingBox(*only_bbox, confidence=1.0, label=str(LABELS[int(bbox[4])])))
    return LeapImageWithBBox(image, bboxes)


def viz_pred(image, prediction) -> LeapImageWithBBox:
    bboxes = []
    for bbox in prediction:
        only_bbox = bbox[:4]
        x, y = (only_bbox[0] + only_bbox[2]) / 2, (only_bbox[1] + only_bbox[3]) / 2
        w, h = np.abs(only_bbox[2] - only_bbox[0]), np.abs(only_bbox[3] - only_bbox[1])
        x, y, w, h = x / 640., y / 640., w / 640., h / 640.
        confidence = bbox[4] * np.max(bbox[5:])
        if confidence < 0.1:
            continue
        label = LABELS[np.argmax(bbox[5:])]
        bboxes.append(
            BoundingBox(
                x=x,
                y=y,
                width=w,
                height=h,
                confidence=confidence,
                label=label
            ))
    return LeapImageWithBBox(image, bboxes)


def to_ref_fpath(image_path):
    path_fname, filename_w_ext = os.path.split(image_path)
    m = re.search('def', filename_w_ext, re.IGNORECASE)
    filename2 = []
    image2 = []
    if m:
        fname_lowercase = filename_w_ext.lower()
        sub = "def"
        ind1 = fname_lowercase.find(sub)
        filename2 = list(filename_w_ext)
        # Ref
        if filename_w_ext[ind1].islower():
            filename2[ind1] = 'r'
        else:
            filename2[ind1] = 'R'
        filename2 = "".join(filename2)
        filename2 = path_fname + '/' + filename2
    #######################################
    else:
        print("Failed to find  REF/DEF image pair!")
        print("the missing file pair is :" + image_path)
    return filename2


def ps_color(image_path_def, image_path_ref):
    def_img = cv2.imread(image_path_def)
    ref_img = cv2.imread(image_path_ref)
    scale = 1
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    def_gray = cv2.cvtColor(def_img, cv2.COLOR_BGR2GRAY)
    fused_img = np.zeros(def_img.shape, dtype=np.uint8)
    try:
        fused_img[:, :, 0] = def_gray.astype("float32")  # Def
        fused_img[:, :, 1] = abs(def_gray.astype("float32") + scale * (
                ref_gray.astype("float32") - def_gray.astype("float32")))  # RefGray
        fused_img[:, :, 2] = def_gray.astype("float32")  # Def      # ???
    except:
        print(image_path_def)
        print(image_path_ref)
    fused_img = fused_img.astype("uint8")
    # clip
    fused_img[fused_img < 0] = 0
    fused_img[fused_img > 255] = 255
    return fused_img


def preproc(img, input_size, swap=(2, 0, 1), to_swap: bool = False):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                             interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img
    if to_swap:
        padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


# leap_binder.add_custom_loss(temp_loss, 'temp_loss')
leap_binder.add_custom_loss(simple_od_loss, 'simple_od_loss')
leap_binder.set_preprocess(preprocess)
leap_binder.set_input(get_image, 'image')
leap_binder.set_ground_truth(get_gt, 'gt')
pred_labels = ['x1', 'y1', 'x2', 'y2', 'confidence'] + LABELS
leap_binder.add_prediction('pred', pred_labels, [])
leap_binder.set_metadata(get_sample_idx, 'idx')
leap_binder.set_metadata(count_gts, 'num_gts')
leap_binder.set_metadata(get_ratio, 'ratio')
leap_binder.set_metadata(get_original_size_w, 'original_w')
leap_binder.set_metadata(get_original_size_h, 'original_h')
leap_binder.set_metadata(get_fname, 'file_name')
leap_binder.set_metadata(avg_bbox_per_img_origin_size, 'avg_bbox_per_img_origin_size')

for label in LABELS:
    leap_binder.set_metadata(get_label_bbox_count_callable(label), f'{label}_bbox_count')

leap_binder.set_visualizer(viz_gt, 'gt_bbox', LeapImageWithBBox.type)
leap_binder.set_visualizer(viz_pred, 'pred_bbox', LeapImageWithBBox.type)
#
# x = preprocess()
# get_image(0, x[0])
# meta_data_min_iou(0, x[0])
