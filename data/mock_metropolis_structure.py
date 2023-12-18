import os
from json import load, dump
import numpy as np

from yolox.config import annotation_files_dir, CONFIG

train_ann_path = os.path.join(annotation_files_dir, CONFIG['train_file'])
val_ann_path = os.path.join(annotation_files_dir, CONFIG['val_file'])

for path in [val_ann_path]:
    with open(path, 'r') as f:
        ann_file = load(f)

    # Edit all file names
    # for i, image in enumerate(ann_file["images"]):
    #     if i % 50 == 0:
    #         image["file_name"] = f"../../true_negative_images/{image['file_name']}"

    # add metropolis annotations
    for i, ann in enumerate(ann_file["annotations"]):
        ann["veh_pose"] = np.random.randint(0, 3)
        ann["veh_type"] = np.random.randint(0, 8)
        ann["plate_state"] = 'CA'
        ann["plate_number"] = "example"
        if i % 6 == 0:
            ann["plate_state"] = None
            ann["plate_number"] = None

    with open(path, 'w') as f:
        dump(ann_file, f)
