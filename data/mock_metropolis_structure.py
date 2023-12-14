import os
from json import load, dump

from yolox.config import annotation_files_dir, CONFIG

train_ann_path = os.path.join(annotation_files_dir, CONFIG['train_file'])
val_ann_path = os.path.join(annotation_files_dir, CONFIG['val_file'])

for path in [train_ann_path, val_ann_path]:
    with open(path, 'r') as f:
        ann_file = load(f)

    # Edit all file names
    for image in ann_file["images"]:
        image["file_name"] = f"../../images/{image['file_name']}"

    with open(path, 'w') as f:
        dump(ann_file, f)