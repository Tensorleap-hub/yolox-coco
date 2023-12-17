import os
from json import load, dump

from yolox.config import annotation_files_dir, CONFIG

train_ann_path = os.path.join(annotation_files_dir, CONFIG['train_file'])

for path in [train_ann_path]:
    with open(path, 'r') as f:
        ann_file = load(f)

    # Edit all file names
    for i, image in enumerate(ann_file["images"]):
        if i % 50 == 0:
            image["file_name"] = f"../../true_negative_images/{image['file_name']}"

    with open(path, 'w') as f:
        dump(ann_file, f)
