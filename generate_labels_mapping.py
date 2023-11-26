import json
import os
from pycocotools.coco import COCO
from yolonas.config import CONFIG

# get dataset path from config
dataset_path = CONFIG['dataset_path']

# load coco train set
try:
    train_coco = COCO(os.path.join(dataset_path, 'train.json'))
    train_categories = train_coco.loadCats(train_coco.getCatIds())
    train_label_ids = set(category['id'] for category in train_categories)
except FileNotFoundError as e:
    print(f"error during the loading of train subset, could not found file: {e.filename}\n"
          f"cannot proceed without train labels")
    raise e

original_labels_to_consecutive = {original_label: consecutive_label for consecutive_label, original_label in enumerate(train_label_ids)}

with open('yolonas/labels_mapping.json', 'w') as f:
    json.dump(original_labels_to_consecutive, f)


