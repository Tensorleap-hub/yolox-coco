import json
import os
from pycocotools.coco import COCO
from yolox.config import CONFIG

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

id_to_name = {class_id: value['name'] for class_id, value in train_coco.cats.items()}
original_labels_to_consecutive = {original_label: consecutive_label for consecutive_label, original_label in
                                  enumerate(train_label_ids)}
logit_to_name = {logit: name for logit, name in zip(original_labels_to_consecutive.values(), id_to_name.values())}

with open('yolox/label_id_to_name.json', 'w') as f:
    json.dump(logit_to_name, f, indent=4)
