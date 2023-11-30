import json
from pathlib import Path
import os
from typing import Dict, Any
import yaml
from pycocotools.coco import COCO


def load_od_config() -> Dict[str, Any]:
    # Load the existing YAML config
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, 'object_detection_config.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    try:
        with open(os.path.join(root, 'label_id_to_name.json'), 'r') as f:
            logit_to_name = json.load(f, object_pairs_hook=lambda pairs: {int(k): v for k, v in pairs})
        config['class_id_to_name'] = logit_to_name
        config['class_name_to_id'] = {v: k for k, v in config['class_id_to_name'].items()}
    except Exception as e:
        print(e)

    coco = COCO(os.path.join(str(Path(config['dataset_path']).absolute()), config['train_file']))
    config['labels_original_to_consecutive'] = {original_label: consecutive_label
                                                for
                                                consecutive_label, original_label in
                                                enumerate(coco.cats.keys())}
    config['BACKGROUND_LABEL'] = max(config['labels_original_to_consecutive'].keys()) + 1
    config['labels_original_to_consecutive'][config['BACKGROUND_LABEL']] = config['BACKGROUND_LABEL']

    return config


CONFIG = load_od_config()

dataset_path = str(Path(CONFIG['dataset_path']).absolute())
