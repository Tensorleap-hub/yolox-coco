import json
from pathlib import Path
import os
from typing import Dict, Any
import yaml


def load_od_config() -> Dict[str, Any]:
    # Load the existing YAML config
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, 'object_detection_config.yml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    try:
        with open(os.path.join(root, 'labels_mapping.json'), 'r') as f:
            labels_mapping = json.load(f, object_pairs_hook=lambda pairs: {int(k): v for k, v in pairs})
        config['BACKGROUND_LABEL'] = max(labels_mapping.keys()) + 1
        labels_mapping[config['BACKGROUND_LABEL']] = config['BACKGROUND_LABEL']
        config['labels_original_to_consecutive'] = labels_mapping
        config['labels_consecutive_to_original'] = {v: k for k, v in labels_mapping.items()}

    except Exception as e:
        print(e)

    return config


CONFIG = load_od_config()

dataset_path = str(Path(CONFIG['dataset_path']).absolute())
