#!/bin/bash
chmod +x set_permuted_model_mapping.sh
chmod +x set_orig_model_mapping.sh
SAVE_PATH="model/yolo_nas_permuted.onnx"
python add_permute_layer.py $1 $SAVE_PATH
if [ $? != 0 ];
then
    echo "Failed to add a permute layer to model"
else
	echo "Added Permute Layer to model - pushing to hub..."
	cp mappings/leap_mapping_permuted_model.yaml leap_mapping.yaml
	leap projects push $SAVE_PATH
fi
