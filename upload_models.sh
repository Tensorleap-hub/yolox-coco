#!/bin/bash
chmod +x set_permuted_model_mapping.sh
chmod +x set_orig_model_mapping.sh
SAVE_PATH="model/yolo_permuted.onnx"
echo "Uploading model from path: $1"
cp mappings/leap_mapping_regular.yaml leap_mapping.yaml
leap projects push $1 || true  # Continue after this even if this fails
python add_permute_layer.py $1 $SAVE_PATH
if [ $? != 0 ];
then
    echo "Failed to add a permute layer to model"
else
	echo "Added Permute Layer to model - pushing to hub..."
	cp mappings/leap_mapping_permuted_model.yaml leap_mapping.yaml
	leap projects push $SAVE_PATH
fi
