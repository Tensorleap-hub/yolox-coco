import onnx
from onnx import helper
import sys

# --------------------------------------------------------------------------
# Your onnx model's name
# model_name = 'yolo_nas_s'

# --------------------------------------------------------------------------
if len(sys.argv) != 3:
    print("Expected Usage is: python add_permute_layer.py input_onnx_path.onnx output_onnx_path.onnx")
model_path = f"{sys.argv[1]}"
model = onnx.load(model_path)

output1_name = model.graph.output[0].name
output2_name = model.graph.output[1].name

new_output1_name = f'{output1_name}_permuted'
new_output2_name = f'{output2_name}_permuted'
# Define the permutation order for the permute layer
perm_order = [0, 2, 1]  # Adjust this based on your specific needs

# Add permute layer after the first output
permute_layer1 = helper.make_node(
    'Transpose',
    [output1_name],
    [new_output1_name],
    perm=perm_order
)

model.graph.node.extend([permute_layer1])

# Add permute layer after the second output
permute_layer2 = helper.make_node(
    'Transpose',
    [output2_name],
    [new_output2_name],
    perm=perm_order
)

model.graph.node.extend([permute_layer2])

# Update the graph to use the permuted output names
model.graph.output[0].name = new_output1_name
model.graph.output[1].name = new_output2_name

# Save the modified model
output_model_path = f"{sys.argv[2]}"
onnx.save(model, output_model_path)
