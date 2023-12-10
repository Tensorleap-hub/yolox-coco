import onnx
from onnx import helper
import sys

if len(sys.argv) != 3:
    print("Expected Usage is: python add_permute_layer.py input_onnx_path.onnx output_onnx_path.onnx")
model_path = f"{sys.argv[1]}"
model = onnx.load(model_path)

output_name = model.graph.output[0].name

new_output_name = f'{output_name}_permuted'
# Define the permutation order for the permute layer
perm_order = [0, 2, 1]  # Adjust this based on your specific needs

# Add permute layer after the first output
permute_layer = helper.make_node(
    'Transpose',
    [output_name],
    [new_output_name],
    perm=perm_order
)

model.graph.node.extend([permute_layer])

# Update the graph to use the permuted output names
model.graph.output[0].name = new_output_name

# Save the modified model
output_model_path = f"{sys.argv[2]}"
onnx.save(model, output_model_path)
