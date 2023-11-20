import onnx
from onnx import helper

# --------------------------------------------------------------------------
# Your onnx model's name
model_name = 'yolo_nas_s'
# Assume you have two outputs, replace these with your actual output names
output1_name = "911"
output2_name = "903"
# --------------------------------------------------------------------------
model_path = f"{model_name}.onnx"
model = onnx.load(model_path)

# Define the permutation order for the permute layer
perm_order = [0, 2, 1]  # Adjust this based on your specific needs

# Add permute layer after the first output
permute_layer1 = helper.make_node(
    'Transpose',
    [output1_name],
    ['permuted_911'],
    perm=perm_order
)

model.graph.node.extend([permute_layer1])

# Add permute layer after the second output
permute_layer2 = helper.make_node(
    'Transpose',
    [output2_name],
    ['permuted_903'],
    perm=perm_order
)

model.graph.node.extend([permute_layer2])

# Update the graph to use the permuted output names
model.graph.output[0].name = 'permuted_911'
model.graph.output[1].name = 'permuted_903'

# Save the modified model
output_model_path = f"{model_name}_permuted_output.onnx"
onnx.save(model, output_model_path)
