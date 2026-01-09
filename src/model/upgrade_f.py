# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "onnx",
#     "numpy",
# ]
# ///

import sys
import onnx
from onnx import TensorProto
import numpy as np

def convert_tensor_proto(tensor):
    """
    Converts a TensorProto from Float16 to Float32.
    Decodes the raw bytes, casts to float32, and re-encodes.
    """
    if tensor.data_type == TensorProto.FLOAT16:
        # Most modern ONNX models store data in raw_data
        if tensor.raw_data:
            # Decode F16 bytes to numpy array
            fp16_data = np.frombuffer(tensor.raw_data, dtype=np.float16)
            # Cast to F32
            fp32_data = fp16_data.astype(np.float32)
            # Save back as raw bytes
            tensor.raw_data = fp32_data.tobytes()
        
        # Update the data type flag
        tensor.data_type = TensorProto.FLOAT
        print(f"  - Converted tensor '{tensor.name}' to FP32")

def convert_model_to_fp32(input_path, output_path):
    print(f"Loading {input_path}...")
    try:
        model = onnx.load(input_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    graph = model.graph
    print("Converting inputs, outputs, and weights...")

    # 1. Convert Inputs
    for inp in graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT

    # 2. Convert Outputs
    for out in graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT16:
            out.type.tensor_type.elem_type = TensorProto.FLOAT

    # 3. Convert Value Info (Intermediate shapes)
    for info in graph.value_info:
        if info.type.tensor_type.elem_type == TensorProto.FLOAT16:
            info.type.tensor_type.elem_type = TensorProto.FLOAT

    # 4. Convert Initializers (Weights)
    for init in graph.initializer:
        convert_tensor_proto(init)

    # 5. Fix Nodes (Cast & Constant)
    print("Fixing graph nodes...")
    for node in graph.node:
        # Fix Cast nodes that are casting TO float16
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    print(f"  - Patching Cast node '{node.name}' target to FP32")
                    attr.i = TensorProto.FLOAT
        
        # Fix Constant nodes that might hold F16 values
        elif node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    convert_tensor_proto(attr.t)

    print(f"Saving converted model to {output_path}...")
    onnx.save(model, output_path)
    print("Done! You can now import this model in Burn.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run convert_f16_to_f32.py <input_model.onnx> <output_model.onnx>")
        sys.exit(1)
    
    convert_model_to_fp32(sys.argv[1], sys.argv[2])