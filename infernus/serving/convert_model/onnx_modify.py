import sys
import os
import onnx


if __name__ == "__main__":


    temp_onnx_model = str(sys.argv[1])
    batch_size = int(sys.argv[2])
    model_name = str(sys.argv[3])
    #add optional index argument
    if len(sys.argv) > 4:
        idx = str(sys.argv[4])
    else:
        idx = None

    print(f"temp_onnx_model: {temp_onnx_model}")
    # Set new fixed batch size to ONNX model
    onnx_model = onnx.load_model(temp_onnx_model)

    inputs = onnx_model.graph.input
    #get input name
    input_name = inputs[0].name

    #get output name
    outputs = onnx_model.graph.output

    print("outputs:", outputs)

    output_name = outputs[0].name
    print(outputs[0].type.tensor_type.shape)
    #assumes all outputs (if there are multiple) have the same shape
    output_shape = outputs[0].type.tensor_type.shape.dim.pop().dim_value
    print(f"output_shape: {output_shape}")

    print(f"input_name: {input_name}")
    print(f"output_name: {output_name}")

    for i in inputs:
        dim1 = i.type.tensor_type.shape.dim[0]
        dim1.dim_value = batch_size

    print(f"saving modified model to {model_name}")

    onnx.save_model(onnx_model, os.path.join(model_name, "1/model.onnx"))
    
    #get ifo from onnx model

    for i in inputs:
        print("input name",i.name)

    if len(inputs) == 2:
        name = "hl"
    
    elif len(inputs) == 3:
        name = "full"
        print("Full model building")
    
    else:
        name = inputs[0].name

    if idx:
        name += "_"+idx
        print("adding index")

    f = open(os.path.join(model_name,"config.pbtxt"), "w")
    f.write("name: " + "\"model_" + name + "\"" + "\n")
    f.write("platform: " + "\"" + "onnxruntime_onnx" + "\"" + "\n")
    f.write("max_batch_size : 0 \n")
    for i in inputs:
        f.write("input [" + "\n")
        f.write("  {" + "\n")
        f.write("    name: " + "\"" + i.name + "\"" + "\n")
        f.write("    data_type: TYPE_FP32" + "\n")
        if i.type.tensor_type.HasField("shape"):
            print(i.type.tensor_type.shape)
            if len(i.type.tensor_type.shape.dim) == 3:
                f.write("    dims: [" + str(batch_size) + ", " + str(i.type.tensor_type.shape.dim[1].dim_value) + ", " + str(i.type.tensor_type.shape.dim[2].dim_value) + "]" + "\n")
            elif len(i.type.tensor_type.shape.dim) == 2:
                f.write("    dims: [" + str(batch_size) + ", " + str(i.type.tensor_type.shape.dim[1].dim_value) + "]" + "\n")
        #f.write("    dims: [" + str(batch_size) + ", " + str(2048) + ", 1]" + "\n")
        f.write("  }" + "\n")
        f.write("]" + "\n")
    for i in outputs:
        f.write("output [" + "\n")
        f.write("  {" + "\n")
        f.write("    name: " + "\"" + str(i.name) + "\"" + "\n")
        f.write("    data_type: TYPE_FP32" + "\n")
        f.write("    dims: [" + str(batch_size) + ", " + str(output_shape) + "]" + "\n")
        f.write("  }" + "\n")
        f.write("]" + "\n")
    f.write("""instance_group [
    {
        count: 6
        kind: KIND_GPU
    }
]""")
    f.write("\n")
    f.write('parameters { key: "enable_mem_arena" value: { string_value: "1" } }')
    f.write("\n")
    f.write('parameters { key: "arena_extend_strategy" value: { string_value: "1" } }')
    f.write("\n")
    f.write('parameters { key: "intra_op_thread_count" value: { string_value: "2" } }')
    f.write("\n")
    f.write('parameters { key: "inter_op_thread_count" value: { string_value: "2" } }')
    f.close()