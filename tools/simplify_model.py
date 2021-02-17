#!/usr/bin/env python
# coding: utf-8

import onnx
from onnx import numpy_helper, helper
import numpy as np
from copy import deepcopy
from onnxsim import simplify


def matmul_to_conv2d(node, init_dict):
    weight_name = node.input[1]
    weight_tensor = init_dict[weight_name]
    weight = numpy_helper.to_array(weight_tensor)
    weight = np.expand_dims(weight.transpose(1,0),[2,3])
    weight_tensor = numpy_helper.from_array(weight, name=weight_name)
    init_dict[weight_name] = weight_tensor

def delete_init(model):
    init_len = len(model.graph.initializer)
    for i in range(init_len):
        model.graph.initializer.pop()

def convert_input_nhwc_nchw(model):
    batch_dim = 1
    dim_list = [dim_val.dim_value for dim_val in model.graph.input[0].type.tensor_type.shape.dim]
    dim_list.insert(0, batch_dim)
    dim_list = np.array(dim_list)[[0,3,1,2]]
   
    input_node = onnx.helper.make_tensor_value_info('input.1', \
                                                    onnx.TensorProto.FLOAT, dim_list.tolist())
    model.graph.input.pop()
    model.graph.input.append(input_node)
    
    dim_list = [dim_val.dim_value for dim_val in model.graph.output[0].type.tensor_type.shape.dim]
    dim_list.insert(0, batch_dim)
    dim_list.insert(2, 1)
    dim_list = np.array(dim_list)[[0,3,1,2]]
   
    out_node = onnx.helper.make_tensor_value_info(model.graph.output[0].name, \
                                                  onnx.TensorProto.FLOAT, dim_list.tolist())
    model.graph.output.pop()
    model.graph.output.append(out_node)
    
def reducemax_to_maxpool(node, model):
    node = helper.make_node(op_type="MaxPool", inputs=node.input, \
                            outputs=node.output, name=node.name,  \
                            ceil_mode = 0, kernel_shape = [1,20], \
                            pads = [0,0,0,0], strides=[1,1])
    model.graph.node.append(node)

def convert_tile(node, init_dict):
    arr_name = node.input[1]
    arr = np.array([1,1,1,20],np.int64)
    tensor = numpy_helper.from_array(arr, name=arr_name)
    init_dict[arr_name] = tensor

def simplify_model(model_path):
    model = onnx.load(model_path)
    if model is None:
        print("File %s is not find! "%model_path)
    return simplify(model)

def simplify_pfe_rpn_model(pfe_model_path, rpn_model_path):
    
    model, check = simplify_model(pfe_model_path)
    if not check:
        print("[ERROR]:Simplify %s error!"% pfe_model_path)    
    onnx.save(model, pfe_model_path)

    model, check = simplify_model(rpn_model_path)
    if not check:
        print("[ERROR]:Simplify %s error!"% rpn_model_path)    
    onnx.save(model, rpn_model_path)



if __name__ == "__main__":

    pfe_model_path = "./onnx_model/pfe.onnx"
    pfe_model_save_path = "./onnx_model/pfe_sim.onnx"

    rpn_model_path = "./onnx_model/rpn.onnx"
  
    simplify_pfe_rpn_model(pfe_model_path, rpn_model_path)
    
    # modify pfe model
    model = onnx.load(pfe_model_path)
    init_dict = {}
    for init_node in model.graph.initializer:
        init_dict[init_node.name] = init_node
    
    #delete nodes
    delete_dict = {}
    for node in model.graph.node:
        if node.op_type in {"Transpose", "Expand", "Squeeze"}:
            delete_dict[node.output[0]] = node

    val_len = len(model.graph.value_info)
    for idx in range(val_len):
        model.graph.value_info.pop()

    delete_init(model)

    matmul_weight_name = []
    convert_input_nhwc_nchw(model)
    rm_list = []
    for node in model.graph.node:
        
        # convert MatMul to Conv2D
        if node.op_type == "MatMul":
            node.op_type = "Conv"
            matmul_to_conv2d(node, init_dict)
        
        if node.input[0] in delete_dict.keys():
            node.input[0] = delete_dict[node.input[0]].input[0]
        
        # convert ReduceMax to maxpool
        if node.op_type == "ReduceMax":
            rm_list.append(node)
            reducemax_to_maxpool(node, model)
        if node.op_type == "Tile":
            convert_tile(node, init_dict)
        if node.op_type == "Concat":
            node.attribute[0].i = 1
    for node in model.graph.output:
        if node.name in delete_dict.keys():
            node.name = delete_dict[node.name].input[0]

    for name,tensor in init_dict.items():
        model.graph.initializer.append(tensor)
        
    for keys,node in delete_dict.items():
        model.graph.node.remove(node)

    for node in rm_list:
        model.graph.node.remove(node)

    onnx.save(model, pfe_model_save_path)
    print("Done")