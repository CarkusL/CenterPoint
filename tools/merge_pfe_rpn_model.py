#!/usr/bin/env python
# coding: utf-8

import onnx
from onnx import helper, numpy_helper
import numpy as np
import copy

def make_scatterND(model, rpn_input_shape, indices_shape, pfe_out_maxpool_name, batch_size, save_for_trt=False):
    output_shape = [batch_size, rpn_input_shape[2]*rpn_input_shape[3], rpn_input_shape[1]]
    
    squeeze_node = helper.make_node(op_type="Squeeze", inputs=[pfe_out_maxpool_name], \
                                    outputs=['pfe_squeeze_1'], name="pfe_Squeeze_1", \
                                    axes = [3])
    
    transpose_node_1 = helper.make_node(op_type="Transpose", inputs=['pfe_squeeze_1',], \
                                        outputs=['pfe_transpose_1'], name="pfe_Transpose_1", \
                                        perm=[0,2,1])
    
    if save_for_trt:
        scatter_node = helper.make_node(op_type="ScatterND", inputs=['scatter_data', 'indices_input', 'pfe_transpose_1'], \
                                        outputs=['scatter_1'], name="ScatterND_1", output_shape=output_shape, index_shape=indices_shape)
    else:
        scatter_node = helper.make_node(op_type="ScatterND", inputs=['scatter_data', 'indices_input', 'pfe_transpose_1'], \
                                        outputs=['scatter_1'], name="ScatterND_1")

    transpose_node_2 = helper.make_node(op_type="Transpose", inputs=['scatter_1',], \
                                        outputs=['pfe_transpose_2'], \
                                        name="pfe_Transpose_2", perm=[0,2,1])
    reshape_node = helper.make_node(op_type="Reshape", inputs=["pfe_transpose_2","pfe_reshape_shape"], \
                                    outputs=['rpn_input'], name="pfe_reshape_1")
    
    squeeze_axes = [3]
    squeeze_tensor = np.array(squeeze_axes, dtype=np.int32)
    squeeze_tensor = numpy_helper.from_array(squeeze_tensor, name="axes")
    model.graph.initializer.append(squeeze_tensor)
    
    data_shape = [batch_size, rpn_input_shape[2]*rpn_input_shape[3], rpn_input_shape[1]]
    data = np.zeros(data_shape, dtype=np.float32)
    data_tensor = numpy_helper.from_array(data, name="scatter_data")
    model.graph.initializer.append(data_tensor)
    
    reshape_shape = np.array(rpn_input_shape, dtype=np.int64)
    reshape_tensor = numpy_helper.from_array(reshape_shape, name="pfe_reshape_shape")
    model.graph.initializer.append(reshape_tensor)    

    input_node = onnx.helper.make_tensor_value_info('indices_input', onnx.TensorProto.INT64, indices_shape)
    model.graph.input.append(input_node)
    
    model.graph.node.append(squeeze_node)
    model.graph.node.append(transpose_node_1)    
    model.graph.node.append(transpose_node_2)    
    
    model.graph.node.append(scatter_node)
    model.graph.node.append(reshape_node)

if __name__ == "__main__":
    
    pfe_sim_model_path = "./onnx_model/pfe_sim.onnx"
    rpn_sim_model_path = "./onnx_model/rpn.onnx"
    pointpillars_save_path = "./onnx_model/pointpillars.onnx"
    pointpillars_trt_save_path = "./onnx_model/pointpillars_trt.onnx"

    pfe_model = onnx.load(pfe_sim_model_path)
    rpn_model = onnx.load(rpn_sim_model_path)

    batch_size = 1
    rpn_input_conv_name = "Conv_15"
    pfe_out_maxpool_name = "46"
    rpn_input_shape = [batch_size,64,512,512]
    indices_shape = [batch_size, 30000,2]


    for node in pfe_model.graph.node:
        node.name = "pfe_"+node.name
    
    # merge nodes, outputs and initializers
    pfe_model.graph.node.extend(rpn_model.graph.node)
    pfe_model.graph.output.pop()
    pfe_model.graph.output.extend(rpn_model.graph.output)
    pfe_model.graph.initializer.extend(rpn_model.graph.initializer)
    
    pfe_model_trt = copy.deepcopy(pfe_model)

    # Connect pfe and rpn with scatterND
    make_scatterND(pfe_model, rpn_input_shape, indices_shape, pfe_out_maxpool_name, batch_size)
    make_scatterND(pfe_model_trt, rpn_input_shape, indices_shape, pfe_out_maxpool_name, batch_size, save_for_trt=True)
    
    def change_input(model):
        for node in model.graph.node:
            if node.name == rpn_input_conv_name:
                node.input[0] = "rpn_input"
                break

            model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = indices_shape[1]
    
    change_input(pfe_model)
    change_input(pfe_model_trt)

    onnx.save(pfe_model, pointpillars_save_path)
    onnx.save(pfe_model_trt, pointpillars_trt_save_path)

    print("Done")