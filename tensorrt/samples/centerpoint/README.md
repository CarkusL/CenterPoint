# Centerpoint Pointpillars For TensorRT From ONNX

## Running the sample

1. Generate the onnx model ```pointpillars_trt.onnx```  for tensrorrt.
```
git lfs clone https://github.com/CarkusL/CenterPoint.git
python tools/merge_pfe_rpn_model.py
```
2. Prepare the tensorrt environment. I use the [nvcr.io/nvidia/tensorrt:21.02-py3](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/running.html) image.
	```
	docker pull nvcr.io/nvidia/tensorrt:21.02-py3
	docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/tensorrt:21.02-py3 /bin/bash
	```
3. (optinal) Generate the tensorrt input file by dump_input.ipynb
4. Copy these code and data to Tensorrt root folder. If you use the docker image in Step 1, the folder should be ```/usr/src/tensorrt/```.
5. Compile this sample by running `make` in the `<TensorRT root directory>/samples/centerpoint` directory. The binary named `centerpoint` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/centerpoint
	make
	```

	Where `<TensorRT root directory>` is where you installed TensorRT.

6.  Run the sample to build and run the MNIST engine from the ONNX model.
	```
	./centerpoint
	```

7.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_onnx_centerpoint # ./centerpoint
	[04/24/2021-07:34:51] [I] Building and running a GPU inference engine for CenterPoint
	[04/24/2021-07:34:58] [I] [TRT] ----------------------------------------------------------------
	[04/24/2021-07:34:58] [I] [TRT] Input filename:   ../data/centerpoint/pointpillars_trt.onnx
	[04/24/2021-07:34:58] [I] [TRT] ONNX IR version:  0.0.6
	[04/24/2021-07:34:58] [I] [TRT] Opset version:    11
	[04/24/2021-07:34:58] [I] [TRT] Producer name:    pytorch
	[04/24/2021-07:34:58] [I] [TRT] Producer version: 1.7
	[04/24/2021-07:34:58] [I] [TRT] Domain:           
	[04/24/2021-07:34:58] [I] [TRT] Model version:    0
	[04/24/2021-07:34:58] [I] [TRT] Doc string:       
	[04/24/2021-07:34:58] [I] [TRT] ----------------------------------------------------------------
	[04/24/2021-07:34:59] [W] [TRT] /home/jenkins/workspace/OSS/L0_MergeRequest/oss/parsers/onnx/onnx2trt_utils.cpp:226: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
	[04/24/2021-07:34:59] [I] [TRT] No importer registered for op: ScatterND. Attempting to import as plugin.
	[04/24/2021-07:34:59] [I] [TRT] Searching for plugin: ScatterND, plugin_version: 1, plugin_namespace: 
	[04/24/2021-07:34:59] [I] [TRT] Successfully created plugin: ScatterND
	[04/24/2021-07:35:00] [I] [TRT] Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
	[04/24/2021-07:35:12] [I] [TRT] Detected 2 inputs and 36 output network tensors.
	[04/24/2021-07:35:13] [I] getNbInputs: 2 

	[04/24/2021-07:35:13] [I] getNbOutputs: 36 

	[04/24/2021-07:35:13] [I] getNbOutputs Name: 594 

	Inference Time: 7 ms
	&&&& PASSED TensorRT.sample_onnx_centerpoint # ./centerpoint
		This output shows that the sample ran successfully; PASSED.
8. This sample just save one output node ```node name:549``` now.
9. Use dump_input.ipynb to compare the result of tensorrt with pytorch
