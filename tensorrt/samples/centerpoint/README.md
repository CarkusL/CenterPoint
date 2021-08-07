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
3. (optinal)  Generate the tensorrt input file by ```TensorRT_Visualize.ipynb``` to get input points.bin in ```CenterPoint/tensorrt/data/centerpoint```
4. Copy these ```CenterPoint/tensorrt/sample/centerpoint``` and ```CenterPoint/tensorrt/data/centerpoint``` to Tensorrt root folder. If you use the docker image in Step 1, the folder should be ```/usr/src/tensorrt/```.
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
	[08/07/2021-08:26:47] [I] Building and running a GPU inference engine for CenterPoint
	[08/07/2021-08:26:55] [I] [TRT] ----------------------------------------------------------------
	[08/07/2021-08:26:55] [I] [TRT] Input filename:   ../data/centerpoint/pointpillars_trt.onnx
	[08/07/2021-08:26:55] [I] [TRT] ONNX IR version:  0.0.6
	[08/07/2021-08:26:55] [I] [TRT] Opset version:    11
	[08/07/2021-08:26:55] [I] [TRT] Producer name:    pytorch
	[08/07/2021-08:26:55] [I] [TRT] Producer version: 1.7
	[08/07/2021-08:26:55] [I] [TRT] Domain:           
	[08/07/2021-08:26:55] [I] [TRT] Model version:    0
	[08/07/2021-08:26:55] [I] [TRT] Doc string:       
	[08/07/2021-08:26:55] [I] [TRT] ----------------------------------------------------------------
	[08/07/2021-08:26:55] [W] [TRT] /home/jenkins/workspace/OSS/L0_MergeRequest/oss/parsers/onnx/onnx2trt_utils.cpp:226: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
	[08/07/2021-08:26:55] [I] [TRT] No importer registered for op: ScatterND. Attempting to import as plugin.
	[08/07/2021-08:26:55] [I] [TRT] Searching for plugin: ScatterND, plugin_version: 1, plugin_namespace: 
	[08/07/2021-08:26:55] [I] [TRT] Successfully created plugin: ScatterND
	[08/07/2021-08:26:55] [W] [TRT] Tensor DataType is determined at build time for tensors not marked as input or output.
	[08/07/2021-08:26:55] [W] [TRT] Tensor DataType is determined at build time for tensors not marked as input or output.
	[08/07/2021-08:26:55] [W] [TRT] Tensor DataType is determined at build time for tensors not marked as input or output.
	[08/07/2021-08:26:55] [W] [TRT] Tensor DataType is determined at build time for tensors not marked as input or output.
	[08/07/2021-08:26:55] [W] [TRT] Tensor DataType is determined at build time for tensors not marked as input or output.
	[08/07/2021-08:26:55] [W] [TRT] Tensor DataType is determined at build time for tensors not marked as input or output.
	[08/07/2021-08:27:02] [I] [TRT] Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
	[08/07/2021-08:27:15] [I] [TRT] Detected 2 inputs and 42 output network tensors.
	[08/07/2021-08:27:15] [I] getNbInputs: 2 

	[08/07/2021-08:27:15] [I] getNbOutputs: 42 

	[08/07/2021-08:27:15] [I] getNbOutputs Name: 594 

	filePath[idx]: ../data/centerpoint//points/048a45dd2cf54aa5808d8ccc85731d44.bin
	[08/07/2021-08:27:15] [I] [INFO] pointNum : 278690
	[08/07/2021-08:27:15] [I] PreProcess Time: 11.9223 ms
	[08/07/2021-08:27:15] [I] inferenceDuration Time: 7.4477 ms
	[08/07/2021-08:27:15] [I] PostProcessDuration Time: 1.97806 ms
	filePath[idx]: ../data/centerpoint//points/06be0e3b665c44fa8d17d9f4770bdf9c.bin
	[08/07/2021-08:27:15] [I] [INFO] pointNum : 258553
	[08/07/2021-08:27:15] [I] PreProcess Time: 7.99008 ms
	[08/07/2021-08:27:15] [I] inferenceDuration Time: 6.95937 ms
	[08/07/2021-08:27:15] [I] PostProcessDuration Time: 1.21248 ms
	filePath[idx]: ../data/centerpoint//points/07fad91090c746ccaa1b2bdb55329e20.bin
	[08/07/2021-08:27:15] [I] [INFO] pointNum : 285130
	[08/07/2021-08:27:15] [I] PreProcess Time: 9.02984 ms
	[08/07/2021-08:27:15] [I] inferenceDuration Time: 6.8961 ms
	[08/07/2021-08:27:15] [I] PostProcessDuration Time: 1.38823 ms
	filePath[idx]: ../data/centerpoint//points/0a0d6b8c2e884134a3b48df43d54c36a.bin
	[08/07/2021-08:27:15] [I] [INFO] pointNum : 278272
	[08/07/2021-08:27:15] [I] PreProcess Time: 9.25149 ms
	[08/07/2021-08:27:15] [I] inferenceDuration Time: 7.06773 ms
	[08/07/2021-08:27:15] [I] PostProcessDuration Time: 1.53539 ms
	filePath[idx]: ../data/centerpoint//points/0af0feb5b1394b928dd13d648de898f5.bin
	[08/07/2021-08:27:15] [I] [INFO] pointNum : 273692
	[08/07/2021-08:27:15] [I] PreProcess Time: 9.12371 ms
	[08/07/2021-08:27:15] [I] inferenceDuration Time: 7.11943 ms
	[08/07/2021-08:27:15] [I] PostProcessDuration Time: 1.63056 ms
	filePath[idx]: ../data/centerpoint//points/0bb62a68055249e381b039bf54b0ccf8.bin
	[08/07/2021-08:27:15] [I] [INFO] pointNum : 286052
	[08/07/2021-08:27:15] [I] PreProcess Time: 10.143 ms
	[08/07/2021-08:27:15] [I] inferenceDuration Time: 7.20578 ms
	[08/07/2021-08:27:15] [I] PostProcessDuration Time: 1.82941 ms
    ...
	&&&& PASSED TensorRT.sample_onnx_centerpoint # ./centerpoint

8. copy the \<TensorRT root directory>/data/centerpoint back the CenterPoint/tensorrt/data
9. Run the ```TensorRT_Visualize.ipynb``` to do evaluation and visualiza tensorrt result.
10. Compare the [TensorRT result](../../../demo/trt_demo/file00.png) with [Pytorch result](../../../demo/torch_demo/file00.png).

|  TensoRT  | Pytroch  |
|  :----:  | :----:  |
| ![avatar](../../../demo/trt_demo/file00.png)  | ![avatar](../../../demo/torch_demo/file00.png) |

## 3D detection on nuScenes Mini dataset
TensorRT postprocess use aligned NMS on Bev, so there are some precision loss.

|         |  mAP    | mATE   | mASE   | mAOE    | mAVE   |  mAAE | NDS    |
|---------|---------|--------|--------|---------|--------|-------|------- |
| Pytorch | 0.4163  | 0.4438 | 0.4516 | 0.5674  | 0.4429 | 0.3288| 0.4847 |
| TensorRT| 0.4007  | 0.4433 | 0.4537 | 0.5665  | 0.4416 | 0.3191| 0.4779 |

