/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/time.h>
#include <chrono>

const std::string gSampleName = "TensorRT.sample_onnx_centerpoint";

int64_t getCurrentTime()
{    
    struct timeval tv;    
    gettimeofday(&tv, NULL);   
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;    
}    


class SampleOnnxMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }


    sample::gLogInfo << "getNbInputs: " << network->getNbInputs() << " \n" << std::endl;
    sample::gLogInfo << "getNbOutputs: " << network->getNbOutputs() << " \n" << std::endl;
    sample::gLogInfo << "getNbOutputs Name: " << network->getOutput(0)->getName() << " \n" << std::endl;

    mInputDims = network->getInput(0)->getDimensions();
    
    mOutputDims = network->getOutput(0)->getDimensions();

  

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{   
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    config->setMaxWorkspaceSize(16_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }


    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    // assert(mParams.inputTensorNames.size() == 2);
    if (!processInput(buffers))
    {
        return false;
    }
    double totalDuration = 0;
    int runTimes = 100;
    for(int idx=0; idx < runTimes; idx++){
        auto startTime = std::chrono::high_resolution_clock::now();

        // Memcpy from host input buffers to device input buffers
        buffers.copyInputToDevice();

        bool status = context->executeV2(buffers.getDeviceBindings().data());
        if (!status)
        {
            return false;
        }
        
        // Memcpy from device output buffers to host output buffers
        buffers.copyOutputToHost();
        auto endTime = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
        totalDuration += duration;
        std::cout << "Inference Time: " << duration << " ms"<< std::endl;
    }
    std::cout << "Average Inference Time: " << totalDuration / runTimes << " ms"<< std::endl;
    
    verifyOutput(buffers);


    return true;
}

bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSizeN= mOutputDims.d[0];
    const int outputSizeC= mOutputDims.d[1];
    const int outputSizeH= mOutputDims.d[2];
    const int outputSizeW= mOutputDims.d[3];

    float* output = static_cast<float*>(buffers.getHostBuffer("594"));
    
    //Save one out node
    fstream file("../"+mParams.dataDirs[0]+"594.bin", ios::out | ios::binary);
    if (!file)
    {
        cout << "Error opening file.";
        return false;
    }
    file.write(reinterpret_cast<char *>(output), outputSizeN*outputSizeC*outputSizeH*outputSizeW*sizeof(float));
    file.close ();
     
}


bool readBinFile(std::string filename, void* bufPtr)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
        std::cout << "[Error] open file " << filename << " failed" << std::endl;
        return false;
    }
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // read the data:
    file.read((char*) bufPtr, fileSize);
    file.close();
    return true;
}
//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{

    float* hostPillars = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    int32_t* hostIndex = static_cast<int32_t*>(buffers.getHostBuffer(mParams.inputTensorNames[1]));

    bool ret = readBinFile("../"+mParams.dataDirs[0]+"pillars.bin", static_cast<void*>(hostPillars));
    if(!ret){
        std::cout << "[Error] open input file failed" << std::endl;
        return ret;
    }

    ret = readBinFile("../"+mParams.dataDirs[0]+"idx.bin", static_cast<void*>(hostIndex));
    if(!ret){
        std::cout << "[Error] open input file failed" << std::endl;
        return ret;
    }

    return ret;
}


//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/centerpoint/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "pointpillars_trt.onnx";
    params.inputTensorNames.push_back("input.1");
    params.inputTensorNames.push_back("indices_input");

    params.dlaCore = args.useDLACore;
    params.fp16 = true;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ././centerpoint --fp16 [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleOnnxMNIST sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for CenterPoint" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
