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
#include <glob.h>
#include <sstream>

#include "preprocess.h"
#include "postprocess.h"

const std::string gSampleName = "TensorRT.sample_onnx_centerpoint";

int64_t getCurrentTime()
{    
    struct timeval tv;    
    gettimeofday(&tv, NULL);   
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;    
}    


std::vector<std::string> glob(const std::string pattern)
{
    std::vector<std::string> filenames;
    using namespace std;
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0){
        globfree(&glob_result);
        return filenames;
    }
    for(auto idx =0; idx <glob_result.gl_pathc; idx++){
        filenames.push_back(string(glob_result.gl_pathv[idx]));

    }
    globfree(&glob_result);
    return filenames;
}


class SampleCenterPoint
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleCenterPoint(const samplesCommon::OnnxSampleParams& params)
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
    bool processInput(void*& points, std::string& pointFilePath, int& pointNum);

    //!
    //! \brief Classifies digits and verify result
    //!
    void saveOutput(std::vector<Box>& predResult, std::string& inputFileName);
    bool testFun(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleCenterPoint::build()
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
bool SampleCenterPoint::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{   
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    config->setMaxWorkspaceSize(1_GiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}
bool SampleCenterPoint::testFun(const samplesCommon::BufferManager& buffers){
    
    size_t num = 38;
    for (size_t idx = 0; idx < num; idx++){
        sample::gLogInfo << "idx:" << idx << std::endl;
        sample::gLogInfo << "num:" << num << std::endl;
        sample::gLogInfo << "compare :" << (num>idx) << std::endl;
    }

}
//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleCenterPoint::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    float* hostPillars = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    int32_t* hostIndex = static_cast<int32_t*>(buffers.getHostBuffer(mParams.inputTensorNames[1]));    
    
    void* inputPointBuf = nullptr;

    std::vector<std::string> filePath = glob("../"+mParams.dataDirs[0]+"/points/*.bin");
    
    for(auto idx = 0; idx < filePath.size(); idx++){
        std::cout << "filePath[idx]: " << filePath[idx] << std::endl;
        int pointNum = 0;
        if (!processInput(inputPointBuf, filePath[idx], pointNum))
        {
            return false;
        }
        
        float* points = static_cast<float*>(inputPointBuf);
    
        std::vector<Box> predResult;

        auto startTime = std::chrono::high_resolution_clock::now();
        preprocess(points, hostPillars, hostIndex, pointNum);
        auto endTime = std::chrono::high_resolution_clock::now();
        double preprocessDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
        
        startTime = std::chrono::high_resolution_clock::now();
        // Memcpy from host input buffers to device input buffers
        buffers.copyInputToDevice();
        
        bool status = context->executeV2(buffers.getDeviceBindings().data());
        if (!status)
        {
            return false;
        }

        // Memcpy from device output buffers to host output buffers
        buffers.copyOutputToHost();
        endTime = std::chrono::high_resolution_clock::now();
        
        double inferenceDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
        
        startTime = std::chrono::high_resolution_clock::now();
        predResult.clear();
        postprocess(buffers, predResult);
        endTime = std::chrono::high_resolution_clock::now();
        double PostProcessDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
        
        sample::gLogInfo << "PreProcess Time: " << preprocessDuration << " ms"<< std::endl;
        sample::gLogInfo << "inferenceDuration Time: " << inferenceDuration << " ms"<< std::endl;
        sample::gLogInfo << "PostProcessDuration Time: " << PostProcessDuration << " ms"<< std::endl;

        saveOutput(predResult, filePath[idx]);

        free(points);  
    }
    
    return true;
}

/* There is a bug. 
 * If I change void to bool, the "for (size_t idx = 0; idx < mEngine->getNbBindings(); idx++)" loop will not stop.
 */
void SampleCenterPoint::saveOutput(std::vector<Box>& predResult, std::string& inputFileName)
{
    
    std::string::size_type pos = inputFileName.find_last_of("/");
    std::string outputFilePath("../"+mParams.dataDirs[0]+"/results/"+ inputFileName.substr(pos) + ".txt");

    ofstream resultFile;

    resultFile.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    try {

        resultFile.open(outputFilePath);
        for (size_t idx = 0; idx < predResult.size(); idx++){
                resultFile << predResult[idx].x << " " << predResult[idx].y << " " << predResult[idx].z << " "<< \
                predResult[idx].l << " " << predResult[idx].h << " " << predResult[idx].w << " " << predResult[idx].velX \
                << " " << predResult[idx].velY << " " << predResult[idx].theta << " " << predResult[idx].score << \ 
                " "<< predResult[idx].cls << std::endl;
        }
        resultFile.close();
    }
    catch (std::ifstream::failure e) {
        sample::gLogError << "Open File: " << outputFilePath << " Falied"<< std::endl;
    }
}


//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleCenterPoint::processInput(void*& inputPointBuf, std::string& pointFilePath, int& pointNum)
{

    bool ret = readBinFile(pointFilePath, inputPointBuf, pointNum);
    if(!ret){
        sample::gLogError << "Error read point file: " << pointFilePath<< std::endl;
        free(inputPointBuf);
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

    SampleCenterPoint sample(initializeSampleParams(args));

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
