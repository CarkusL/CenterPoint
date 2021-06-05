/**
 * For the usage of those member function, please refer to the
 * offical api doc.
 * https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_ext.html
 */

#include "ScatterNDPlugin.h"
#include <cassert>
#include <iostream>
#include <string.h>

#include "cuda_runtime.h"
#include "cuda_fp16.h"



// Use fp16 mode for inference
#define DATA_TYPE nvinfer1::DataType::kHALF
#define THREAD_NUM 1024

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}
// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

using namespace nvinfer1;
using nvinfer1::plugin::ScatterNDPlugin;
using nvinfer1::plugin::ScatterNDSamplePluginCreator;

static const char* SCATTERND_PLUGIN_VERSION{"1"};
static const char* SCATTERND_PLUGIN_NAME{"ScatterND"};

PluginFieldCollection ScatterNDSamplePluginCreator::mFC{};
std::vector<PluginField> ScatterNDSamplePluginCreator::mPluginAttributes;



ScatterNDPlugin::ScatterNDPlugin(const std::string name, const size_t outputShapeArray[], 
                                 const size_t indexShapeArray[], const DataType type) : mLayerName(name), mDataType(type)
{
    mOutputSize[0] = outputShapeArray[0];
    mOutputSize[1] = outputShapeArray[1];

    mInputIndexSize[0] = indexShapeArray[0];
    mInputIndexSize[1] = indexShapeArray[1];

}

ScatterNDPlugin::ScatterNDPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    const char *d = reinterpret_cast<const char *>(data);
    const char *a = d;

    mDataType = readFromBuffer<DataType>(d);
    mOutputSize[0] = readFromBuffer<size_t>(d);
    mOutputSize[1] = readFromBuffer<size_t>(d);
    mInputIndexSize[0] = readFromBuffer<size_t>(d);
    mInputIndexSize[1] = readFromBuffer<size_t>(d);

    assert(d == a + length);
}

int ScatterNDPlugin::getNbOutputs() const
{
    return 1;
}

Dims ScatterNDPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{   
    // scatterND data input 
    return Dims2(inputs[0].d[0],inputs[0].d[1]);
}

int ScatterNDPlugin::initialize()
{
    return 0;
}

size_t ScatterNDPlugin::getWorkspaceSize(int) const
{
    return 0;
}

DataType ScatterNDPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[2];
}

template <typename Dtype>
__global__ void _ScatterNDKernel(const Dtype *updata_input, const int *indicesInputPtr , Dtype* output,
        int channel_num, int max_index_num) {
    
    int idx_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx_num >= max_index_num) return;    
    
    int idx_output = indicesInputPtr[idx_num*2+1];
    if (idx_output < 0) return;
    
    for(int idx=0; idx < channel_num; idx++){
        output[idx_output*channel_num+idx] = updata_input[idx_num*channel_num+idx];
    }
}

int ScatterNDPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int channel_num = mOutputSize[1];
    int max_index_num = mInputIndexSize[0];

    int totalElems = mOutputSize[0]*channel_num;
    
    dim3 blockSize(THREAD_NUM);
    dim3 gridsize(max_index_num/blockSize.x+1);
     
    // if you want to inference use fp32, change the DATA_TYPE
    switch (mDataType)
    {
    case nvinfer1::DataType::kFLOAT:
        cudaMemset(outputs[0], 0, totalElems * sizeof(float));
        _ScatterNDKernel<<<gridsize, blockSize,0,stream>>>(static_cast<float const*> (inputs[2]), static_cast<int32_t const*> (inputs[1]), 
                                                    static_cast<float *> (outputs[0]), channel_num, max_index_num);
        break;

    case nvinfer1::DataType::kHALF:
        cudaMemset(outputs[0], 0, totalElems * sizeof(float)/2);
        _ScatterNDKernel<<<gridsize, blockSize,0,stream>>>(static_cast<int16_t const*> (inputs[2]), static_cast<int32_t const*> (inputs[1]), 
                                                           static_cast<int16_t *> (outputs[0]), channel_num, max_index_num);
        
        break;
    
    default:
        std::cout << "[ERROR]: mDataType dones't support" << std::endl;
    }
    return 0;
}

void ScatterNDPlugin::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer);
    char *a = d;
    writeToBuffer<DataType>(d, mDataType);
    writeToBuffer<size_t>(d, mOutputSize[0]);
    writeToBuffer<size_t>(d, mOutputSize[1]);
    writeToBuffer<size_t>(d, mInputIndexSize[0]);
    writeToBuffer<size_t>(d, mInputIndexSize[1]);

    assert(d == a + getSerializationSize());
}

void ScatterNDPlugin::terminate() {
}

size_t ScatterNDPlugin::getSerializationSize() const
{
    return sizeof(DataType)+ 4*sizeof(size_t);
}

bool ScatterNDPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool ScatterNDPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void ScatterNDPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    mOutputSize[0] = outputDims[0].d[0];
    mOutputSize[1] = outputDims[0].d[1];
    mInputIndexSize[0] = inputDims[1].d[0];
    mInputIndexSize[1] = inputDims[1].d[1];
}

bool ScatterNDPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    switch (type)
    {   
        case nvinfer1::DataType::kINT32: return true;
        case nvinfer1::DataType::kFLOAT: return true;
        case nvinfer1::DataType::kHALF: return true;
    }
    return false;
}

/**
 * NO NEED TO MODIFY
 */
const char* ScatterNDPlugin::getPluginType() const
{
    return SCATTERND_PLUGIN_NAME;
}

/**
 * NO NEED TO MODIFY
 */
const char* ScatterNDPlugin::getPluginVersion() const
{
    return SCATTERND_PLUGIN_VERSION;
}

void ScatterNDPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* ScatterNDPlugin::clone() const
{
    auto* plugin = new ScatterNDPlugin(mLayerName, mOutputSize, mInputIndexSize, mDataType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

/**
 * NO NEED TO MODIFY
 */
void ScatterNDPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

/**
 * NO NEED TO MODIFY
 */
const char* ScatterNDPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

ScatterNDSamplePluginCreator::ScatterNDSamplePluginCreator()
{   
    mPluginAttributes.emplace_back(PluginField("output_shape", nullptr, PluginFieldType::kINT32, 3));
    mPluginAttributes.emplace_back(PluginField("index_shape", nullptr, PluginFieldType::kINT32, 3));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

/**
 * NO NEED TO MODIFY
 */
const char* ScatterNDSamplePluginCreator::getPluginName() const
{
    return SCATTERND_PLUGIN_NAME;
}

/**
 * NO NEED TO MODIFY
 */
const char* ScatterNDSamplePluginCreator::getPluginVersion() const
{
    return SCATTERND_PLUGIN_VERSION;
}

/**
 * NO NEED TO MODIFY
 */
const PluginFieldCollection* ScatterNDSamplePluginCreator::getFieldNames()
{   
    return &mFC;
}

IPluginV2Ext* ScatterNDSamplePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    
    const nvinfer1::PluginField* fields = fc->fields;
    
    mDataType = DATA_TYPE;

    size_t indexShapeArray[2] = {0};
    size_t outputShapeArray[2] = {0};

    for (int i=0; i<fc->nbFields; i++) {
        if(!strcmp(fields[i].name, "output_shape")){
            const auto *outputShapeAttr = static_cast<const int32_t*>(fields[i].data);
            outputShapeArray[0] = outputShapeAttr[1];
            outputShapeArray[1] = outputShapeAttr[2];

        }
        if(!strcmp(fields[i].name, "index_shape")){
            const auto * indexShapeAttr = static_cast<const int32_t*>(fields[i].data);

            indexShapeArray[0] = indexShapeAttr[1];
            indexShapeArray[1] = indexShapeAttr[2];
        }
    }
    
    auto* plugin = new ScatterNDPlugin(name, outputShapeArray, indexShapeArray, mDataType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* ScatterNDSamplePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{   
    return new ScatterNDPlugin(name, serialData, serialLength);
}

REGISTER_TENSORRT_PLUGIN(ScatterNDSamplePluginCreator);
