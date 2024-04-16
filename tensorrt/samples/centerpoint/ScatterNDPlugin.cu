// see doc: https://developer.nvidia.com/docs/drive/drive-os/6.0.9.1/public/drive-os-tensorrt/api-reference/docs/cpp/classnvinfer1_1_1_i_plugin_v2.html#a6a9cd7a410494f90b527a413adc84ce4

#include <iostream> // For std::cerr and std::endl

#include "ScatterNDPlugin.h"
#include <cassert>
#include <cstring> // for memcpy
#include "cuda_runtime.h"
#include "cuda_fp16.h"

// Define these in your .h or .cu file
#define SCATTERND_PLUGIN_NAME "ScatterND"
#define SCATTERND_PLUGIN_VERSION "1"
#define THREAD_NUM 1024  // Ensure this is set before using

namespace
{
template <typename T>
void write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void read(const char*& buffer, T& val)
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}
} // anonymous namespace

template <typename T>
T readFromBuffer(const char*& buffer) {
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

namespace nvinfer1
{
namespace plugin
{
ScatterNDPlugin::ScatterNDPlugin(const std::string& name, const size_t outputShape[], const size_t inputShape[], DataType type)
: mLayerName(name), mDataType(type)
{
    mOutputSize[0] = outputShape[0];
    mOutputSize[1] = outputShape[1];
    mInputIndexSize[0] = inputShape[0];
    mInputIndexSize[1] = inputShape[1];
}

ScatterNDPlugin::ScatterNDPlugin(const std::string& name, const void* data, size_t length)
: mLayerName(name)
{
    const char* d = reinterpret_cast<const char*>(data);
    mDataType = readFromBuffer<DataType>(d);
    mOutputSize[0] = readFromBuffer<size_t>(d);
    mOutputSize[1] = readFromBuffer<size_t>(d);
    mInputIndexSize[0] = readFromBuffer<size_t>(d);
    mInputIndexSize[1] = readFromBuffer<size_t>(d);
}

int ScatterNDPlugin::getNbOutputs() const TRT_NOEXCEPT
{
    return 1;
}

Dims ScatterNDPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
{
    assert(index == 0);
    assert(nbInputDims == 2);
    return Dims3(inputs[0].d[0], inputs[0].d[1], 1); // Example modification
}

int ScatterNDPlugin::initialize() TRT_NOEXCEPT
{
    return 0;
}

void ScatterNDPlugin::terminate() TRT_NOEXCEPT
{
}

size_t ScatterNDPlugin::getWorkspaceSize(int) const TRT_NOEXCEPT
{
    return 0;
}

// DataType ScatterNDPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override {
//     return inputTypes[index];  // Assuming the return type is based on the input type at the same index
// }

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

int ScatterNDPlugin::enqueue(
        int32_t batchSize, 
        void const *const * inputs, 
        void *const * outputs, 
        void * workspace, 
        cudaStream_t stream
    ) TRT_NOEXCEPT
{
    int channel_num = mOutputSize[1];
    int max_index_num = mInputIndexSize[0];
    int totalElems = mOutputSize[0] * channel_num;

    dim3 blockSize(THREAD_NUM);
    dim3 gridsize((max_index_num + blockSize.x - 1) / blockSize.x);

    switch (mDataType) {
    case nvinfer1::DataType::kFLOAT:
        cudaMemset(outputs[0], 0, totalElems * sizeof(float));
        _ScatterNDKernel<float><<<gridsize, blockSize, 0, stream>>>((float const*) inputs[2], (int32_t const*) inputs[1], (float*) outputs[0], channel_num, max_index_num);
        break;
    case nvinfer1::DataType::kHALF:
        cudaMemset(outputs[0], 0, totalElems * sizeof(__half));
        _ScatterNDKernel<__half><<<gridsize, blockSize, 0, stream>>>((__half const*) inputs[2], (int32_t const*) inputs[1], (__half*) outputs[0], channel_num, max_index_num);
        break;
    default:
        std::cerr << "[ERROR]: Unsupported data type!" << std::endl;
        return -1;
    }
    return 0;
}

bool ScatterNDPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT
{
    // Ensure the tensor format is kLINEAR, which is required.
    if (inOut[pos].format != TensorFormat::kLINEAR) {
        return false;
    }
    // Check if the data type is one of the supported formats.
    switch (inOut[pos].type) {
        case DataType::kFLOAT:
        case DataType::kINT32:
        case DataType::kHALF:
            return true;
        default:
            return false;
    }
}

void ScatterNDPlugin::serialize(void* buffer) const TRT_NOEXCEPT
{
    char* d = static_cast<char*>(buffer);
    write(d, mDataType);
    write(d, mOutputSize[0]);
    write(d, mOutputSize[1]);
    write(d, mInputIndexSize[0]);
    write(d, mInputIndexSize[1]);
}

size_t ScatterNDPlugin::getSerializationSize() const TRT_NOEXCEPT
{
    return sizeof(mDataType) + 4 * sizeof(size_t);
}

bool ScatterNDPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
{
  return false;
}

bool ScatterNDPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
{
  return false;
}

void ScatterNDPlugin::configurePlugin(const PluginTensorDesc* in, int32_t nbInput, const PluginTensorDesc* out, int32_t nbOutput) noexcept {
    // Check the number of inputs and outputs first to avoid accessing out of bounds
    if (nbInput > 1 && nbOutput > 0) {
        // Configure internal buffer sizes based on the input and output tensor dimensions
        mOutputSize[0] = out[0].dims.d[0];
        mOutputSize[1] = out[0].dims.d[1];
        mInputIndexSize[0] = in[1].dims.d[0];
        mInputIndexSize[1] = in[1].dims.d[1];
    }
}

const char* ScatterNDPlugin::getPluginType() const TRT_NOEXCEPT
{
    return SCATTERND_PLUGIN_NAME;
}

const char* ScatterNDPlugin::getPluginVersion() const TRT_NOEXCEPT
{
    return SCATTERND_PLUGIN_VERSION;
}

void ScatterNDPlugin::destroy() TRT_NOEXCEPT
{
    delete this;
}

IPluginV2Ext* ScatterNDPlugin::clone() const TRT_NOEXCEPT {
    ScatterNDPlugin* clonedPlugin = new ScatterNDPlugin(mLayerName, mOutputSize, mInputIndexSize, mDataType);
    clonedPlugin->setPluginNamespace(mNamespace.c_str());
    return clonedPlugin;  // Cast is not needed if ScatterNDPlugin is derived from IPluginV2Ext
}

void ScatterNDPlugin::setPluginNamespace(const char* libNamespace) TRT_NOEXCEPT
{
    mNamespace = libNamespace;
}

const char* ScatterNDPlugin::getPluginNamespace() const TRT_NOEXCEPT
{
    return mNamespace.c_str();
}

ScatterNDSamplePluginCreator::ScatterNDSamplePluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("output_shape", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("index_shape", nullptr, PluginFieldType::kINT32, 2));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ScatterNDSamplePluginCreator::getPluginName() const TRT_NOEXCEPT
{
    return SCATTERND_PLUGIN_NAME;
}

const char* ScatterNDSamplePluginCreator::getPluginVersion() const TRT_NOEXCEPT
{
    return SCATTERND_PLUGIN_VERSION;
}

const PluginFieldCollection* ScatterNDSamplePluginCreator::getFieldNames() TRT_NOEXCEPT
{
    return &mFC;
}

IPluginV2Ext* ScatterNDSamplePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT
{
    const PluginField* fields = fc->fields;
    size_t outputShapeArray[2] = {0, 0};
    size_t indexShapeArray[2] = {0, 0};
    DataType dataType = DataType::kFLOAT;  // Default data type

    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string fieldName(fields[i].name);
        if (fieldName == "output_shape")
        {
            const int32_t* shape = static_cast<const int32_t*>(fields[i].data);
            outputShapeArray[0] = shape[0];
            outputShapeArray[1] = shape[1];
        }
        else if (fieldName == "index_shape")
        {
            const int32_t* shape = static_cast<const int32_t*>(fields[i].data);
            indexShapeArray[0] = shape[0];
            indexShapeArray[1] = shape[1];
        }
    }

    ScatterNDPlugin* plugin = new ScatterNDPlugin(name, outputShapeArray, indexShapeArray, dataType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void ScatterNDSamplePluginCreator::setPluginNamespace(const char* libNamespace) TRT_NOEXCEPT
{
    mNamespace = libNamespace;
}

const char* ScatterNDSamplePluginCreator::getPluginNamespace() const TRT_NOEXCEPT
{
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(ScatterNDSamplePluginCreator);

} // namespace plugin
} // namespace nvinfer1
