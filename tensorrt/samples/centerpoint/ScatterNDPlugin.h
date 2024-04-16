#ifndef SCATTER_ND_PLUGIN_H
#define SCATTER_ND_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

namespace nvinfer1
{
namespace plugin
{
class ScatterNDPlugin : public IPluginV2IOExt
{
public:
    ScatterNDPlugin(const std::string& name, const size_t outputShape[], const size_t inputShape[], DataType type);
    ScatterNDPlugin(const std::string& name, const void* data, size_t length);
    // ScatterNDPlugin() = delete;
    int getNbOutputs() const TRT_NOEXCEPT override;
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT override;
    int initialize() TRT_NOEXCEPT override;
    void terminate() TRT_NOEXCEPT override;
    size_t getWorkspaceSize(int) const TRT_NOEXCEPT override;
    int enqueue(int32_t batchSize, void const *const * inputs, void *const * outputs, void * workspace, cudaStream_t stream) noexcept override;
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override;
    size_t getSerializationSize() const TRT_NOEXCEPT override;
    void serialize(void* buffer) const TRT_NOEXCEPT override;
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT override;
    bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;
    void configurePlugin(const PluginTensorDesc* in, int32_t nbInput, const PluginTensorDesc* out, int32_t nbOutput) noexcept override;
    const char* getPluginType() const TRT_NOEXCEPT override;
    const char* getPluginVersion() const TRT_NOEXCEPT override;
    void destroy() TRT_NOEXCEPT override;
    IPluginV2Ext* clone() const noexcept override;
    //IPluginV2* clone() const noexcept override;
    void setPluginNamespace(const char* libNamespace) TRT_NOEXCEPT override;
    const char* getPluginNamespace() const TRT_NOEXCEPT override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT override;

private:
    std::string mLayerName;
    size_t mOutputSize[2];  // [H*W, C]
    size_t mInputIndexSize[2];  // [H*W, C]
    DataType mDataType;
    std::string mNamespace;
    // std::map<nvinfer1::DataType, bool> supportedFormats;
};

class ScatterNDSamplePluginCreator : public IPluginCreator
{
public:
    ScatterNDSamplePluginCreator();
    const char* getPluginName() const TRT_NOEXCEPT override;
    const char* getPluginVersion() const TRT_NOEXCEPT override;
    const PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;
    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT override;
    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* libNamespace) TRT_NOEXCEPT override;
    const char* getPluginNamespace() const TRT_NOEXCEPT override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
    DataType mDataType;
};

} // namespace plugin
} // namespace nvinfer1

#endif // SCATTER_ND_PLUGIN_H

