#ifndef BATCHTILEPLUGIN_H
#define BATCHTILEPLUGIN_H
#include "NvInferPlugin.h"
#include <string>
#include <vector>
namespace nvinfer1
{
namespace plugin
{
class ScatterNDPlugin : public IPluginV2Ext
{
public:
    ScatterNDPlugin(const std::string name, const size_t mOutputSizeAttr[], const size_t inputShapeAttr[], const DataType type);

    ScatterNDPlugin(const std::string name, const void* data, size_t length);

    ScatterNDPlugin() = delete;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;
    void terminate() override;

    size_t getWorkspaceSize(int) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;

private:
    const std::string mLayerName;
    size_t mCopySize;
    std::string mNamespace;
    DataType mDataType;
    size_t mOutputSize[2]; // [H*W, C]
    size_t mInputIndexSize[2]; // [H*W, C]

};

class ScatterNDSamplePluginCreator : public IPluginCreator
{
public:
    ScatterNDSamplePluginCreator();

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* libNamespace) override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mNamespace.c_str();
    }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
    DataType mDataType;
};

} // namespace plugin
} // namespace nvinfer1

#endif