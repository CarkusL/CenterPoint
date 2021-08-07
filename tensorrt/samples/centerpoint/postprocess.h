#ifndef __CENTERPOINT_POSTPROCESS__
#define __CENTERPOINT_POSTPROCESS__

#include "buffers.h"
#include "common.h"
#include "config.h"

struct Box{
    float x;
    float y;
    float z;
    float l;
    float h;
    float w;
    float velX;
    float velY;
    float theta;

    float score;
    int cls;
    bool isDrop; // for nms
};

void postprocess(const samplesCommon::BufferManager& buffers, std::vector<Box>& predResult);
#endif