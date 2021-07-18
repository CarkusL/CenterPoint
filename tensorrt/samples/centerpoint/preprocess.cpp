#include"preprocess.h"
#include <string>
#include <sys/time.h>
#include <chrono>
#include <thread>
#include <vector>

#include "logger.h"

#define BEV_W 512
#define BEV_H 512
#define MAX_PILLARS 30000
#define MAX_PIONT_IN_PILLARS 20
#define FEATURE_NUM 10
#define X_STEP 0.2f
#define Y_STEP 0.2f
#define X_MIN -51.2
#define X_MAX 51.2f
#define Y_MIN -51.2
#define Y_MAX 51.2f
#define Z_MIN -5.0f
#define Z_MAX 3.0f
#define THREAD_NUM 2

void PreprocessWorker(float* points, float* feature, int* indices, int pointNum, int threadIdx, int pillarsPerThread){
    // 0 ~ MAX_PIONT_IN_PILLARS
    unsigned short pointCount[MAX_PILLARS] = {0};

    // 0 ~ MAX_PILLARS
    int pillarsIndices[BEV_W*BEV_H] = {0};
    for(size_t idx = 0; idx < BEV_W*BEV_H; idx++){
        pillarsIndices[idx] = -1;
    }
    int pillarCount = threadIdx*pillarsPerThread;

    for(size_t idx = 0; idx < pointNum; idx++){
        
        auto x = points[idx*5];
        auto y = points[idx*5+1];
        auto z = points[idx*5+2];
        if(x < X_MIN || x > X_MAX || y < Y_MIN || y > Y_MAX || 
           z < Z_MIN || z > Z_MAX)
           continue;

        int xIdx = int((x-X_MIN)/X_STEP);
        int yIdx = int((y-Y_MIN)/Y_STEP);
        
        if(xIdx % THREAD_NUM != threadIdx)
            continue;


        int pillarIdx = yIdx*BEV_W+xIdx;
        auto pillarCountIdx = pillarsIndices[pillarIdx];

        auto pointNumInPillar = pointCount[pillarCountIdx];
        if(pointNumInPillar > MAX_PIONT_IN_PILLARS - 1)
           continue;
  
        // new pillar index
        if(pillarCountIdx == -1){
            pillarCountIdx = pillarCount;
            pillarsIndices[pillarIdx] = pillarCount;
            indices[pillarCount*2 + 1] = pillarIdx;
            ++pillarCount;
        }
 
        feature[                                     pillarCountIdx*MAX_PIONT_IN_PILLARS + pointNumInPillar] = x;
        feature[1*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarCountIdx*MAX_PIONT_IN_PILLARS + pointNumInPillar] = y;
        feature[2*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarCountIdx*MAX_PIONT_IN_PILLARS + pointNumInPillar] = z; // z
        feature[3*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarCountIdx*MAX_PIONT_IN_PILLARS + pointNumInPillar] = points[idx*5+3]; // instence
        feature[4*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarCountIdx*MAX_PIONT_IN_PILLARS + pointNumInPillar] = points[idx*5+4]; // time_lag

        feature[8*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarCountIdx*MAX_PIONT_IN_PILLARS + pointNumInPillar] = x - (xIdx*X_STEP + X_MIN + X_STEP/2);
        feature[9*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarCountIdx*MAX_PIONT_IN_PILLARS + pointNumInPillar] = y - (yIdx*Y_STEP + Y_MIN + Y_STEP/2);

        ++pointNumInPillar;
        pointCount[pillarCountIdx] = pointNumInPillar;
        
    }
    
    for(size_t pillarIdx = threadIdx*pillarsPerThread; pillarIdx < (threadIdx+1)*pillarsPerThread; pillarIdx++)
    {
        float xCenter = 0;
        float yCenter = 0;
        float zCenter = 0;
        auto pointNum = pointCount[pillarIdx];
        for(size_t pointIdx=0; pointIdx < pointNum; pointIdx++)
        {
            auto x = feature[                                     pillarIdx*MAX_PIONT_IN_PILLARS + pointIdx];
            auto y = feature[1*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarIdx*MAX_PIONT_IN_PILLARS + pointIdx];
            auto z = feature[2*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarIdx*MAX_PIONT_IN_PILLARS + pointIdx];
            xCenter += x;
            yCenter += y;
            zCenter += z;
        }
        xCenter = xCenter / pointNum;
        yCenter = yCenter / pointNum;
        zCenter = zCenter / pointNum;
        
        for(size_t pointIdx=0; pointIdx < pointNum; pointIdx++)
        {    
            auto x = feature[                                     pillarIdx*MAX_PIONT_IN_PILLARS + pointIdx];
            auto y = feature[1*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarIdx*MAX_PIONT_IN_PILLARS + pointIdx];
            auto z = feature[2*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarIdx*MAX_PIONT_IN_PILLARS + pointIdx];
       
            feature[5*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarIdx*MAX_PIONT_IN_PILLARS + pointIdx] = x - xCenter;
            feature[6*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarIdx*MAX_PIONT_IN_PILLARS + pointIdx] = y - yCenter;
            feature[7*MAX_PILLARS*MAX_PIONT_IN_PILLARS + pillarIdx*MAX_PIONT_IN_PILLARS + pointIdx] = z - zCenter;

        }
    }
    
}

void preprocess(float* points, float* feature, int* indices, int pointNum)
{
    for(auto idx=0; idx< MAX_PILLARS*2; idx++){
        indices[idx] = -1;
    }
    for(auto idx=0; idx< MAX_PILLARS*FEATURE_NUM*MAX_PIONT_IN_PILLARS; idx++){
        feature[idx] = 0;
    }

    std::vector<std::thread> threadPool;
    for(auto idx=0; idx < THREAD_NUM; idx++){
        std::thread worker(PreprocessWorker,
                                             points,
                                             feature,
                                             indices,
                                             pointNum,
                                             idx,
                                             MAX_PILLARS/THREAD_NUM
                                             );
        
        threadPool.push_back(std::move(worker));
    }

    for(auto idx=0; idx < THREAD_NUM; idx++){
        threadPool[idx].join();
    }
}

bool readBinFile(std::string& filename, void*& bufPtr, int& pointNum)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file) {
        sample::gLogError << "[Error] Open file " << filename << " failed" << std::endl;
        return false;
    }
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    bufPtr = malloc(fileSize);
    if(bufPtr == nullptr){
        sample::gLogError << "[Error] Malloc Memory Failed! Size: " << fileSize << std::endl;
        return false;
    }
    // read the data:
    file.read((char*) bufPtr, fileSize);
    file.close();
    
    constexpr int featureNum = 5;
    pointNum = fileSize /sizeof(float) / featureNum;
    if( fileSize /sizeof(float) % featureNum != 0){
         sample::gLogError << "[Error] File Size Error! " << fileSize << std::endl;
    }
    sample::gLogInfo << "[INFO] pointNum : " << pointNum << std::endl;
    return true;
}


