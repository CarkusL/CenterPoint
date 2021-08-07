#ifndef __CENTERNET_CONFIG_H__
#define __CENTERNET_CONFIG_H__

#define X_STEP 0.2f
#define Y_STEP 0.2f
#define X_MIN -51.2f
#define X_MAX 51.2f
#define Y_MIN -51.2f
#define Y_MAX 51.2f
#define Z_MIN -5.0f
#define Z_MAX 3.0f
#define PI 3.141592653f

// paramerters for preprocess
#define BEV_W 512
#define BEV_H 512
#define MAX_PILLARS 30000
#define MAX_PIONT_IN_PILLARS 20
#define FEATURE_NUM 10
#define THREAD_NUM 2

// paramerters for postprocess
#define SCORE_THREAHOLD 0.1f
#define NMS_THREAHOLD 0.2f
#define INPUT_NMS_MAX_SIZE 1000
#define OUT_SIZE_FACTOR 4.0f
#define TASK_NUM 6
#define REG_CHANNEL 2
#define HEIGHT_CHANNEL 1
#define ROT_CHANNEL 2
#define VEL_CHANNEL 2
#define DIM_CHANNEL 3
#define OUTPUT_H 128
#define OUTPUT_W 128

#endif