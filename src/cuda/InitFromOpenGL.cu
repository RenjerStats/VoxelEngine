#include "cuda/KernelLauncher.h"
#include "core/Types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void initSoAFromGLKernel(
    const RenderVoxel* OpenGLVoxels,
    float* posX, float* posY, float* posZ,
    float* oldX, float* oldY, float* oldZ,
    float* velX, float* velY, float* velZ,
    float* mass, float* friction, unsigned int* colorID,
    unsigned int countVoxels
    ) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= countVoxels) return;
    RenderVoxel v = OpenGLVoxels[i];

    posX[i] = v.x;
    posY[i] = v.y;
    posZ[i] = v.z;

    colorID[i] = v.colorID;

    oldX[i] = v.x;
    oldY[i] = v.y;
    oldZ[i] = v.z;

    velX[i] = 0;
    velY[i] = 0;
    velZ[i] = 0;

    if (colorID[i] == 255){ // зарезервированный цвет для неподвижных вокселей
        mass[i] = 0;
        friction[i] = 0.9;
    }
    else{
        mass[i] = 1;
        friction[i] = 0.1;
    }
}

void launch_initSoAFromGL(
    const RenderVoxel* OpenGLVoxels,
    float* posX, float* posY, float* posZ,
    float* oldX, float* oldY, float* oldZ,
    float* velX, float* velY, float* velZ,
    float* mass, float* friction, unsigned int* d_colorID,
    unsigned int countVoxels)
{
    if (countVoxels == 0) return;

    int threads = 256;
    int blocks = (countVoxels + threads - 1) / threads;

    initSoAFromGLKernel<<<blocks, threads>>>(
        OpenGLVoxels,
        posX, posY, posZ,
        oldX, oldY, oldZ,
        velX, velY, velZ,
        mass, friction, d_colorID,
        countVoxels
        );
}
