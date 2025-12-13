#include "cuda/KernelLauncher.h"
#include "core/Types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void copyToGLKernel(
    RenderVoxel* d_out,
    const float* d_posX,
    const float* d_posY,
    const float* d_posZ,
    const unsigned int* d_colorID,
    unsigned int numVoxels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVoxels) return;

    float x = d_posX[i];
    float y = d_posY[i];
    float z = d_posZ[i];
    unsigned int c = d_colorID[i];

    RenderVoxel v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.colorID = c;

    d_out[i] = v;
}

extern "C" {

void launch_copyDataToOpenGL(
    RenderVoxel* d_renderBuffer,
    const float* d_posX,
    const float* d_posY,
    const float* d_posZ,
    const unsigned int* d_colorID,
    unsigned int numVoxels
    ) {

    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;

    copyToGLKernel<<<blocks, threads>>>(
        d_renderBuffer,
        d_posX,
        d_posY,
        d_posZ,
        d_colorID,
        numVoxels
        );
}

}
