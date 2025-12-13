#include "cuda/KernelLauncher.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void reorderDataFloatKernel(
    const unsigned int* sortedIndices,
    const float* input,
    float* output,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    // sortedIndices[i] хранит индекс "где эта частица была раньше"
    unsigned int originalIdx = sortedIndices[i];
    output[i] = input[originalIdx];
}

__global__ void reorderDataIntKernel(
    const unsigned int* sortedIndices,
    const unsigned int* input,
    unsigned int* output,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = input[sortedIndices[i]];
}

__global__ void reorderDataIntKernel(
    const unsigned int* sortedIndices,
    const int* input,
    int* output,
    unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned int originalIdx = sortedIndices[i];
    output[i] = input[originalIdx];
}

extern "C" {
void launch_sortVoxels(
    unsigned int numVoxels,
    const unsigned int* sortedIndices,
    // Входные (Unsorted)
    const float* inPX, const float* inPY, const float* inPZ,
    const float* inOX, const float* inOY, const float* inOZ,
    const float* inVX, const float* inVY, const float* inVZ,
    const float* inMass, const float* inFric, const unsigned int* inColor,

    const int* inClusterID,
    const float* inOffX, const float* inOffY, const float* inOffZ,

    // Выходные (Sorted)
    float* outPX, float* outPY, float* outPZ,
    float* outOX, float* outOY, float* outOZ,
    float* outVX, float* outVY, float* outVZ,
    float* outMass, float* outFric, unsigned int* outColor,

    int* outClusterID,
    float* outOffX, float* outOffY, float* outOffZ
    ) {
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;

    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inPX, outPX, numVoxels);
    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inPY, outPY, numVoxels);
    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inPZ, outPZ, numVoxels);

    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inOX, outOX, numVoxels);
    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inOY, outOY, numVoxels);
    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inOZ, outOZ, numVoxels);

    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inVX, outVX, numVoxels);
    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inVY, outVY, numVoxels);
    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inVZ, outVZ, numVoxels);

    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inMass, outMass, numVoxels);
    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inFric, outFric, numVoxels);

    reorderDataIntKernel<<<blocks, threads>>>(sortedIndices, inColor, outColor, numVoxels);

    reorderDataIntKernel<<<blocks, threads>>>(sortedIndices, inClusterID, outClusterID, numVoxels);

    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inOffX, outOffX, numVoxels);
    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inOffY, outOffY, numVoxels);
    reorderDataFloatKernel<<<blocks, threads>>>(sortedIndices, inOffZ, outOffZ, numVoxels);
}
}
