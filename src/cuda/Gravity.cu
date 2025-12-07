#include "cuda/KernelLauncher.h"
#include "core/Types.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <GL/gl.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void gravityKernel(CudaVoxel* voxels, int count, float dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    CudaVoxel v = voxels[idx];
    if (v.mass == 0.0f) return;

    v.vy -= dt / 5;
    if(v.vy < -1) v.vy = -1;

    v.y += v.vy;

    voxels[idx] = v;
}

extern "C" {
void launch_gravityKernel(float dt, size_t numVoxels, CudaVoxel* d_ptr) {
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;
    gravityKernel<<<blocks, threads>>>(d_ptr, numVoxels, dt);
}
}

