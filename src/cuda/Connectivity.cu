#include "cuda/KernelLauncher.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// --- HASH HELPERS (Должен совпадать с SpatialHash.cu) ---
// В идеале вынести в Common.h, но пока копируем для автономности.
__device__ __forceinline__ unsigned int expandBits(unsigned int n)
{
    n = (n ^ (n << 16)) & 0x030000ff;
    n = (n ^ (n <<  8)) & 0x0300f00f;
    n = (n ^ (n <<  4)) & 0x030c30c3;
    n = (n ^ (n <<  2)) & 0x09249249;
    return n;
}

__device__ __forceinline__ unsigned int morton3D(unsigned int x, unsigned int y, unsigned int z)
{
    return (expandBits(z) << 2) | (expandBits(y) << 1) | expandBits(x);
}

__device__ inline int getGridHash(int gridPos_x, int gridPos_y, int gridPos_z, int gridSize) {
    unsigned int x = static_cast<unsigned int>(gridPos_x) & 0xFF;
    unsigned int y = static_cast<unsigned int>(gridPos_y) & 0xFF;
    unsigned int z = static_cast<unsigned int>(gridPos_z) & 0xFF;
    unsigned int mCode = morton3D(x, y, z);
    return mCode & (gridSize - 1);
}

// --------------------------------------------------------

__global__ void initClusterIDsKernel(int* clusterID, const float* mass, unsigned int numVoxels) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;
    if (mass[idx] <= 0.0f) {
        clusterID[idx] = -1;
    } else {
        clusterID[idx] = idx;
    }

}

// Итерация распространения меток (Label Propagation)
// Предполагается, что posX/Y/Z и clusterID уже отсортированы по хешу,
// и cellStart/End указывают на прямые индексы.
__global__ void propagateClusterIDsKernel(
    const float* posX, const float* posY, const float* posZ,
    int* clusterID,
    unsigned int* changedFlag,
    const unsigned int* cellStart, const unsigned int* cellEnd,
    unsigned int gridSize, float cellSize, float connectDistSq,
    unsigned int numVoxels
    ) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;

    // Текущая позиция
    float px = posX[idx];
    float py = posY[idx];
    float pz = posZ[idx];
    int myID = clusterID[idx];

    // Вычисляем свою ячейку
    int gridX = floorf(px / cellSize);
    int gridY = floorf(py / cellSize);
    int gridZ = floorf(pz / cellSize);

    int minID = myID;
    if (myID < 0) return;

    // Проход по соседям (3x3x3)
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int hash = getGridHash(gridX + x, gridY + y, gridZ + z, gridSize);

                // Используем __ldg для CellStart/End (read-only)
                unsigned int start = __ldg(&cellStart[hash]);
                if (start == 0xFFFFFFFF) continue;

                unsigned int end = __ldg(&cellEnd[hash]);

                for (unsigned int k = start; k < end; k++) {
                    unsigned int neighborIdx = k;
                    if (neighborIdx == idx) continue;
                    int neighborID = clusterID[neighborIdx];
                    if (neighborID < 0) continue;

                    // Проверка расстояния
                    float nx = posX[neighborIdx];
                    float ny = posY[neighborIdx];
                    float nz = posZ[neighborIdx];

                    float dX = px - nx;
                    float dY = py - ny;
                    float dZ = pz - nz;
                    float dSq = dX*dX + dY*dY + dZ*dZ;

                    if (dSq <= connectDistSq) {
                        if (neighborID < minID) {
                            minID = neighborID;
                        }
                    }
                }
            }
        }
    }

    if (minID < myID) {
        atomicMin(&clusterID[idx], minID);
        atomicExch(changedFlag, 1);
    }
}

__global__ void compressClusterIDsKernel(int* clusterID, unsigned int* changedFlag, unsigned int numVoxels) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;

    int id = clusterID[idx];
    if (id < 0) return;

    int parent = clusterID[id];
    if (parent >= 0 && parent < id) {
        clusterID[idx] = parent;
        atomicExch(changedFlag, 1);
    }
}




extern "C" {
void launch_findConnectedComponents(
    const float* posX, const float* posY, const float* posZ,
    const float* mass,
    int* clusterID,
    const unsigned int* cellStart, const unsigned int* cellEnd,
    unsigned int gridSize, float cellSize,
    unsigned int numVoxels
    ) {
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;

    // 1. Init IDs
    initClusterIDsKernel<<<blocks, threads>>>(clusterID, mass, numVoxels);
    cudaDeviceSynchronize();

    unsigned int* d_changed;
    cudaMalloc(&d_changed, sizeof(unsigned int));

    float connectDist = 1.05f;
    float distSq = connectDist * connectDist;

    // 2. Loop until convergence (Label Propagation)
    int iter = 0;
    while (iter < 2048) {
        unsigned int h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(unsigned int), cudaMemcpyHostToDevice);

        propagateClusterIDsKernel<<<blocks, threads>>>(
            posX, posY, posZ,
            clusterID,
            d_changed, cellStart, cellEnd,
            gridSize, cellSize, distSq,
            numVoxels
            );

        compressClusterIDsKernel<<<blocks, threads>>>(clusterID, d_changed, numVoxels);

        cudaDeviceSynchronize();
        cudaMemcpy(&h_changed, d_changed, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (h_changed == 0) break;
        iter++;
    }

    cudaFree(d_changed);
}
}
