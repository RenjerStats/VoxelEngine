#include "cuda/KernelLauncher.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

// --- KERNELS ---

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

__device__ int calcGridHash(int gridPos_x, int gridPos_y, int gridPos_z, int gridSize) {
    // 1. Ограничиваем координаты "виртуальной сеткой" 256x256x256
    // Важно: координаты должны быть положительными для Мортона.
    // Если сцена центрирована в (0,0,0), лучше добавить offset перед кастом в unsigned.
    // Здесь мы просто берем младшие биты (& 0xFF), что работает как быстрый modulo 256.

    unsigned int x = static_cast<unsigned int>(gridPos_x) & 0xFF;
    unsigned int y = static_cast<unsigned int>(gridPos_y) & 0xFF;
    unsigned int z = static_cast<unsigned int>(gridPos_z) & 0xFF;

    // 2. Считаем Z-индекс
    unsigned int mCode = morton3D(x, y, z);

    // 3. Ограничиваем размером хеш-таблицы
    return mCode & (gridSize - 1);
}

__global__ void calcHashKernel(unsigned int* gridParticleHash,
                               unsigned int* gridParticleIndex,
                               const float* posX,
                               const float* posY,
                               const float* posZ,
                               unsigned int numVoxels,
                               float cellSize,
                               int gridSize)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numVoxels) return;

    float x = posX[index];
    float y = posY[index];
    float z = posZ[index];

    int gridPos_x = floorf(x / cellSize);
    int gridPos_y = floorf(y / cellSize);
    int gridPos_z = floorf(z / cellSize);

    gridParticleHash[index]  = calcGridHash(gridPos_x, gridPos_y, gridPos_z, gridSize);
    gridParticleIndex[index] = index;
}

__global__ void resetCellBoundsKernel(unsigned int* cellStart, unsigned int* cellEnd, unsigned int gridSize) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= gridSize) return;

    cellStart[index] = 0xFFFFFFFF;
    cellEnd[index]   = 0xFFFFFFFF;
}

__global__ void findCellBoundsKernel(unsigned int* gridParticleHash, unsigned int* cellStart, unsigned int* cellEnd, unsigned int numVoxels) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numVoxels) return;

    unsigned int hash = gridParticleHash[index];

    // Начало ячейки: если это первый элемент или предыдущий имеет другой хеш
    if (index == 0 || hash != gridParticleHash[index - 1]) {
        cellStart[hash] = index;
    }
    // Конец ячейки: если это последний элемент или следующий имеет другой хеш
    if (index == numVoxels - 1 || hash != gridParticleHash[index + 1]) {
        cellEnd[hash] = index + 1;
    }
}

// --- LAUNCHER IMPLEMENTATION ---
extern "C" {
void launch_buildSpatialHash(
    const float* d_posX,
    const float* d_posY,
    const float* d_posZ,
    unsigned int numVoxels,
    unsigned int* d_gridParticleHash,
    unsigned int* d_gridParticleIndex,
    unsigned int* d_cellStart,
    unsigned int* d_cellEnd,
    unsigned int gridSize,
    float cellSize)
{
    int threads = 256;
    int blocksParticles = (numVoxels + threads - 1) / threads;
    int blocksGrid      = (gridSize + threads - 1) / threads;

    // 1. Считаем хеши
    calcHashKernel<<<blocksParticles, threads>>>(
        d_gridParticleHash,
        d_gridParticleIndex,
        d_posX,
        d_posY,
        d_posZ,
        numVoxels,
        cellSize,
        gridSize
        );

    // 2. Сортируем (используем Thrust)
    // Thrust сам разберется с указателями, device_ptr просто оборачивает сырой указатель
    thrust::device_ptr<unsigned int> t_hash(d_gridParticleHash);
    thrust::device_ptr<unsigned int> t_index(d_gridParticleIndex);

    // Сортируем хеши и синхронно переставляем индексы частиц
    thrust::sort_by_key(t_hash, t_hash + numVoxels, t_index);

    // 3. Очищаем таблицу ячеек
    resetCellBoundsKernel<<<blocksGrid, threads>>>(d_cellStart, d_cellEnd, gridSize);

    // 4. Находим границы ячеек
    findCellBoundsKernel<<<blocksParticles, threads>>>(
        d_gridParticleHash, d_cellStart, d_cellEnd, numVoxels
        );
}

}
