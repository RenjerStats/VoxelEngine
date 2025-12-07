#include "cuda/SpatialHash.h"
#include "cuda/KernelLauncher.h"
#include "core/Types.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>


// Конструктор
SpatialHash::SpatialHash(unsigned int gridSize, float cellSize)
    : m_numVoxels(0), m_gridSize(gridSize), m_cellSize(cellSize),
    d_gridParticleHash(nullptr), d_gridParticleIndex(nullptr),
    d_cellStart(nullptr), d_cellEnd(nullptr)
{
    // Выделяем память под фиксированную таблицу ячеек сразу
    cudaMalloc((void**)&d_cellStart, m_gridSize * sizeof(unsigned int));
    cudaMalloc((void**)&d_cellEnd, m_gridSize * sizeof(unsigned int));
}

// Деструктор
SpatialHash::~SpatialHash() {
    if (d_gridParticleHash) cudaFree(d_gridParticleHash);
    if (d_gridParticleIndex) cudaFree(d_gridParticleIndex);
    if (d_cellStart) cudaFree(d_cellStart);
    if (d_cellEnd) cudaFree(d_cellEnd);
}

// Метод resize
void SpatialHash::resize(unsigned int numVoxels) {
    if (m_numVoxels == numVoxels) return;

    if (d_gridParticleHash) cudaFree(d_gridParticleHash);
    if (d_gridParticleIndex) cudaFree(d_gridParticleIndex);

    m_numVoxels = numVoxels;

    if (m_numVoxels > 0) {
        cudaMalloc((void**)&d_gridParticleHash, m_numVoxels * sizeof(unsigned int));
        cudaMalloc((void**)&d_gridParticleIndex, m_numVoxels * sizeof(unsigned int));
    }
}

// --- KERNELS (как и были, но теперь статические или в анонимном пространстве имен) ---

__device__ int calcGridHash(int gridPos_x, int gridPos_y, int gridPos_z, int gridSize) {
    const int p1 = 73856093;
    const int p2 = 19349663;
    const int p3 = 83492791;
    int n = p1 * gridPos_x ^ p2 * gridPos_y ^ p3 * gridPos_z;
    return n & (gridSize - 1);
}

__global__ void calcHashKernel(unsigned int* gridParticleHash, unsigned int* gridParticleIndex,
                               CudaVoxel* voxels, unsigned int numVoxels, float cellSize, int gridSize) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numVoxels) return;

    CudaVoxel v = voxels[index];
    // Проверка на "мертвые" воксели, если нужно
    if (v.mass == 0.0f) return;

    int gridPos_x = floorf(v.x / cellSize);
    int gridPos_y = floorf(v.y / cellSize);
    int gridPos_z = floorf(v.z / cellSize);

    gridParticleHash[index] = calcGridHash(gridPos_x, gridPos_y, gridPos_z, gridSize);
    gridParticleIndex[index] = index;
}

__global__ void resetCellBoundsKernel(unsigned int* cellStart, unsigned int* cellEnd, unsigned int gridSize) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= gridSize) return;
    cellStart[index] = 0xFFFFFFFF;
    cellEnd[index] = 0xFFFFFFFF;
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

// Единая точка входа для построения структуры данных
void launch_buildSpatialHash(
    CudaVoxel* d_voxels,
    unsigned int numVoxels,
    unsigned int* d_gridParticleHash,
    unsigned int* d_gridParticleIndex,
    unsigned int* d_cellStart,
    unsigned int* d_cellEnd,
    unsigned int gridSize,
    float cellSize
    ) {
    if (numVoxels == 0) return;

    int threads = 256;
    int blocksParticles = (numVoxels + threads - 1) / threads;
    int blocksGrid = (gridSize + threads - 1) / threads;

    // 1. Считаем хеши
    calcHashKernel<<<blocksParticles, threads>>>(
        d_gridParticleHash, d_gridParticleIndex, d_voxels, numVoxels, cellSize, gridSize
        );

    // 2. Сортируем (используем Thrust)
    // Thrust оборачивается здесь, чтобы PhysicsManager не тянул зависимость от Thrust
    thrust::device_ptr<unsigned int> t_hash(d_gridParticleHash);
    thrust::device_ptr<unsigned int> t_index(d_gridParticleIndex);
    thrust::sort_by_key(t_hash, t_hash + numVoxels, t_index);

    // 3. Очищаем таблицу ячеек
    resetCellBoundsKernel<<<blocksGrid, threads>>>(d_cellStart, d_cellEnd, gridSize);

    // 4. Находим границы
    findCellBoundsKernel<<<blocksParticles, threads>>>(
        d_gridParticleHash, d_cellStart, d_cellEnd, numVoxels
        );
}

}
