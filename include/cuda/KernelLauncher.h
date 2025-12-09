#pragma once

struct CudaVoxel;

extern "C" {

// 1. Интеграция и предсказание позиции (на основе гравитации и скорости)
void launch_predictPositions(
    CudaVoxel* d_voxels,
    size_t numVoxels,
    float dt,
    float gravity
    );

// 2. Построение хеша (оставляем вашу функцию, она подходит)
void launch_buildSpatialHash(
    CudaVoxel* d_voxels,
    unsigned int numVoxels,
    unsigned int* d_gridParticleHash,
    unsigned int* d_gridParticleIndex,
    unsigned int* d_cellStart,
    unsigned int* d_cellEnd,
    unsigned int gridSize,
    float cellSize
    );

// 3. Решение констрейнтов (столкновения + трение)
void launch_solveCollisionsPBD(
    CudaVoxel* d_voxels,
    unsigned int numVoxels,
    unsigned int* d_gridParticleIndex,
    unsigned int* d_cellStart,
    unsigned int* d_cellEnd,
    unsigned int gridSize,
    float cellSize,
    float dt // Нужно для расчета трения
    );

// 4. Обновление скоростей (Velocity Update)
void launch_updateVelocities(
    CudaVoxel* d_voxels,
    size_t numVoxels,
    float dt,
    float damping
    );

}
