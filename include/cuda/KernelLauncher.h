#pragma once

struct CudaVoxel;

extern "C" {
void launch_gravityKernel(float dt, size_t numVoxels,  CudaVoxel* d_ptr);
void launch_floorKernel(float floorElasticity, size_t numVoxels,  CudaVoxel* d_ptr);


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


void launch_updatePhysicsPBD(
    CudaVoxel* d_voxels,
    unsigned int numVoxels,
    float dt,
    unsigned int substeps,
    unsigned int* d_gridParticleHash,
    unsigned int* d_gridParticleIndex,
    unsigned int* d_cellStart,
    unsigned int* d_cellEnd,
    unsigned int gridSize,
    float cellSize
    );
}
