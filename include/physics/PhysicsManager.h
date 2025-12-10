#pragma once
#include <vector>
#include "core/Types.h"

#include <memory>

struct cudaGraphicsResource;
class SpatialHash;

class PhysicsManager {
public:
    PhysicsManager();
    PhysicsManager(unsigned int frameRate, unsigned int countVoxels);
    ~PhysicsManager();

    PhysicsManager(PhysicsManager&& other) noexcept;
    PhysicsManager& operator=(PhysicsManager&& other) noexcept;

    void uploadVoxelsToGPU(const std::vector<CudaVoxel>& voxels); // только для отладки

    void registerVoxelSharedBuffer(unsigned int vboID);

    void initSumulation(bool withVoxelConnection = false, bool withVoxelCollision = true);

    void updatePhysics(float speedSimulation, float stability);

    void freeResources();

private:
    cudaGraphicsResource* cuda_vbo_resource = nullptr;
    unsigned int frameRate = 60;
    unsigned int countVoxels = 0;

    CudaVoxel* d_standalone_buffer = nullptr; // только для отладки

    std::unique_ptr<SpatialHash> m_spatialHash;
};
