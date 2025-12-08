#pragma once

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

    void registerVoxelSharedBuffer(unsigned int vboID);

    void initSumulation(bool withVoxelConnection = false, bool withVoxelCollision = true);

    void updatePhysics(float speedSimulation, float stability);

    void freeResources();

private:
    cudaGraphicsResource* cuda_vbo_resource = nullptr;
    unsigned int frameRate = 60;
    unsigned int countVoxels = 0;

    std::unique_ptr<SpatialHash> m_spatialHash;
};
