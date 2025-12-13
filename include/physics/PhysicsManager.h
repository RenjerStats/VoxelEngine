#pragma once

#include <vector>
#include "core/Types.h"

struct cudaGraphicsResource;
class SpatialHash;
class ClusterManager;

class PhysicsManager {
public:
    PhysicsManager();
    PhysicsManager(unsigned int frameRate, unsigned int countVoxels);
    ~PhysicsManager();


    PhysicsManager(PhysicsManager&& other) noexcept;
    PhysicsManager& operator=(PhysicsManager&& other) noexcept;

    void uploadVoxelsToGPU(const std::vector<RenderVoxel>& voxels); // только для отладки
    void connectToOpenGL(unsigned int vboID);
    void updatePhysics(float speedSimulation, float stability);

    void freeResources();

    void sortVoxels();

private:
    void initClusters();
    cudaGraphicsResource* openGLConnector = nullptr;
    unsigned int frameRate = 60;
    unsigned int countVoxels = 0;
    SpatialHash* m_spatialHash = nullptr;
    ClusterManager* m_clusterManager = nullptr;

    // Каноничное хранение физики: SoA
    float* d_posX    = nullptr;
    float* d_posY    = nullptr;
    float* d_posZ    = nullptr;

    float* d_oldX    = nullptr;
    float* d_oldY    = nullptr;
    float* d_oldZ    = nullptr;

    float* d_velX    = nullptr;
    float* d_velY    = nullptr;
    float* d_velZ    = nullptr;

    float* d_mass    = nullptr;
    float* d_friction= nullptr;

    unsigned int* d_colorID = nullptr;

    // Каноничное хранение физики: SoA
    float* d_sortedPosX = nullptr;
    float* d_sortedPosY = nullptr;
    float* d_sortedPosZ = nullptr;

    float* d_sortedOldX = nullptr;
    float* d_sortedOldY = nullptr;
    float* d_sortedOldZ = nullptr;

    float* d_sortedVelX = nullptr;
    float* d_sortedVelY = nullptr;
    float* d_sortedVelZ = nullptr;

    float* d_sortedMass     = nullptr;
    float* d_sortedFriction = nullptr;
    unsigned int* d_sortedColorID = nullptr;
};
