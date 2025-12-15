#pragma once

#include <vector>
#include <functional>
#include "core/Types.h"

struct cudaGraphicsResource;
class SpatialHash;
class ClusterManager;

using VoxelCallback = std::function<void(unsigned int, unsigned int)>;

class PhysicsManager {
public:
    PhysicsManager(){};
    PhysicsManager(unsigned int frameRate, unsigned int countVoxels);
    ~PhysicsManager();


    PhysicsManager(PhysicsManager&& other) noexcept;
    PhysicsManager& operator=(PhysicsManager&& other) noexcept;

    PhysicsManager(const PhysicsManager&) = delete;
    PhysicsManager& operator=(const PhysicsManager&) = delete;

    void uploadVoxelsToGPU(const std::vector<RenderVoxel>& voxels, unsigned int _maxVoxels); // только для отладки
    void connectToOpenGL(unsigned int vboID, unsigned int _activeVoxels);

    void spawnSphere(float x, float y, float z, float radius, float vx, float vy, float vz, int colorID);
    void spawnCube(float x, float y, float z, int size, float vx, float vy, float vz, int colorID);

    void updatePhysics(float speedSimulation, float stability);

    void freeResources();

    void setVoxelCallback(VoxelCallback callback) {
        voxelCallback = callback;
    }

    void updateGLResource(unsigned int newVboID);
    unsigned int getActiveVoxels() const { return activeVoxels; }


private:
    void sortVoxels();
    void initClusters();
    void resizeMemory();
    void initMemory();
    void addNewVoxels(float vz, size_t count, std::vector<RenderVoxel> newVoxels, float vy, float vx, float startY, float startX, float startZ);
    void resizeOpenGLBuffer();
    void spawnVoxels(const std::vector<RenderVoxel>& newVoxels,
                             float offsetX, float offsetY, float offsetZ,
                             float velX, float velY, float velZ,
                             unsigned int colorID);

    cudaGraphicsResource* openGLConnector = nullptr;
    unsigned int frameRate = 60;
    SpatialHash* m_spatialHash = nullptr;
    ClusterManager* m_clusterManager = nullptr;
    int maxClusterID = -1;

    VoxelCallback voxelCallback;

    unsigned int maxVoxels = 1000000;
    unsigned int activeVoxels = 0;


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
    void sendVoxelToOpenGL();
};
