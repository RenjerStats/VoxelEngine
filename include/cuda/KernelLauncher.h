#pragma once

struct double3;
struct RenderVoxel;

extern "C" {

void launch_findConnectedComponents(
    const float* posX, const float* posY, const float* posZ,
    const float* mass,
    int* clusterID,
    const unsigned int* cellStart, const unsigned int* cellEnd,
    unsigned int gridSize, float cellSize,
    unsigned int numVoxels
    );


void launch_initClusterState(
    float* posX, float* posY, float* posZ,
    const float* mass,
    int* clusterID,
    float* restOffsetX, float* restOffsetY, float* restOffsetZ,
    double3* clusterCM, float* clusterMass,
    unsigned int numVoxels
    );

void launch_initSingleClusterState(
    const float* posX, const float* posY, const float* posZ,
    const float* mass,
    const int* clusterID,
    float* restOffsetX, float* restOffsetY, float* restOffsetZ,
    double3* clusterCM, float* clusterMass,
    int targetClusterID,
    unsigned int begin, unsigned int count
    );

void launch_shapeMatchingStep(
    float* posX, float* posY, float* posZ,
    const float* mass,
    int* clusterID,
    float* restOffsetX, float* restOffsetY, float* restOffsetZ,
    double3* clusterCM, float* clusterMass, double* clusterMatrixA, float* clusterRot,
    unsigned int numVoxels,
    unsigned int maxClusters,
    float stiffness, float rotStiffness, float breakLimit
    );

void launch_initSoAFromGL(
    const RenderVoxel* OpenGLVoxels,
    float* posX, float* posY, float* posZ,
    float* oldX, float* oldY, float* oldZ,
    float* velX, float* velY, float* velZ,
    float* mass, float* friction, unsigned int* d_colorID,
    unsigned int countVoxels
    );

void launch_predictPositions(
    float* posX, float* posY, float* posZ,
    float* oldX, float* oldY, float* oldZ,
    float* velX, float* velY, float* velZ,
    float* mass,
    size_t numVoxels,
    float dt,
    float gravity
    );

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
    float cellSize
    );

void launch_sortVoxels(
    unsigned int numVoxels,
    const unsigned int* sortedIndices,
    const float* inPX, const float* inPY, const float* inPZ,
    const float* inOX, const float* inOY, const float* inOZ,
    const float* inVX, const float* inVY, const float* inVZ,
    const float* inMass, const float* inFric, const unsigned int* inColor,

    const int* inClusterID,
    const float* inOffX, const float* inOffY, const float* inOffZ,

    float* outPX, float* outPY, float* outPZ,
    float* outOX, float* outOY, float* outOZ,
    float* outVX, float* outVY, float* outVZ,
    float* outMass, float* outFric, unsigned int* outColor,

    int* outClusterID,
    float* outOffX, float* outOffY, float* outOffZ
    );

void launch_solveCollisionsPBD(
    float* posX, float* posY, float* posZ,
    float* oldX, float* oldY, float* oldZ,
    float* mass, float* friction, int* clusterID,
    float* clusterMass,
    unsigned int numVoxels,
    unsigned int* d_cellStart,
    unsigned int* d_cellEnd,
    unsigned int gridSize,
    float cellSize,
    float dt);

void launch_updateVelocities(
    float* posX, float* posY, float* posZ,
    float* oldX, float* oldY, float* oldZ,
    float* velX, float* velY, float* velZ,
    float* mass,
    size_t numVoxels,
    float dt,
    float damping
    );

void launch_copyDataToOpenGL(
    RenderVoxel* d_renderBuffer,
    const float* d_posX,
    const float* d_posY,
    const float* d_posZ,
    const unsigned int* d_colorID,
    unsigned int numVoxels
    );

}
