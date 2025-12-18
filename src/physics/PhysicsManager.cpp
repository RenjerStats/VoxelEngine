#include "physics/PhysicsManager.h"
#include "cuda/SpatialHash.h"
#include "cuda/ClusterManager.h"
#include "core/Types.h"
#include "cuda/KernelLauncher.h"
#include "cuda/CudaInitializer.h"
#include "physics/PhysicsHelper.h"


#include <cuda_runtime.h>
#include <QDebug>

PhysicsManager::PhysicsManager(unsigned int frameRate, unsigned int maxVoxels)
    : frameRate(frameRate), maxVoxels(maxVoxels) {

    if (maxVoxels == 0)
        return;

    initMemory();
}

PhysicsManager::PhysicsManager(PhysicsManager&& other) noexcept
    : openGLConnector(other.openGLConnector),
    frameRate(other.frameRate),
    maxVoxels(other.maxVoxels),
    activeVoxels(other.activeVoxels),
    m_spatialHash(other.m_spatialHash),
    m_clusterManager(other.m_clusterManager),

    d_posX(other.d_posX), d_posY(other.d_posY), d_posZ(other.d_posZ),
    d_oldX(other.d_oldX), d_oldY(other.d_oldY), d_oldZ(other.d_oldZ),
    d_velX(other.d_velX), d_velY(other.d_velY), d_velZ(other.d_velZ),
    d_mass(other.d_mass), d_friction(other.d_friction),
    d_colorID(other.d_colorID),

    d_sortedPosX(other.d_sortedPosX), d_sortedPosY(other.d_sortedPosY), d_sortedPosZ(other.d_sortedPosZ),
    d_sortedOldX(other.d_sortedOldX), d_sortedOldY(other.d_sortedOldY), d_sortedOldZ(other.d_sortedOldZ),
    d_sortedVelX(other.d_sortedVelX), d_sortedVelY(other.d_sortedVelY), d_sortedVelZ(other.d_sortedVelZ),
    d_sortedMass(other.d_sortedMass), d_sortedFriction(other.d_sortedFriction),
    d_sortedColorID(other.d_sortedColorID)
{
    other.openGLConnector = nullptr;
    other.m_spatialHash = nullptr;
    other.m_clusterManager = nullptr;

    other.d_posX = other.d_posY = other.d_posZ = nullptr;
    other.d_oldX = other.d_oldY = other.d_oldZ = nullptr;
    other.d_velX = other.d_velY = other.d_velZ = nullptr;
    other.d_mass = other.d_friction = nullptr;
    other.d_colorID = nullptr;

    other.d_sortedPosX = other.d_sortedPosY = other.d_sortedPosZ = nullptr;
    other.d_sortedOldX = other.d_sortedOldY = other.d_sortedOldZ = nullptr;
    other.d_sortedVelX = other.d_sortedVelY = other.d_sortedVelZ = nullptr;
    other.d_sortedMass = other.d_sortedFriction = nullptr;
    other.d_sortedColorID = nullptr;
}


PhysicsManager& PhysicsManager::operator=(PhysicsManager&& other) noexcept {
    if (this != &other) {
        freeResources();

        openGLConnector = other.openGLConnector;
        frameRate = other.frameRate;
        maxVoxels = other.maxVoxels;
        activeVoxels = other.activeVoxels;
        m_spatialHash = other.m_spatialHash;
        m_clusterManager = other.m_clusterManager;

        d_posX = other.d_posX; d_posY = other.d_posY; d_posZ = other.d_posZ;
        d_oldX = other.d_oldX; d_oldY = other.d_oldY; d_oldZ = other.d_oldZ;
        d_velX = other.d_velX; d_velY = other.d_velY; d_velZ = other.d_velZ;
        d_mass = other.d_mass; d_friction = other.d_friction;
        d_colorID = other.d_colorID;

        d_sortedPosX = other.d_sortedPosX; d_sortedPosY = other.d_sortedPosY; d_sortedPosZ = other.d_sortedPosZ;
        d_sortedOldX = other.d_sortedOldX; d_sortedOldY = other.d_sortedOldY; d_sortedOldZ = other.d_sortedOldZ;
        d_sortedVelX = other.d_sortedVelX; d_sortedVelY = other.d_sortedVelY; d_sortedVelZ = other.d_sortedVelZ;
        d_sortedMass = other.d_sortedMass; d_sortedFriction = other.d_sortedFriction;
        d_sortedColorID = other.d_sortedColorID;

        other.openGLConnector = nullptr;
        other.m_spatialHash = nullptr;
        other.m_clusterManager = nullptr;

        other.d_posX = other.d_posY = other.d_posZ = nullptr;
        other.d_oldX = other.d_oldY = other.d_oldZ = nullptr;
        other.d_velX = other.d_velY = other.d_velZ = nullptr;
        other.d_mass = other.d_friction = nullptr;
        other.d_colorID = nullptr;

        other.d_sortedPosX = other.d_sortedPosY = other.d_sortedPosZ = nullptr;
        other.d_sortedOldX = other.d_sortedOldY = other.d_sortedOldZ = nullptr;
        other.d_sortedVelX = other.d_sortedVelY = other.d_sortedVelZ = nullptr;
        other.d_sortedMass = other.d_sortedFriction = nullptr;
        other.d_sortedColorID = nullptr;

    }
    return *this;
}


PhysicsManager::~PhysicsManager() {
    freeResources();
}


void PhysicsManager::initMemory()
{
    cudaMalloc(&d_posX,     maxVoxels * sizeof(float));
    cudaMalloc(&d_posY,     maxVoxels * sizeof(float));
    cudaMalloc(&d_posZ,     maxVoxels * sizeof(float));

    cudaMalloc(&d_oldX,     maxVoxels * sizeof(float));
    cudaMalloc(&d_oldY,     maxVoxels * sizeof(float));
    cudaMalloc(&d_oldZ,     maxVoxels * sizeof(float));

    cudaMalloc(&d_velX,     maxVoxels * sizeof(float));
    cudaMalloc(&d_velY,     maxVoxels * sizeof(float));
    cudaMalloc(&d_velZ,     maxVoxels * sizeof(float));

    cudaMalloc(&d_mass,     maxVoxels * sizeof(float));
    cudaMalloc(&d_friction, maxVoxels * sizeof(float));
    cudaMalloc(&d_colorID,  maxVoxels * sizeof(unsigned int));


    cudaMalloc(&d_sortedPosX, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedPosY, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedPosZ, maxVoxels * sizeof(float));

    cudaMalloc(&d_sortedOldX, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedOldY, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedOldZ, maxVoxels * sizeof(float));

    cudaMalloc(&d_sortedVelX, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedVelY, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedVelZ, maxVoxels * sizeof(float));

    cudaMalloc(&d_sortedMass, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedFriction, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedColorID, maxVoxels * sizeof(unsigned int));

    m_spatialHash = new SpatialHash();
    m_clusterManager = new ClusterManager();

    m_clusterManager->initMemory(maxVoxels);
    m_spatialHash->initMemory(maxVoxels);
}

void PhysicsManager::resizeMemory(int countNewVoxels) {

    while (activeVoxels + countNewVoxels > maxVoxels){
        maxVoxels = maxVoxels * 2;
    }

    float *old_posX = d_posX, *old_posY = d_posY, *old_posZ = d_posZ;
    float *old_oldX = d_oldX, *old_oldY = d_oldY, *old_oldZ = d_oldZ;
    float *old_velX = d_velX, *old_velY = d_velY, *old_velZ = d_velZ;
    float *old_mass = d_mass, *old_friction = d_friction;
    unsigned int *old_colorID = d_colorID;

    float *old_sortedPosX = d_sortedPosX, *old_sortedPosY = d_sortedPosY, *old_sortedPosZ = d_sortedPosZ;
    float *old_sortedOldX = d_sortedOldX, *old_sortedOldY = d_sortedOldY, *old_sortedOldZ = d_sortedOldZ;
    float *old_sortedVelX = d_sortedVelX, *old_sortedVelY = d_sortedVelY, *old_sortedVelZ = d_sortedVelZ;
    float *old_sortedMass = d_sortedMass, *old_sortedFriction = d_sortedFriction;
    unsigned int *old_sortedColorID = d_sortedColorID;

    cudaMalloc(&d_posX, maxVoxels * sizeof(float));
    cudaMalloc(&d_posY, maxVoxels * sizeof(float));
    cudaMalloc(&d_posZ, maxVoxels * sizeof(float));
    cudaMalloc(&d_oldX, maxVoxels * sizeof(float));
    cudaMalloc(&d_oldY, maxVoxels * sizeof(float));
    cudaMalloc(&d_oldZ, maxVoxels * sizeof(float));
    cudaMalloc(&d_velX, maxVoxels * sizeof(float));
    cudaMalloc(&d_velY, maxVoxels * sizeof(float));
    cudaMalloc(&d_velZ, maxVoxels * sizeof(float));
    cudaMalloc(&d_mass, maxVoxels * sizeof(float));
    cudaMalloc(&d_friction, maxVoxels * sizeof(float));
    cudaMalloc(&d_colorID, maxVoxels * sizeof(unsigned int));

    cudaMalloc(&d_sortedPosX, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedPosY, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedPosZ, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedOldX, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedOldY, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedOldZ, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedVelX, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedVelY, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedVelZ, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedMass, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedFriction, maxVoxels * sizeof(float));
    cudaMalloc(&d_sortedColorID, maxVoxels * sizeof(unsigned int));

    if (activeVoxels > 0) {
        cudaMemcpy(d_posX, old_posX, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_posY, old_posY, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_posZ, old_posZ, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_oldX, old_oldX, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_oldY, old_oldY, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_oldZ, old_oldZ, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_velX, old_velX, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_velY, old_velY, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_velZ, old_velZ, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_mass, old_mass, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_friction, old_friction, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_colorID, old_colorID, activeVoxels * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

        cudaMemcpy(d_sortedPosX, old_sortedPosX, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_sortedPosY, old_sortedPosY, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_sortedPosZ, old_sortedPosZ, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_sortedOldX, old_sortedOldX, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_sortedOldY, old_sortedOldY, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_sortedOldZ, old_sortedOldZ, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_sortedVelX, old_sortedVelX, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_sortedVelY, old_sortedVelY, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_sortedVelZ, old_sortedVelZ, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_sortedMass, old_sortedMass, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_sortedFriction, old_sortedFriction, activeVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_sortedColorID, old_sortedColorID, activeVoxels * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    }

    m_clusterManager->resizeMemory(maxVoxels);
    m_spatialHash->resizeMemory(maxVoxels);

    cudaFree(old_posX); cudaFree(old_posY); cudaFree(old_posZ);
    cudaFree(old_oldX); cudaFree(old_oldY); cudaFree(old_oldZ);
    cudaFree(old_velX); cudaFree(old_velY); cudaFree(old_velZ);
    cudaFree(old_mass); cudaFree(old_friction);
    cudaFree(old_colorID);

    cudaFree(old_sortedPosX); cudaFree(old_sortedPosY); cudaFree(old_sortedPosZ);
    cudaFree(old_sortedOldX); cudaFree(old_sortedOldY); cudaFree(old_sortedOldZ);
    cudaFree(old_sortedVelX); cudaFree(old_sortedVelY); cudaFree(old_sortedVelZ);
    cudaFree(old_sortedMass); cudaFree(old_sortedFriction);
    cudaFree(old_sortedColorID);
}

void PhysicsManager::freeResources()
{
    if (openGLConnector) {
        cuda_cleanup(openGLConnector);
        openGLConnector = nullptr;
    }

    if (d_posX)     cudaFree(d_posX);
    if (d_posY)     cudaFree(d_posY);
    if (d_posZ)     cudaFree(d_posZ);

    if (d_oldX)     cudaFree(d_oldX);
    if (d_oldY)     cudaFree(d_oldY);
    if (d_oldZ)     cudaFree(d_oldZ);

    if (d_velX)     cudaFree(d_velX);
    if (d_velY)     cudaFree(d_velY);
    if (d_velZ)     cudaFree(d_velZ);

    if (d_mass)     cudaFree(d_mass);
    if (d_friction) cudaFree(d_friction);

    if (d_colorID)  cudaFree(d_colorID);

    d_posX = d_posY = d_posZ = nullptr;
    d_oldX = d_oldY = d_oldZ = nullptr;
    d_velX = d_velY = d_velZ = nullptr;
    d_mass = d_friction = nullptr;
    d_colorID = nullptr;

    if (d_sortedPosX) cudaFree(d_sortedPosX);
    if (d_sortedPosY) cudaFree(d_sortedPosY);
    if (d_sortedPosZ) cudaFree(d_sortedPosZ);

    if (d_sortedOldX) cudaFree(d_sortedOldX);
    if (d_sortedOldY) cudaFree(d_sortedOldY);
    if (d_sortedOldZ) cudaFree(d_sortedOldZ);

    if (d_sortedVelX) cudaFree(d_sortedVelX);
    if (d_sortedVelY) cudaFree(d_sortedVelY);
    if (d_sortedVelZ) cudaFree(d_sortedVelZ);

    if (d_sortedMass) cudaFree(d_sortedMass);
    if (d_sortedFriction) cudaFree(d_sortedFriction);
    if (d_sortedColorID) cudaFree(d_sortedColorID);

    d_sortedPosX = d_sortedPosY = d_sortedPosZ = nullptr;
    d_sortedOldX = d_sortedOldY = d_sortedOldZ = nullptr;
    d_sortedVelX = d_sortedVelY = d_sortedVelZ = nullptr;
    d_sortedMass = d_sortedFriction = nullptr;
    d_sortedColorID = nullptr;

    if (m_spatialHash) {
        delete m_spatialHash;
        m_spatialHash = nullptr;
    }

    if (m_clusterManager){
        delete m_clusterManager;
        m_clusterManager = nullptr;
    };
}


void PhysicsManager::connectToOpenGL(unsigned int vboID, unsigned int _activeVoxels)
{
    cuda_registerGLBuffer(vboID, &openGLConnector);

    activeVoxels = _activeVoxels;

    if (!openGLConnector || _activeVoxels == 0) {
        return;
    }

    RenderVoxel* OpenGLVoxels = nullptr;
    size_t num_bytes;

    cudaError_t err = cudaGraphicsMapResources(1, &openGLConnector, 0);
    if (err == cudaSuccess) {
        err = cudaGraphicsResourceGetMappedPointer((void**)&OpenGLVoxels, &num_bytes, openGLConnector);

        if (err == cudaSuccess) {
            launch_initSoAFromGL(
                OpenGLVoxels,
                d_posX, d_posY, d_posZ,
                d_oldX, d_oldY, d_oldZ,
                d_velX, d_velY, d_velZ,
                d_mass, d_friction, d_colorID,
                _activeVoxels
                );
        }
        cudaGraphicsUnmapResources(1, &openGLConnector, 0);
    }

    initClusters();
}

void PhysicsManager::updateGLResource(unsigned int newVboID)
{
    cuda_cleanup(openGLConnector);
    openGLConnector = nullptr;
    cuda_registerGLBuffer(newVboID, &openGLConnector);
}

void PhysicsManager::resizeOpenGLBuffer()
{
    if (openGLConnector) {
        cuda_cleanup(openGLConnector);
        openGLConnector = nullptr;
    }

    if (voxelCallback) {
        voxelCallback(maxVoxels, activeVoxels);
    }
}

void PhysicsManager::uploadVoxelsToGPU(const std::vector<RenderVoxel>& voxels, unsigned int _maxVoxels)
{
    if (openGLConnector) {
        cuda_cleanup(openGLConnector);
        openGLConnector = nullptr;
    }
    maxVoxels = _maxVoxels;

    activeVoxels = static_cast<unsigned int>(voxels.size());
    if (activeVoxels == 0)
        return;

    initMemory();

    std::vector<float> h_posX(activeVoxels), h_posY(activeVoxels), h_posZ(activeVoxels);
    std::vector<float> h_oldX(activeVoxels), h_oldY(activeVoxels), h_oldZ(activeVoxels);
    std::vector<float> h_velX(activeVoxels), h_velY(activeVoxels), h_velZ(activeVoxels);
    std::vector<float> h_mass(activeVoxels), h_friction(activeVoxels);
    std::vector<unsigned int> h_colorID(activeVoxels);

    for (unsigned int i = 0; i < activeVoxels; ++i) {
        const auto& v = voxels[i];

        h_posX[i] = v.x;
        h_posY[i] = v.y;
        h_posZ[i] = v.z;

        h_oldX[i] = v.x;
        h_oldY[i] = v.y;
        h_oldZ[i] = v.z;

        h_velX[i] = 0;
        h_velY[i] = 0;
        h_velZ[i] = 0;

        h_mass[i]     = 1;
        h_friction[i] = 0.5;
        h_colorID[i]  = v.colorID;
    }

    cudaMemcpy(d_posX,     h_posX.data(),     activeVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY,     h_posY.data(),     activeVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ,     h_posZ.data(),     activeVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldX,     h_oldX.data(),     activeVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldY,     h_oldY.data(),     activeVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldZ,     h_oldZ.data(),     activeVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_velX,     h_velX.data(),     activeVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY,     h_velY.data(),     activeVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_velZ,     h_velZ.data(),     activeVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass,     h_mass.data(),     activeVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_friction, h_friction.data(), activeVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_colorID,  h_colorID.data(),  activeVoxels * sizeof(unsigned int), cudaMemcpyHostToDevice);


    initClusters();

    if (maxClusterID < 0) {
        maxClusterID = 0;
        std::vector<int> h_clusterID(activeVoxels, 0);
        cudaMemcpy(m_clusterManager->getVoxelClusterID(), h_clusterID.data(),
                   activeVoxels * sizeof(int), cudaMemcpyHostToDevice);
    }
}

void PhysicsManager::initClusters(){
    launch_buildSpatialHash(
        d_posX, d_posY, d_posZ, activeVoxels,
        m_spatialHash->getGridParticleHash(),
        m_spatialHash->getGridParticleIndex(),
        m_spatialHash->getCellStart(),
        m_spatialHash->getCellEnd(),
        m_spatialHash->getGridSize(),
        m_spatialHash->getCellSize()
        );

    sortVoxels();

    launch_findConnectedComponents(
        d_posX, d_posY, d_posZ, d_mass,
        m_clusterManager->getVoxelClusterID(),
        m_spatialHash->getCellStart(),
        m_spatialHash->getCellEnd(),
        m_spatialHash->getGridSize(),
        m_spatialHash->getCellSize(),
        activeVoxels
        );

    launch_initClusterState(
        d_posX, d_posY, d_posZ,
        d_mass,
        m_clusterManager->getVoxelClusterID(),
        m_clusterManager->getRestOffsetX(),
        m_clusterManager->getRestOffsetY(),
        m_clusterManager->getRestOffsetZ(),
        m_clusterManager->getClusterCM(),
        m_clusterManager->getClusterMass(),
        activeVoxels
        );

    std::vector<int> h_clusterID(activeVoxels);
    cudaMemcpy(h_clusterID.data(), m_clusterManager->getVoxelClusterID(), activeVoxels * sizeof(int), cudaMemcpyDeviceToHost);

    std::unordered_set<int> roots;
    roots.reserve(activeVoxels);

    for (int id : h_clusterID) {
        if (id >= 0) roots.insert(id);
    }

    maxClusterID = activeVoxels;

    qDebug() << "clusters =" << static_cast<int>(roots.size());
}


void PhysicsManager::sortVoxels()
{
    launch_sortVoxels(
        activeVoxels,
        m_spatialHash->getGridParticleIndex(),

        d_posX, d_posY, d_posZ,
        d_oldX, d_oldY, d_oldZ,
        d_velX, d_velY, d_velZ,
        d_mass, d_friction, d_colorID,
        m_clusterManager->getVoxelClusterID(),
        m_clusterManager->getRestOffsetX(), m_clusterManager->getRestOffsetY(), m_clusterManager->getRestOffsetZ(),

        d_sortedPosX, d_sortedPosY, d_sortedPosZ,
        d_sortedOldX, d_sortedOldY, d_sortedOldZ,
        d_sortedVelX, d_sortedVelY, d_sortedVelZ,
        d_sortedMass, d_sortedFriction, d_sortedColorID,
        m_clusterManager->getSortedVoxelClusterID(),
        m_clusterManager->getSortedRestOffsetX(), m_clusterManager->getSortedRestOffsetY(), m_clusterManager->getSortedRestOffsetZ()
        );

    std::swap(d_posX, d_sortedPosX);
    std::swap(d_posY, d_sortedPosY);
    std::swap(d_posZ, d_sortedPosZ);

    std::swap(d_oldX, d_sortedOldX);
    std::swap(d_oldY, d_sortedOldY);
    std::swap(d_oldZ, d_sortedOldZ);

    std::swap(d_velX, d_sortedVelX);
    std::swap(d_velY, d_sortedVelY);
    std::swap(d_velZ, d_sortedVelZ);

    std::swap(d_mass, d_sortedMass);
    std::swap(d_friction, d_sortedFriction);
    std::swap(d_colorID, d_sortedColorID);

    m_clusterManager->swapSortedBuffers();
}

void PhysicsManager::updatePhysics(float speedSimulation, float stability)
{
    if (activeVoxels == 0)
        return;

    float fixedDt = 1.0f / static_cast<float>(frameRate);
    float totalDt = fixedDt * speedSimulation;

    unsigned int substeps = std::max(1u, static_cast<unsigned int>(stability * 5));
    float subDt = totalDt / static_cast<float>(substeps);

    int solverIterations = 2;

    float gravity = -12.8f;
    float damping = 0.9999f;
    float shapeMatchingStiffness = 0.9f;
    float shapeMatchingRotateStiffness = 0.99f;
    float breakLimit = 0.6;

    for (unsigned int s = 0; s < substeps; ++s) {

        launch_predictPositions(
            d_posX, d_posY, d_posZ,
            d_oldX, d_oldY, d_oldZ,
            d_velX, d_velY, d_velZ,
            d_mass,
            activeVoxels,
            subDt,
            gravity
            );

        launch_buildSpatialHash(
            d_posX, d_posY, d_posZ,
            activeVoxels,
            m_spatialHash->getGridParticleHash(),
            m_spatialHash->getGridParticleIndex(),
            m_spatialHash->getCellStart(),
            m_spatialHash->getCellEnd(),
            m_spatialHash->getGridSize(),
            m_spatialHash->getCellSize()
            );

        sortVoxels();

        launch_shapeMatchingStep(
            d_posX, d_posY, d_posZ,
            d_mass,
            m_clusterManager->getVoxelClusterID(),
            m_clusterManager->getRestOffsetX(),
            m_clusterManager->getRestOffsetY(),
            m_clusterManager->getRestOffsetZ(),
            m_clusterManager->getClusterCM(),
            m_clusterManager->getClusterMass(),
            m_clusterManager->getClusterMatrixA(),
            m_clusterManager->getClusterRot(),
            activeVoxels,
            m_clusterManager->getNumVoxels(),
            shapeMatchingStiffness,
            shapeMatchingRotateStiffness,
            breakLimit,
            m_clusterManager->getClusterIsBraked()
            );

        for (int i = 0; i < solverIterations; ++i) {
            launch_solveCollisionsPBD(
                d_posX, d_posY, d_posZ,
                d_oldX, d_oldY, d_oldZ,
                d_mass, d_friction,
                m_clusterManager->getVoxelClusterID(),
                m_clusterManager->getClusterMass(),
                activeVoxels,
                m_spatialHash->getCellStart(),
                m_spatialHash->getCellEnd(),
                m_spatialHash->getGridSize(),
                m_spatialHash->getCellSize(),
                subDt);
        }


        launch_updateVelocities(
            d_posX, d_posY, d_posZ,
            d_oldX, d_oldY, d_oldZ,
            d_velX, d_velY, d_velZ,
            d_mass,
            activeVoxels,
            subDt,
            damping
            );
    }


    sendVoxelToOpenGL();
}

void PhysicsManager::sendVoxelToOpenGL()
{
    RenderVoxel* OpenGLVoxels = nullptr;
    size_t num_bytes;

    if (openGLConnector) {
        cudaError_t err = cudaGraphicsMapResources(1, &openGLConnector, 0);
        if (err == cudaSuccess) {
            err = cudaGraphicsResourceGetMappedPointer((void**)&OpenGLVoxels, &num_bytes, openGLConnector);
            if (err == cudaSuccess) {
                launch_copyDataToOpenGL(
                    OpenGLVoxels,
                    d_posX,
                    d_posY,
                    d_posZ,
                    d_colorID,
                    activeVoxels
                    );
            }
            cudaGraphicsUnmapResources(1, &openGLConnector, 0);
        }
    }
}


void PhysicsManager::spawnSphere(float x, float y, float z, float radius,
                                 float vx, float vy, float vz, int colorID) {
    std::vector<RenderVoxel> sphereVoxels = PhysicsHelper::generateSphereVoxels(radius, colorID);
    spawnVoxels(sphereVoxels, x, y, z, vx, vy, vz, colorID);
}

void PhysicsManager::spawnCube(float x, float y, float z, int size,
                               float vx, float vy, float vz, int colorID) {
    std::vector<RenderVoxel> cubeVoxels = PhysicsHelper::generateCubeVoxels(size, colorID);
    spawnVoxels(cubeVoxels, x, y, z, vx, vy, vz, colorID);
}

void PhysicsManager::spawnVoxels(const std::vector<RenderVoxel>& newVoxels,
                                         float offsetX, float offsetY, float offsetZ,
                                         float velX, float velY, float velZ,
                                         unsigned int colorID) {


    if (newVoxels.empty()) {
        return;
    }

    size_t count = newVoxels.size();

    while (activeVoxels + count > maxVoxels) {
        resizeMemory(count);
        resizeOpenGLBuffer();
    }

    std::vector<float> h_posX(count), h_posY(count), h_posZ(count);
    std::vector<float> h_oldX(count), h_oldY(count), h_oldZ(count);
    std::vector<float> h_velX(count), h_velY(count), h_velZ(count);
    std::vector<float> h_mass(count, 1.0f);
    std::vector<float> h_friction(count, 0.7f);
    std::vector<int> h_clusterID(count);

    int currentClusterID = ++maxClusterID;
    maxClusterID += count;

    for (size_t i = 0; i < count; ++i) {
        h_posX[i] = offsetX + newVoxels[i].x;
        h_posY[i] = offsetY + newVoxels[i].y;
        h_posZ[i] = offsetZ + newVoxels[i].z;

        h_oldX[i] = h_posX[i];
        h_oldY[i] = h_posY[i];
        h_oldZ[i] = h_posZ[i];

        h_velX[i] = velX;
        h_velY[i] = velY;
        h_velZ[i] = velZ;

        h_clusterID[i] = currentClusterID;
    }

    size_t offset = activeVoxels;

    cudaMemcpy(d_posX + offset, h_posX.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY + offset, h_posY.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ + offset, h_posZ.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_oldX + offset, h_oldX.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldY + offset, h_oldY.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldZ + offset, h_oldZ.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_velX + offset, h_velX.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY + offset, h_velY.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velZ + offset, h_velZ.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mass + offset, h_mass.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_friction + offset, h_friction.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<unsigned int> h_colorID(count, colorID);
    cudaMemcpy(d_colorID + offset, h_colorID.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMemcpy(m_clusterManager->getVoxelClusterID() + offset,
               h_clusterID.data(), count * sizeof(int), cudaMemcpyHostToDevice);

    activeVoxels += count;

    launch_initSingleClusterState(
        d_posX, d_posY, d_posZ,
        d_mass,
        m_clusterManager->getVoxelClusterID(),
        m_clusterManager->getRestOffsetX(),
        m_clusterManager->getRestOffsetY(),
        m_clusterManager->getRestOffsetZ(),
        m_clusterManager->getClusterCM(),
        m_clusterManager->getClusterMass(),
        currentClusterID,
        (unsigned)offset,
        (unsigned)count
        );
}
