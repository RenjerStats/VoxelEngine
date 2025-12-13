#include "physics/PhysicsManager.h"
#include "cuda/SpatialHash.h"
#include "cuda/ClusterManager.h"
#include "core/Types.h"
#include "cuda/KernelLauncher.h"
#include "cuda/CudaInitializer.h"
#include <cuda_runtime.h>

#include <QDebug>


PhysicsManager::PhysicsManager()
    : frameRate(60), countVoxels(0), openGLConnector(nullptr)
{
}

PhysicsManager::PhysicsManager(unsigned int frameRate, unsigned int countVoxels)
    : frameRate(frameRate), countVoxels(countVoxels) {

    if (countVoxels == 0)
        return;

    cudaMalloc(&d_posX,     countVoxels * sizeof(float));
    cudaMalloc(&d_posY,     countVoxels * sizeof(float));
    cudaMalloc(&d_posZ,     countVoxels * sizeof(float));

    cudaMalloc(&d_oldX,     countVoxels * sizeof(float));
    cudaMalloc(&d_oldY,     countVoxels * sizeof(float));
    cudaMalloc(&d_oldZ,     countVoxels * sizeof(float));

    cudaMalloc(&d_velX,     countVoxels * sizeof(float));
    cudaMalloc(&d_velY,     countVoxels * sizeof(float));
    cudaMalloc(&d_velZ,     countVoxels * sizeof(float));

    cudaMalloc(&d_mass,     countVoxels * sizeof(float));
    cudaMalloc(&d_friction, countVoxels * sizeof(float));
    cudaMalloc(&d_colorID,  countVoxels * sizeof(unsigned int));


    cudaMalloc(&d_sortedPosX, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedPosY, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedPosZ, countVoxels * sizeof(float));

    cudaMalloc(&d_sortedOldX, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedOldY, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedOldZ, countVoxels * sizeof(float));

    cudaMalloc(&d_sortedVelX, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedVelY, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedVelZ, countVoxels * sizeof(float));

    cudaMalloc(&d_sortedMass, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedFriction, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedColorID, countVoxels * sizeof(unsigned int));

    m_spatialHash = new SpatialHash();
    m_clusterManager = new ClusterManager();

    m_clusterManager->resize(countVoxels);
    m_spatialHash->resize(countVoxels);
}

PhysicsManager::PhysicsManager(PhysicsManager&& other) noexcept
    : openGLConnector(other.openGLConnector),
    frameRate(other.frameRate),
    countVoxels(other.countVoxels),
    m_spatialHash(other.m_spatialHash),
    m_clusterManager(other.m_clusterManager),
    // Основные буферы
    d_posX(other.d_posX), d_posY(other.d_posY), d_posZ(other.d_posZ),
    d_oldX(other.d_oldX), d_oldY(other.d_oldY), d_oldZ(other.d_oldZ),
    d_velX(other.d_velX), d_velY(other.d_velY), d_velZ(other.d_velZ),
    d_mass(other.d_mass), d_friction(other.d_friction),
    d_colorID(other.d_colorID),
    // Сортировочные буферы
    d_sortedPosX(other.d_sortedPosX), d_sortedPosY(other.d_sortedPosY), d_sortedPosZ(other.d_sortedPosZ),
    d_sortedOldX(other.d_sortedOldX), d_sortedOldY(other.d_sortedOldY), d_sortedOldZ(other.d_sortedOldZ),
    d_sortedVelX(other.d_sortedVelX), d_sortedVelY(other.d_sortedVelY), d_sortedVelZ(other.d_sortedVelZ),
    d_sortedMass(other.d_sortedMass), d_sortedFriction(other.d_sortedFriction),
    d_sortedColorID(other.d_sortedColorID)
{
    // Забираем владение, зануляем источник
    other.openGLConnector = nullptr;
    other.m_spatialHash = nullptr;
    other.m_clusterManager = nullptr;
    other.countVoxels = 0;

    other.d_posX = other.d_posY = other.d_posZ = nullptr;
    other.d_oldX = other.d_oldY = other.d_oldZ = nullptr;
    other.d_velX = other.d_velY = other.d_velZ = nullptr;
    other.d_mass = other.d_friction = nullptr;
    other.d_colorID = nullptr;

    // Зануляем sorted у источника
    other.d_sortedPosX = other.d_sortedPosY = other.d_sortedPosZ = nullptr;
    other.d_sortedOldX = other.d_sortedOldY = other.d_sortedOldZ = nullptr;
    other.d_sortedVelX = other.d_sortedVelY = other.d_sortedVelZ = nullptr;
    other.d_sortedMass = other.d_sortedFriction = nullptr;
    other.d_sortedColorID = nullptr;
}


PhysicsManager& PhysicsManager::operator=(PhysicsManager&& other) noexcept {
    if (this != &other) {
        freeResources(); // Освобождаем свои ресурсы перед перезаписью

        // Переносим поля
        openGLConnector = other.openGLConnector;
        frameRate = other.frameRate;
        countVoxels = other.countVoxels;
        m_spatialHash = other.m_spatialHash;
        m_clusterManager = other.m_clusterManager;

        // Переносим основные буферы
        d_posX = other.d_posX; d_posY = other.d_posY; d_posZ = other.d_posZ;
        d_oldX = other.d_oldX; d_oldY = other.d_oldY; d_oldZ = other.d_oldZ;
        d_velX = other.d_velX; d_velY = other.d_velY; d_velZ = other.d_velZ;
        d_mass = other.d_mass; d_friction = other.d_friction;
        d_colorID = other.d_colorID;

        // Переносим сортировочные буферы
        d_sortedPosX = other.d_sortedPosX; d_sortedPosY = other.d_sortedPosY; d_sortedPosZ = other.d_sortedPosZ;
        d_sortedOldX = other.d_sortedOldX; d_sortedOldY = other.d_sortedOldY; d_sortedOldZ = other.d_sortedOldZ;
        d_sortedVelX = other.d_sortedVelX; d_sortedVelY = other.d_sortedVelY; d_sortedVelZ = other.d_sortedVelZ;
        d_sortedMass = other.d_sortedMass; d_sortedFriction = other.d_sortedFriction;
        d_sortedColorID = other.d_sortedColorID;

        // Зануляем other
        other.openGLConnector = nullptr;
        other.m_spatialHash = nullptr;
        other.m_clusterManager = nullptr;
        other.countVoxels = 0;

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


void PhysicsManager::connectToOpenGL(unsigned int vboID)
{
    cuda_registerGLBuffer(vboID, &openGLConnector);

    if (!openGLConnector || countVoxels == 0) {
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
                countVoxels
                );
        }
        cudaGraphicsUnmapResources(1, &openGLConnector, 0);
    }

    initClusters();
}
void PhysicsManager::uploadVoxelsToGPU(const std::vector<RenderVoxel>& voxels)
{
    if (openGLConnector) {
        cuda_cleanup(openGLConnector);
        openGLConnector = nullptr;
    }

    countVoxels = static_cast<unsigned int>(voxels.size());
    if (countVoxels == 0)
        return;

    freeResources();

    cudaMalloc(&d_posX,     countVoxels * sizeof(float));
    cudaMalloc(&d_posY,     countVoxels * sizeof(float));
    cudaMalloc(&d_posZ,     countVoxels * sizeof(float));
    cudaMalloc(&d_oldX,     countVoxels * sizeof(float));
    cudaMalloc(&d_oldY,     countVoxels * sizeof(float));
    cudaMalloc(&d_oldZ,     countVoxels * sizeof(float));
    cudaMalloc(&d_velX,     countVoxels * sizeof(float));
    cudaMalloc(&d_velY,     countVoxels * sizeof(float));
    cudaMalloc(&d_velZ,     countVoxels * sizeof(float));
    cudaMalloc(&d_mass,     countVoxels * sizeof(float));
    cudaMalloc(&d_friction, countVoxels * sizeof(float));
    cudaMalloc(&d_colorID,  countVoxels * sizeof(unsigned int));

    cudaMalloc(&d_sortedPosX, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedPosY, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedPosZ, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedOldX, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedOldY, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedOldZ, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedVelX, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedVelY, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedVelZ, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedMass, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedFriction, countVoxels * sizeof(float));
    cudaMalloc(&d_sortedColorID, countVoxels * sizeof(unsigned int));

    // Временные host-массивы под SoA
    std::vector<float> h_posX(countVoxels), h_posY(countVoxels), h_posZ(countVoxels);
    std::vector<float> h_oldX(countVoxels), h_oldY(countVoxels), h_oldZ(countVoxels);
    std::vector<float> h_velX(countVoxels), h_velY(countVoxels), h_velZ(countVoxels);
    std::vector<float> h_mass(countVoxels), h_friction(countVoxels);
    std::vector<unsigned int> h_colorID(countVoxels);

    for (unsigned int i = 0; i < countVoxels; ++i) {
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

    cudaMemcpy(d_posX,     h_posX.data(),     countVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY,     h_posY.data(),     countVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ,     h_posZ.data(),     countVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldX,     h_oldX.data(),     countVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldY,     h_oldY.data(),     countVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldZ,     h_oldZ.data(),     countVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_velX,     h_velX.data(),     countVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY,     h_velY.data(),     countVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_velZ,     h_velZ.data(),     countVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass,     h_mass.data(),     countVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_friction, h_friction.data(), countVoxels * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_colorID,  h_colorID.data(),  countVoxels * sizeof(unsigned int), cudaMemcpyHostToDevice);


    initClusters();
}

void PhysicsManager::initClusters(){
    // 1. Build Spatial Hash (для получения отсортированного порядка)
    launch_buildSpatialHash(
        d_posX, d_posY, d_posZ, countVoxels,
        m_spatialHash->getGridParticleHash(),
        m_spatialHash->getGridParticleIndex(),
        m_spatialHash->getCellStart(),
        m_spatialHash->getCellEnd(),
        m_spatialHash->getGridSize(),
        m_spatialHash->getCellSize()
        );

    sortVoxels();

    // 2. Найти связные компоненты (заполнить d_voxelClusterID)
    // d_posX/Y/Z уже в отсортированном порядке, поэтому gridParticleIndex не нужен.
    launch_findConnectedComponents(
        d_posX, d_posY, d_posZ, d_mass,
        m_clusterManager->getVoxelClusterID(),
        m_spatialHash->getCellStart(),
        m_spatialHash->getCellEnd(),
        m_spatialHash->getGridSize(),
        m_spatialHash->getCellSize(),
        countVoxels
        );

    // 3. Рассчитать Rest State (CM и Offsets) для найденных кластеров
    launch_initClusterState(
        d_posX, d_posY, d_posZ,
        d_mass,
        m_clusterManager->getVoxelClusterID(),
        m_clusterManager->getRestOffsetX(),
        m_clusterManager->getRestOffsetY(),
        m_clusterManager->getRestOffsetZ(),
        m_clusterManager->getClusterCM(),
        m_clusterManager->getClusterMass(),
        countVoxels
        );

    std::vector<int> h_clusterID(countVoxels);
    cudaMemcpy(h_clusterID.data(),
               m_clusterManager->getVoxelClusterID(),
               countVoxels * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::unordered_set<int> roots;
    roots.reserve(countVoxels);

    for (int id : h_clusterID) {
        if (id >= 0) roots.insert(id);
    }

    qDebug() << "clusters (unique roots) =" << static_cast<int>(roots.size());
}


void PhysicsManager::sortVoxels()
{
    launch_sortVoxels(
        countVoxels,
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
    if (countVoxels == 0)
        return;

    float fixedDt = 1.0f / static_cast<float>(frameRate);
    float totalDt = fixedDt * speedSimulation;

    unsigned int substeps = std::max(1u, static_cast<unsigned int>(stability * 5));
    float subDt = totalDt / static_cast<float>(substeps);

    int solverIterations = 2;

    float gravity = -12.8f;
    float damping = 0.9999f;
    float shapeMatchingStiffness = 0.9f;

    for (unsigned int s = 0; s < substeps; ++s) {

        launch_predictPositions(
            d_posX, d_posY, d_posZ,
            d_oldX, d_oldY, d_oldZ,
            d_velX, d_velY, d_velZ,
            d_mass,
            countVoxels,
            subDt,
            gravity
            );

        launch_buildSpatialHash(
            d_posX, d_posY, d_posZ,
            countVoxels,
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
            countVoxels,
            m_clusterManager->getNumVoxels(),
            shapeMatchingStiffness
            );

        for (int i = 0; i < solverIterations; ++i) {
            launch_solveCollisionsPBD(
                d_posX, d_posY, d_posZ,
                d_oldX, d_oldY, d_oldZ,
                d_mass, d_friction,
                m_clusterManager->getVoxelClusterID(),
                countVoxels,
                m_spatialHash->getCellStart(),
                m_spatialHash->getCellEnd(),
                m_spatialHash->getGridSize(),
                m_spatialHash->getCellSize(),
                subDt
                );
        }


        // 5. Velocity Update
        launch_updateVelocities(
            d_posX, d_posY, d_posZ,
            d_oldX, d_oldY, d_oldZ,
            d_velX, d_velY, d_velZ,
            d_mass,
            countVoxels,
            subDt,
            damping
            );
    }


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
                    countVoxels
                    );
            }
            cudaGraphicsUnmapResources(1, &openGLConnector, 0);
        }
    }
}
