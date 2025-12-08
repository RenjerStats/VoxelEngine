#include "physics/PhysicsManager.h"
#include "cuda/SpatialHash.h"
#include "core/Types.h"
#include "cuda/KernelLauncher.h"
#include "cuda/CudaInitializer.h"
#include <cuda_runtime.h>

#include <QDebug>


PhysicsManager::PhysicsManager()
    : frameRate(60), countVoxels(0), cuda_vbo_resource(nullptr)
{
}

// Конструктор с параметрами
PhysicsManager::PhysicsManager(unsigned int frameRate, unsigned int countVoxels)
    : frameRate(frameRate), countVoxels(countVoxels), cuda_vbo_resource(nullptr)
{
}

// Деструктор (очень важен для unique_ptr!)
PhysicsManager::~PhysicsManager() {
    freeResources();
}

void PhysicsManager::freeResources()
{
    cuda_cleanup(cuda_vbo_resource);
    cuda_vbo_resource = nullptr;
    m_spatialHash.reset();
}

PhysicsManager::PhysicsManager(PhysicsManager&& other) noexcept
    : cuda_vbo_resource(other.cuda_vbo_resource),
    frameRate(other.frameRate),
    countVoxels(other.countVoxels),
    m_spatialHash(std::move(other.m_spatialHash)) {
    other.cuda_vbo_resource = nullptr;
}

PhysicsManager& PhysicsManager::operator=(PhysicsManager&& other) noexcept {
    if (this != &other) {
        freeResources();

        cuda_vbo_resource = other.cuda_vbo_resource;
        frameRate = other.frameRate;
        countVoxels = other.countVoxels;
        m_spatialHash = std::move(other.m_spatialHash);

        other.cuda_vbo_resource = nullptr;
    }
    return *this;
}


void PhysicsManager::registerVoxelSharedBuffer(unsigned int vboID)
{
    cuda_registerGLBuffer(vboID, &cuda_vbo_resource);
}

void PhysicsManager::initSumulation(bool withVoxelConnection, bool withVoxelCollision)
{
    m_spatialHash = std::make_unique<SpatialHash>();
}

void PhysicsManager::updatePhysics(float speedSimulation, float stability)
{
    CudaVoxel* d_voxels;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_voxels , &num_bytes, cuda_vbo_resource);

    m_spatialHash->resize(countVoxels);
    float dt = (1.0f / frameRate) * speedSimulation;
    unsigned int substeps =  1 + (unsigned int)(stability * 5);

    launch_buildSpatialHash(
        d_voxels, countVoxels,
        m_spatialHash->getGridParticleHash(),
        m_spatialHash->getGridParticleIndex(),
        m_spatialHash->getCellStart(),
        m_spatialHash->getCellEnd(),
        m_spatialHash->getGridSize(),
        m_spatialHash->getCellSize()
        );

    launch_updatePhysicsPBD(
        d_voxels,
        countVoxels,
        dt,
        substeps,
        m_spatialHash->getGridParticleHash(),
        m_spatialHash->getGridParticleIndex(),
        m_spatialHash->getCellStart(),
        m_spatialHash->getCellEnd(),
        m_spatialHash->getGridSize(),
        m_spatialHash->getCellSize()
        );

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);


    //cudaDeviceSynchronize();
    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess) {
    //    qCritical() << "CUDA registration failed:" << cudaGetErrorString(err);
    //}
}
