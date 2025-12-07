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

PhysicsManager::PhysicsManager(PhysicsManager&& other) noexcept
    : cuda_vbo_resource(other.cuda_vbo_resource)
    , frameRate(other.frameRate)
    , countVoxels(other.countVoxels)
    , m_spatialHash(std::move(other.m_spatialHash))
{
    other.cuda_vbo_resource = nullptr; // Забираем владение ресурсом
}

// Реализация перемещающего оператора присваивания
PhysicsManager& PhysicsManager::operator=(PhysicsManager&& other) noexcept {
    if (this != &other) {
        freeResources(); // Очищаем свои текущие ресурсы

        // Забираем данные
        cuda_vbo_resource = other.cuda_vbo_resource;
        frameRate = other.frameRate;
        countVoxels = other.countVoxels;
        m_spatialHash = std::move(other.m_spatialHash);

        // Обнуляем источник, чтобы его деструктор не удалил наш ресурс
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
    // Создаем объект хэша (память выделяется в конструкторе или первом resize)
    m_spatialHash = std::make_unique<SpatialHash>();
}

void PhysicsManager::updatePhysics(float speedSimulation, float stability, bool enableGravity)
{
    CudaVoxel* d_voxels;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_voxels , &num_bytes, cuda_vbo_resource);

    m_spatialHash->resize(countVoxels);
    float dt = (1.0f / frameRate) * speedSimulation;

    if (enableGravity) {
        launch_gravityKernel(dt, countVoxels, d_voxels);
    }

    for (int var = 0; var < 1; ++var) {
        launch_buildSpatialHash(
            d_voxels,
            countVoxels,
            m_spatialHash->getGridParticleHash(),
            m_spatialHash->getGridParticleIndex(),
            m_spatialHash->getCellStart(),
            m_spatialHash->getCellEnd(),
            m_spatialHash->getGridSize(),
            m_spatialHash->getCellSize()
            );

        launch_solveCollisions(
            d_voxels,
            countVoxels,
            m_spatialHash->getGridParticleIndex(), // Передаем отсортированные индексы, если нужно
            m_spatialHash->getCellStart(),
            m_spatialHash->getCellEnd(),
            m_spatialHash->getGridSize(),
            m_spatialHash->getCellSize()
            );
    }

    launch_floorKernel(0.9, countVoxels, d_voxels );

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);


    //cudaDeviceSynchronize();
    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess) {
    //    qCritical() << "CUDA registration failed:" << cudaGetErrorString(err);
    //}
}

void PhysicsManager::freeResources()
{
    cuda_cleanup(cuda_vbo_resource);
    cuda_vbo_resource = nullptr;
    m_spatialHash.reset();
}
