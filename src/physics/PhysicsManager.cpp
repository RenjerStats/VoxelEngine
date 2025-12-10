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

    if (cuda_vbo_resource) {
        cuda_cleanup(cuda_vbo_resource);
        cuda_vbo_resource = nullptr;
    }


    if (d_standalone_buffer) {
        cudaFree(d_standalone_buffer);
        d_standalone_buffer = nullptr;
    }

    m_spatialHash.reset();
}

PhysicsManager::PhysicsManager(PhysicsManager&& other) noexcept
    : cuda_vbo_resource(other.cuda_vbo_resource),
    d_standalone_buffer(other.d_standalone_buffer),
    frameRate(other.frameRate),
    countVoxels(other.countVoxels),
    m_spatialHash(std::move(other.m_spatialHash))
{
    other.cuda_vbo_resource = nullptr;
    other.d_standalone_buffer = nullptr;
}

PhysicsManager& PhysicsManager::operator=(PhysicsManager&& other) noexcept {
    if (this != &other) {
        freeResources();

        cuda_vbo_resource = other.cuda_vbo_resource;
        d_standalone_buffer = other.d_standalone_buffer;
        frameRate = other.frameRate;
        countVoxels = other.countVoxels;
        m_spatialHash = std::move(other.m_spatialHash);

        other.cuda_vbo_resource = nullptr;
        other.d_standalone_buffer = nullptr;
    }
    return *this;
}


void PhysicsManager::registerVoxelSharedBuffer(unsigned int vboID)
{
    if (d_standalone_buffer) { cudaFree(d_standalone_buffer); d_standalone_buffer = nullptr; }
    cuda_registerGLBuffer(vboID, &cuda_vbo_resource);
}

void PhysicsManager::uploadVoxelsToGPU(const std::vector<CudaVoxel>& voxels) {
    if (cuda_vbo_resource) {
        // Если был GL ресурс, освобождаем
        cuda_cleanup(cuda_vbo_resource);
        cuda_vbo_resource = nullptr;
    }

    countVoxels = voxels.size();
    if (countVoxels == 0) return;

    // Выделяем память напрямую через CUDA
    cudaError_t err = cudaMalloc((void**)&d_standalone_buffer, countVoxels * sizeof(CudaVoxel));
    if (err != cudaSuccess) {
        qCritical() << "CUDA Malloc failed:" << cudaGetErrorString(err);
        return;
    }

    // Копируем данные
    cudaMemcpy(d_standalone_buffer, voxels.data(), countVoxels * sizeof(CudaVoxel), cudaMemcpyHostToDevice);
}

void PhysicsManager::initSumulation(bool withVoxelConnection, bool withVoxelCollision)
{
    m_spatialHash = std::make_unique<SpatialHash>();
}

void PhysicsManager::updatePhysics(float speedSimulation, float stability)
{
    if (countVoxels == 0) return;

    CudaVoxel* d_voxels;
    size_t num_bytes;
    bool usingGL = cuda_vbo_resource != nullptr;


    if (usingGL) {
        cudaError_t err = cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        if (err != cudaSuccess) return;
        cudaGraphicsResourceGetMappedPointer((void**)&d_voxels , &num_bytes, cuda_vbo_resource);
    } else if (d_standalone_buffer) {
        d_voxels = d_standalone_buffer;
    } else {
        return;
    }


    float fixedDt = 1.0f / (float)frameRate;
    float totalDt = fixedDt * speedSimulation;

    // PBD требует малых шагов для жестких контактов
    unsigned int substeps = std::max(1u, (unsigned int)(stability * 5));
    float subDt = totalDt / (float)substeps;

    // Итерации солвера внутри одного сабстепа (Solver Iterations)
    // Для вокселей часто хватает 1-2 итераций если много сабстепов.
    int solverIterations = 2;

    m_spatialHash->resize(countVoxels);
    float gravity = -15.8f;
    float damping = 0.9999f; // Глобальное затухание

    // --- MAIN PBD LOOP ---
    for (unsigned int s = 0; s < substeps; s++) {

        // 1. Prediction (Apply Gravity, x* = x + v*dt)
        launch_predictPositions(d_voxels, countVoxels, subDt, gravity);

        // 2. Broad Phase (Spatial Hash)
        // Хеш нужно перестраивать, т.к. позиции (x*) изменились
        launch_buildSpatialHash(
            d_voxels, countVoxels,
            m_spatialHash->getGridParticleHash(),
            m_spatialHash->getGridParticleIndex(),
            m_spatialHash->getCellStart(),
            m_spatialHash->getCellEnd(),
            m_spatialHash->getGridSize(),
            m_spatialHash->getCellSize()
            );

        // 3. Constraint Solver Loop
        for (int i = 0; i < solverIterations; i++) {
            launch_solveCollisionsPBD(
                d_voxels,
                countVoxels,
                m_spatialHash->getGridParticleIndex(),
                m_spatialHash->getCellStart(),
                m_spatialHash->getCellEnd(),
                m_spatialHash->getGridSize(),
                m_spatialHash->getCellSize(),
                subDt
                );
        }



        // 4. Velocity Update (v = (x* - x_old) / dt)
        launch_updateVelocities(d_voxels, countVoxels, subDt, damping);
    }

    if (usingGL) {
        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    }
}
