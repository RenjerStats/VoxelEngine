#include "cuda/VoxelData.cuh"
#include "cuda/CudaUtils.cuh"

namespace CudaPhysics {

VoxelDataManager::VoxelDataManager()
    : m_allocated(false) {
    std::memset(&m_palette, 0, sizeof(VoxelPalette));
}

VoxelDataManager::~VoxelDataManager() {
    release();
}

bool VoxelDataManager::allocate(uint32_t voxelCount) {
    if (voxelCount == 0) return false;

    // Освобождаем старую память, если была
    release();

    size_t floatSize = voxelCount * sizeof(float);
    size_t uint32Size = voxelCount * sizeof(uint32_t);

    // Выделяем память для всех массивов
    cudaError_t err;

    err = cudaMalloc(&m_deviceData.posX, floatSize);
    if (err != cudaSuccess) return false;

    err = cudaMalloc(&m_deviceData.posY, floatSize);
    if (err != cudaSuccess) { release(); return false; }

    err = cudaMalloc(&m_deviceData.posZ, floatSize);
    if (err != cudaSuccess) { release(); return false; }

    err = cudaMalloc(&m_deviceData.velX, floatSize);
    if (err != cudaSuccess) { release(); return false; }

    err = cudaMalloc(&m_deviceData.velY, floatSize);
    if (err != cudaSuccess) { release(); return false; }

    err = cudaMalloc(&m_deviceData.velZ, floatSize);
    if (err != cudaSuccess) { release(); return false; }

    err = cudaMalloc(&m_deviceData.colorIndex, uint32Size);
    if (err != cudaSuccess) { release(); return false; }

    err = cudaMalloc(&m_deviceData.mass, floatSize);
    if (err != cudaSuccess) { release(); return false; }

    m_deviceData.count = voxelCount;
    m_allocated = true;

    return true;
}

bool VoxelDataManager::uploadVoxels(const float* posX, const float* posY, const float* posZ,
                                    const uint32_t* colorIndex, uint32_t count) {
    if (!m_allocated || count != m_deviceData.count) {
        return false;
    }

    size_t floatSize = count * sizeof(float);
    size_t uint32Size = count * sizeof(uint32_t);

    // Копируем данные на GPU
    cudaError_t err;

    err = cudaMemcpy(m_deviceData.posX, posX, floatSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return false;

    err = cudaMemcpy(m_deviceData.posY, posY, floatSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return false;

    err = cudaMemcpy(m_deviceData.posZ, posZ, floatSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return false;

    err = cudaMemcpy(m_deviceData.colorIndex, colorIndex, uint32Size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return false;

    return true;
}

bool VoxelDataManager::uploadPalette(const uint32_t* colors) {
    if (!colors) return false;

    std::memcpy(m_palette.colors, colors, sizeof(VoxelPalette));
    return true;
}

// CUDA kernel для инициализации физических параметров
__global__ void initPhysicsKernel(float* velX, float* velY, float* velZ,
                                   float* mass, uint32_t count, float defaultMass) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Инициализируем нулевыми скоростями
    velX[idx] = 0.0f;
    velY[idx] = 0.0f;
    velZ[idx] = 0.0f;

    // Устанавливаем массу
    mass[idx] = defaultMass;
}

void VoxelDataManager::initializePhysics(float defaultMass) {
    if (!m_allocated) return;

    const int blockSize = 256;
    const int numBlocks = (m_deviceData.count + blockSize - 1) / blockSize;

    initPhysicsKernel<<<numBlocks, blockSize>>>(
        m_deviceData.velX, m_deviceData.velY, m_deviceData.velZ,
        m_deviceData.mass, m_deviceData.count, defaultMass
    );

    cudaDeviceSynchronize();
}

void VoxelDataManager::release() {
    if (m_deviceData.posX) cudaFree(m_deviceData.posX);
    if (m_deviceData.posY) cudaFree(m_deviceData.posY);
    if (m_deviceData.posZ) cudaFree(m_deviceData.posZ);
    if (m_deviceData.velX) cudaFree(m_deviceData.velX);
    if (m_deviceData.velY) cudaFree(m_deviceData.velY);
    if (m_deviceData.velZ) cudaFree(m_deviceData.velZ);
    if (m_deviceData.colorIndex) cudaFree(m_deviceData.colorIndex);
    if (m_deviceData.mass) cudaFree(m_deviceData.mass);

    m_deviceData = VoxelsSoA();
    m_allocated = false;
}

}
