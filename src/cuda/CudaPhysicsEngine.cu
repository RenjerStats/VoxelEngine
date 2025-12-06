#define CUDA_CHECK(call) \
do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

#include "cuda/CudaPhysicsEngine.h"
#include "core/Types.h"

// --- ДОБАВИТЬ ЭТОТ БЛОК ---
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

// Подключаем стандартный GL заголовок, чтобы nvcc узнал, что такое GLuint
#include <GL/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>

// Глобальный ресурс, связывающий графику и вычисления
cudaGraphicsResource* cuda_vbo_resource = nullptr;

// --- KERNEL (Выполняется на GPU) ---
__global__ void gravityKernel(CudaVoxel* voxels, int count, float dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;
    if (voxels[idx].mass == 0) return; // неподвижные блоки

    // Простая физика: падаем вниз
    voxels[idx].vy -= dt / 10;
    if(voxels[idx].vy < -0.5) voxels[idx].vy = -0.5;

    voxels[idx].y += voxels[idx].vy;

    if (voxels[idx].y < -1.0f) {
        voxels[idx].y = 100.0f;
    }
}

// --- HOST FUNCTIONS (Вызываются из C++) ---

// 1. Регистрация буфера OpenGL в CUDA
void cuda_registerGLBuffer(unsigned int vboID) {
    if (cuda_vbo_resource != nullptr) {
        cudaGraphicsUnregisterResource(cuda_vbo_resource);
        cuda_vbo_resource = nullptr;
    }

    // Регистрируем ТОЛЬКО ОДИН РАЗ
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_vbo_resource,
        vboID,
        cudaGraphicsMapFlagsWriteDiscard
        ));
}

// 2. Шаг симуляции
void cuda_runSimulation(float dt, size_t numVoxels) {
    CudaVoxel* d_ptr;
    size_t num_bytes;

    // А. "Картируем" ресурс (блокируем доступ OpenGL, открываем для CUDA)
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);

    // Б. Получаем сырой указатель на память GPU
    cudaGraphicsResourceGetMappedPointer((void**)&d_ptr, &num_bytes, cuda_vbo_resource);

    // В. Запускаем ядро
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;
    gravityKernel<<<blocks, threads>>>(d_ptr, numVoxels, dt);

    // Г. "Разкартируем" (возвращаем доступ OpenGL)
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}

// 3. Очистка
void cuda_cleanup() {
    if (cuda_vbo_resource) {
        cudaGraphicsUnregisterResource(cuda_vbo_resource);
    }
}
