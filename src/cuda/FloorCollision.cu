#include "cuda/KernelLauncher.h"
#include "core/Types.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <GL/gl.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Константы для физики пола
#define FLOOR_Y -3.0f
// Предполагаем, что размер вокселя 1.0f, значит половина (радиус) = 0.5f.
// Если pivot point в центре вокселя, то столкновение происходит на FLOOR_Y + 0.5f
#define VOXEL_RADIUS 0.5f

__global__ void floorKernel(CudaVoxel* voxels, int count, float floorElasticity) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    CudaVoxel v = voxels[idx];
    if (v.mass == 0.0f) return;

    // Проверка столкновения (учитываем радиус, чтобы воксель стоял НА полу, а не центром В полу)
    float collisionPlane = FLOOR_Y + VOXEL_RADIUS;

    if (v.y < collisionPlane) {
        // 1. Коррекция позиции (Projection)
        // Жестко выталкиваем воксель на поверхность, чтобы избежать проваливания
        v.y = collisionPlane;

        // 2. Отражение скорости (Response)
        // Отражаем только если скорость направлена вниз
        if (v.vy < 0.0f) {
            // Итоговая упругость = упругость вокселя * упругость пола
            float combinedElasticity = v.elasticity * floorElasticity;
            v.vy = -v.vy * combinedElasticity;

            // 3. Трение (Friction)
            // Применяем трение к горизонтальным осям при контакте
            // Чем больше v.friction, тем сильнее торможение (0.0 - нет трения, 1.0 - мгновенная остановка)
            float inverseFriction = 1.0f - v.friction;
            // Защита от отрицательных значений
            if (inverseFriction < 0.0f) inverseFriction = 0.0f;

            v.vx *= inverseFriction;
            v.vz *= inverseFriction;
        }
    }

    // Записываем обновленные данные обратно в глобальную память
    voxels[idx] = v;
}

extern "C" {
void launch_floorKernel(float floorElasticity, size_t numVoxels, CudaVoxel* d_ptr) {
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;
    floorKernel<<<blocks, threads>>>(d_ptr, numVoxels, floorElasticity);
}
}
