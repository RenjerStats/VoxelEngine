#include "cuda/KernelLauncher.h"
#include "core/Types.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// --- КОНСТАНТЫ ---
#define VOXEL_SIZE 1.0f
#define VOXEL_HALF 0.5f
#define EPSILON 1e-6f

// Порог скорости для "засыпания" контактов.
// Если скорость удара меньше 2 * G * dt (примерно), считаем это покоем.
#define RESTITUTION_THRESHOLD 2.0f

__device__ inline int getGridHashPBD(int gridPos_x, int gridPos_y, int gridPos_z, int gridSize) {
    const int p1 = 73856093;
    const int p2 = 19349663;
    const int p3 = 83492791;
    int n = p1 * gridPos_x ^ p2 * gridPos_y ^ p3 * gridPos_z;
    return n & (gridSize - 1);
}

__device__ bool checkIntersection(
    const CudaVoxel& a,
    const CudaVoxel& b,
    float3& normal,
    float& depth
    ) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;

    float x_overlap = VOXEL_SIZE - fabsf(dx);
    float y_overlap = VOXEL_SIZE - fabsf(dy);
    float z_overlap = VOXEL_SIZE - fabsf(dz);

    if (x_overlap <= 0.0f || y_overlap <= 0.0f || z_overlap <= 0.0f)
        return false;

    if (x_overlap < y_overlap && x_overlap < z_overlap) {
        normal = make_float3(dx > 0 ? 1.0f : -1.0f, 0.0f, 0.0f);
        depth = x_overlap;
    } else if (y_overlap < z_overlap) {
        normal = make_float3(0.0f, dy > 0 ? 1.0f : -1.0f, 0.0f);
        depth = y_overlap;
    } else {
        normal = make_float3(0.0f, 0.0f, dz > 0 ? 1.0f : -1.0f);
        depth = z_overlap;
    }
    return true;
}

// 1. PREDICTION STEP
__global__ void predictPositionsKernel(CudaVoxel* voxels, int count, float sub_dt, float gravityY) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    CudaVoxel v = voxels[idx];
    if (v.mass == 0.0f) return;

    v.oldX = v.x;
    v.oldY = v.y;
    v.oldZ = v.z;

    // Применяем гравитацию к текущей скорости
    v.vy += gravityY * sub_dt;

    // Предсказываем позицию
    v.x += v.vx * sub_dt;
    v.y += v.vy * sub_dt;
    v.z += v.vz * sub_dt;

    voxels[idx] = v;
}

// 2. SOLVER STEP (С исправленным отскоком и стабильностью стека)
__global__ void solveConstraintsKernel(
    CudaVoxel* voxels,
    int numVoxels,
    unsigned int* gridParticleIndex,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int gridSize,
    float cellSize,
    float sub_dt
    ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numVoxels) return;

    CudaVoxel me = voxels[idx];
    if (me.mass == 0.0f) return;

    float3 pos_correction = make_float3(0.0f, 0.0f, 0.0f);
    float3 velocity_bias = make_float3(0.0f, 0.0f, 0.0f); // Для отскока и трения
    int contact_count = 0;

    int gridX = floorf(me.x / cellSize);
    int gridY = floorf(me.y / cellSize);
    int gridZ = floorf(me.z / cellSize);

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int hash = getGridHashPBD(gridX + x, gridY + y, gridZ + z, gridSize);
                unsigned int start = cellStart[hash];
                unsigned int end = cellEnd[hash];

                for (unsigned int k = start; k < end; k++) {
                    unsigned int neighborIdx = gridParticleIndex[k];
                    if (neighborIdx == idx) continue;

                    CudaVoxel other = voxels[neighborIdx];
                    float3 normal;
                    float depth;

                    if (checkIntersection(me, other, normal, depth)) {
                        float w1 = (me.mass > 0.0f) ? 1.0f / me.mass : 0.0f;
                        float w2 = (other.mass > 0.0f) ? 1.0f / other.mass : 0.0f;
                        float wSum = w1 + w2;

                        if (wSum > EPSILON) {
                            float factor = w1 / wSum;

                            // 1. Позиционное выталкивание (anti-penetration)
                            // Это жесткое ограничение (Inequality constraint)
                            pos_correction.x += normal.x * depth * factor;
                            pos_correction.y += normal.y * depth * factor;
                            pos_correction.z += normal.z * depth * factor;

                            // 2. Расчет относительной скорости для трения и отскока
                            // Вычисляем текущую "предсказанную" скорость на основе разницы позиций
                            // (me.x - me.oldX) / sub_dt
                            float3 v_me = make_float3(
                                (me.x - me.oldX) / sub_dt,
                                (me.y - me.oldY) / sub_dt,
                                (me.z - me.oldZ) / sub_dt
                                );

                            // Для соседа (если он статический, скорость 0)
                            float3 v_other = make_float3(0.f, 0.f, 0.f);
                            if (other.mass > 0.0f) {
                                v_other = make_float3(
                                    (other.x - other.oldX) / sub_dt,
                                    (other.y - other.oldY) / sub_dt,
                                    (other.z - other.oldZ) / sub_dt
                                    );
                            }

                            float3 v_rel = make_float3(v_me.x - v_other.x, v_me.y - v_other.y, v_me.z - v_other.z);
                            float v_normal = v_rel.x * normal.x + v_rel.y * normal.y + v_rel.z * normal.z;

                            // 3. Упругость (Elasticity / Restitution)
                            // Применяем ТОЛЬКО если скорость столкновения выше порога.
                            // Это предотвращает дрожание стека (catapult effect).
                            // Если v_normal < 0, значит тела сближаются
                            if (v_normal < -RESTITUTION_THRESHOLD) {
                                float restitution = me.elasticity * other.elasticity;
                                // Формула отскока: изменяем oldPos, чтобы solver "почувствовал" изменение скорости
                                // Импульс скорости: j = - (1 + e) * v_normal
                                // Мы применяем это как смещение oldPos назад
                                float restitution_bias = -v_normal * restitution;

                                velocity_bias.x += normal.x * restitution_bias;
                                velocity_bias.y += normal.y * restitution_bias;
                                velocity_bias.z += normal.z * restitution_bias;
                            }

                            // 4. Трение (Friction)
                            float3 v_tangent = make_float3(
                                v_rel.x - v_normal * normal.x,
                                v_rel.y - v_normal * normal.y,
                                v_rel.z - v_normal * normal.z
                                );
                            float friction_coeff = me.friction * other.friction;

                            // Трение всегда пытается остановить тангенциальное движение
                            velocity_bias.x += v_tangent.x * friction_coeff;
                            velocity_bias.y += v_tangent.y * friction_coeff;
                            velocity_bias.z += v_tangent.z * friction_coeff;

                            contact_count++;
                        }
                    }
                }
            }
        }
    }

    if (contact_count > 0) {
        // Усреднение коррекций - важно для стабильности, когда контактов много
        float inv_contact = 1.0f / (float)contact_count;

        // Применяем выталкивание к текущей позиции
        me.x += pos_correction.x * inv_contact;
        me.y += pos_correction.y * inv_contact;
        me.z += pos_correction.z * inv_contact;

        // Применяем отскок и трение к OLD позиции.
        // PBD трюк: изменение oldPos напрямую влияет на вычисленную скорость на следующем шаге.
        // x_new не меняем (чтобы не нарушить контакт), а old_x сдвигаем так,
        // чтобы вектор (x_new - x_old) стал нужной нам скоростью отскока.
        me.oldX -= velocity_bias.x * inv_contact * sub_dt;
        me.oldY -= velocity_bias.y * inv_contact * sub_dt;
        me.oldZ -= velocity_bias.z * inv_contact * sub_dt;

        voxels[idx] = me;
    }
}

// 3. VELOCITY UPDATE
__global__ void updateVelocitiesKernel(CudaVoxel* voxels, int count, float sub_dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    CudaVoxel v = voxels[idx];
    if (v.mass == 0.0f) {
        // Для статики обнуляем всё явно, на всякий случай
        v.vx = 0; v.vy = 0; v.vz = 0;
        voxels[idx] = v;
        return;
    }

    // PBD Velocity update: v = (x - oldX) / dt
    // Благодаря правкам oldX в солвере, здесь уже учтен отскок.
    v.vx = (v.x - v.oldX) / sub_dt;
    v.vy = (v.y - v.oldY) / sub_dt;
    v.vz = (v.z - v.oldZ) / sub_dt;

    // Небольшое глобальное затухание для стабильности
    const float global_damping = 0.9999f;
    v.vx *= global_damping;
    v.vy *= global_damping;
    v.vz *= global_damping;

    voxels[idx] = v;
}

extern "C" {
void launch_updatePhysicsPBD(CudaVoxel* d_voxels, unsigned int numVoxels, float dt, unsigned int substeps,
                             unsigned int* d_gridParticleHash, unsigned int* d_gridParticleIndex,
                             unsigned int* d_cellStart, unsigned int* d_cellEnd, unsigned int gridSize, float cellSize) {

    if (numVoxels == 0) return;
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;
    float sub_dt = dt / (float)substeps;
    float gravityY = -9.81f;

    for (unsigned int s = 0; s < substeps; s++) {
        predictPositionsKernel<<<blocks, threads>>>(d_voxels, numVoxels, sub_dt, gravityY);
        solveConstraintsKernel<<<blocks, threads>>>(d_voxels, numVoxels, d_gridParticleIndex, d_cellStart, d_cellEnd, gridSize, cellSize, sub_dt);
        updateVelocitiesKernel<<<blocks, threads>>>(d_voxels, numVoxels, sub_dt);
    }
}

} // extern "C"
