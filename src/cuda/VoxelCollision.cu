#include "cuda/KernelLauncher.h"
#include "core/Types.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// --- КОНСТАНТЫ ---
#define VOXEL_SIZE 1.0f
#define VOXEL_HALF_SIZE 0.5f

// Physics constants
#define MAX_CONTACT_ITERATIONS 5  // Количество итераций решения контактов
#define CONTACT_BIAS 0.005f       // Слип-толеранс (penetration allowance)
#define CONTACT_BAUMGARTE 0.2f    // Baumgarte stabilization factor (0.1-0.3)
#define MIN_RELATIVE_VELOCITY 0.5f // Минимальная скорость для трения

// Структура контактной информации
struct Contact {
    float3 normal;           // Нормаль контакта
    float penetration_depth; // Глубина проникновения
    float relative_velocity; // Относительная скорость вдоль нормали
    int voxel_a_idx;        // Индекс первого вокселя
    int voxel_b_idx;        // Индекс второго вокселя
    bool is_valid;          // Валидна ли контактная точка
};

/**
 * Хеш-функция для пространственного разбиения
 */
__device__ inline int calcGridHash_device(int gridPos_x, int gridPos_y, int gridPos_z, int gridSize) {
    const int p1 = 73856093;
    const int p2 = 19349663;
    const int p3 = 83492791;
    int n = p1 * gridPos_x ^ p2 * gridPos_y ^ p3 * gridPos_z;
    return n & (gridSize - 1);
}

/**
 * Вычисление вектора от объекта A к B
 */
__device__ inline float3 normalized(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-6f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ inline float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * Проверка AABB коллизии и извлечение данных контакта
 * Основано на SAT (Separating Axis Theorem) для кубов одинакового размера
 */
__device__ Contact checkVoxelContact(
    const CudaVoxel& a,
    const CudaVoxel& b,
    int idx_a,
    int idx_b
    ) {
    Contact contact;
    contact.voxel_a_idx = idx_a;
    contact.voxel_b_idx = idx_b;
    contact.is_valid = false;

    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;

    float x_overlap = VOXEL_SIZE - fabsf(dx);
    float y_overlap = VOXEL_SIZE - fabsf(dy);
    float z_overlap = VOXEL_SIZE - fabsf(dz);

    // Нет коллизии если нет перекрытия по какой-то оси
    if (x_overlap <= 0.0f || y_overlap <= 0.0f || z_overlap <= 0.0f) {
        return contact;
    }

    // Находим ось минимального проникновения (MTV)
    float min_overlap = fminf(x_overlap, fminf(y_overlap, z_overlap));

    if (x_overlap == min_overlap) {
        contact.normal = make_float3(dx > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
        contact.penetration_depth = x_overlap;
    } else if (y_overlap == min_overlap) {
        contact.normal = make_float3(0.0f, dy > 0.0f ? 1.0f : -1.0f, 0.0f);
        contact.penetration_depth = y_overlap;
    } else {
        contact.normal = make_float3(0.0f, 0.0f, dz > 0.0f ? 1.0f : -1.0f);
        contact.penetration_depth = z_overlap;
    }

    // Вычисляем относительную скорость
    float rel_vx = a.vx - b.vx;
    float rel_vy = a.vy - b.vy;
    float rel_vz = a.vz - b.vz;
    contact.relative_velocity = rel_vx * contact.normal.x +
                                rel_vy * contact.normal.y +
                                rel_vz * contact.normal.z;

    contact.is_valid = true;
    return contact;
}

/**
 * Применение импульса (velocity-level constraint)
 * Использует коэффициент восстановления (restitution)
 */
__device__ void applyCollisionImpulse(
    CudaVoxel& voxel_a,
    CudaVoxel& voxel_b,
    const Contact& contact,
    float friction_coeff
    ) {
    // Если уже расходятся - не решаем
    if (contact.relative_velocity > 0.0f) {
        return;
    }

    float invMass_a = (voxel_a.mass > 0.0f) ? 1.0f / voxel_a.mass : 0.0f;
    float invMass_b = (voxel_b.mass > 0.0f) ? 1.0f / voxel_b.mass : 0.0f;
    float invMass_sum = invMass_a + invMass_b;

    // Если оба статичны - ничего не делаем
    if (invMass_sum < 1e-6f) {
        return;
    }

    // Коэффициент восстановления (эластичность)
    float e = fminf(voxel_a.elasticity, voxel_b.elasticity);

    // Нормальный импульс
    float impulse = -(1.0f + e) * contact.relative_velocity / invMass_sum;

    // Применяем нормальный импульс
    float3 impulse_vec = make_float3(
        contact.normal.x * impulse,
        contact.normal.y * impulse,
        contact.normal.z * impulse
        );

    voxel_a.vx += impulse_vec.x * invMass_a;
    voxel_a.vy += impulse_vec.y * invMass_a;
    voxel_a.vz += impulse_vec.z * invMass_a;

    voxel_b.vx -= impulse_vec.x * invMass_b;
    voxel_b.vy -= impulse_vec.y * invMass_b;
    voxel_b.vz -= impulse_vec.z * invMass_b;

    // === FRICTION (Coulomb model) ===
    // Вычисляем тангенциальную скорость
    float rel_vx = voxel_a.vx - voxel_b.vx;
    float rel_vy = voxel_a.vy - voxel_b.vy;
    float rel_vz = voxel_a.vz - voxel_b.vz;

    float vel_along_normal = rel_vx * contact.normal.x +
                             rel_vy * contact.normal.y +
                             rel_vz * contact.normal.z;

    float3 tangent_vel = make_float3(
        rel_vx - vel_along_normal * contact.normal.x,
        rel_vy - vel_along_normal * contact.normal.y,
        rel_vz - vel_along_normal * contact.normal.z
        );

    float tangent_speed = sqrtf(tangent_vel.x * tangent_vel.x +
                                tangent_vel.y * tangent_vel.y +
                                tangent_vel.z * tangent_vel.z);

    // Если есть тангенциальное движение
    if (tangent_speed > MIN_RELATIVE_VELOCITY) {
        float3 tangent_dir = normalized(tangent_vel);

        // Коэффициент трения (среднее между двумя объектами)
        float mu = (voxel_a.friction + voxel_b.friction) * 0.5f;

        // Friction impulse (ограничиваем friction cone: |f| <= mu * |n|)
        float friction_impulse = -tangent_speed / invMass_sum;
        friction_impulse = fmaxf(-mu * fabsf(impulse),
                                 fminf(friction_impulse, mu * fabsf(impulse)));

        float3 friction_vec = make_float3(
            tangent_dir.x * friction_impulse,
            tangent_dir.y * friction_impulse,
            tangent_dir.z * friction_impulse
            );

        voxel_a.vx += friction_vec.x * invMass_a;
        voxel_a.vy += friction_vec.y * invMass_a;
        voxel_a.vz += friction_vec.z * invMass_a;

        voxel_b.vx -= friction_vec.x * invMass_b;
        voxel_b.vy -= friction_vec.y * invMass_b;
        voxel_b.vz -= friction_vec.z * invMass_b;
    }
}

/**
 * Позиционная коррекция (position-level constraint)
 * Решает проникновения, которые остались после velocity-level
 */
__device__ void applyPositionalCorrection(
    CudaVoxel& voxel_a,
    CudaVoxel& voxel_b,
    const Contact& contact
    ) {
    if (contact.penetration_depth <= CONTACT_BIAS) {
        return; // Допустимое проникновение
    }

    float invMass_a = (voxel_a.mass > 0.0f) ? 1.0f / voxel_a.mass : 0.0f;
    float invMass_b = (voxel_b.mass > 0.0f) ? 1.0f / voxel_b.mass : 0.0f;
    float invMass_sum = invMass_a + invMass_b;

    if (invMass_sum < 1e-6f) {
        return;
    }

    // Слип-толеранс: не корректируем если проникновение меньше этого
    float penetration_to_correct = fmaxf(0.0f, contact.penetration_depth - CONTACT_BIAS);

    // Baumgarte техника: слегка переиспользуем корректировку
    float correction_scalar = CONTACT_BAUMGARTE * penetration_to_correct / invMass_sum;
    correction_scalar = fminf(correction_scalar, VOXEL_HALF_SIZE);

    float3 correction = make_float3(
        contact.normal.x * correction_scalar,
        contact.normal.y * correction_scalar,
        contact.normal.z * correction_scalar
        );

    voxel_a.x += correction.x * invMass_a;
    voxel_a.y += correction.y * invMass_a;
    voxel_a.z += correction.z * invMass_a;

    voxel_b.x -= correction.x * invMass_b;
    voxel_b.y -= correction.y * invMass_b;
    voxel_b.z -= correction.z * invMass_b;
}

/**
 * Кернел обнаружения контактов
 * Собирает все контакты в глобальный буфер
 */
__global__ void detectContactsKernel(
    CudaVoxel* voxels,
    unsigned int numVoxels,
    unsigned int* gridParticleIndex,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int gridSize,
    float cellSize,
    Contact* contacts,
    unsigned int* contact_count,
    unsigned int max_contacts
    ) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;

    CudaVoxel me = voxels[idx];
    if (me.mass == 0.0f) return; // Пропускаем статичные объекты при обнаружении

    int gridPos_x = floorf(me.x / cellSize);
    int gridPos_y = floorf(me.y / cellSize);
    int gridPos_z = floorf(me.z / cellSize);

    // Проходим по соседним ячейкам 3x3x3
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int neighborHash = calcGridHash_device(gridPos_x + x, gridPos_y + y, gridPos_z + z, gridSize);
                unsigned int start = cellStart[neighborHash];
                unsigned int end = cellEnd[neighborHash];

                if (start == 0xFFFFFFFF) continue;

                for (unsigned int k = start; k < end; k++) {
                    unsigned int neighborIdx = gridParticleIndex[k];
                    if (neighborIdx <= idx) continue; // Избегаем дубликатов (только idx < neighborIdx)

                    CudaVoxel other = voxels[neighborIdx];
                    Contact contact = checkVoxelContact(me, other, idx, neighborIdx);

                    if (contact.is_valid) {
                        // Безопасно добавляем контакт в глобальный список
                        unsigned int pos = atomicInc(contact_count, max_contacts - 1);
                        if (pos < max_contacts) {
                            contacts[pos] = contact;
                        }
                    }
                }
            }
        }
    }
}

/**
 * Кернел решения контактов (итеративный)
 * Применяет impulse и positional correction
 */
__global__ void solveContactsKernel(
    CudaVoxel* voxels,
    Contact* contacts,
    unsigned int num_contacts,
    int iteration
    ) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_contacts) return;

    Contact contact = contacts[idx];
    if (!contact.is_valid) return;

    CudaVoxel voxel_a = voxels[contact.voxel_a_idx];
    CudaVoxel voxel_b = voxels[contact.voxel_b_idx];

    // Velocity level (все итерации)
    applyCollisionImpulse(voxel_a, voxel_b, contact, 0.3f);

    // Position level (только последняя итерация для стабильности)
    if (iteration == MAX_CONTACT_ITERATIONS - 1) {
        applyPositionalCorrection(voxel_a, voxel_b, contact);
    }

    voxels[contact.voxel_a_idx] = voxel_a;
    voxels[contact.voxel_b_idx] = voxel_b;
}

/**
 * Главный кернел разрешения коллизий
 * Интегрирует обнаружение и решение
 */
__global__ void solveCollisionsIterativeKernel(
    CudaVoxel* voxels,
    unsigned int numVoxels,
    unsigned int* gridParticleIndex,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int gridSize,
    float cellSize,
    Contact* contacts_buffer,
    unsigned int* contact_count
    ) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVoxels) return;

    CudaVoxel me = voxels[idx];
    if (me.mass == 0.0f) return;

    int gridPos_x = floorf(me.x / cellSize);
    int gridPos_y = floorf(me.y / cellSize);
    int gridPos_z = floorf(me.z / cellSize);


    // Временный буфер для локальных контактов этого треда
    Contact local_contacts[27]; // Максимум 27 соседей
    int local_contact_count = 0;



    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int neighborHash = calcGridHash_device(gridPos_x + x, gridPos_y + y, gridPos_z + z, gridSize);
                unsigned int start = cellStart[neighborHash];
                unsigned int end = cellEnd[neighborHash];

                if (start == 0xFFFFFFFF) continue;

                for (unsigned int k = start; k < end; k++) {
                    unsigned int neighborIdx = gridParticleIndex[k];
                    if (neighborIdx == idx) continue;

                    CudaVoxel other = voxels[neighborIdx];
                    Contact contact = checkVoxelContact(me, other, idx, neighborIdx);

                    if (contact.is_valid && local_contact_count < 27) {
                        local_contacts[local_contact_count++] = contact;
                    }
                }
            }
        }
    }

    // Итеративное решение контактов
    for (int iter = 0; iter < MAX_CONTACT_ITERATIONS; iter++) {
        for (int i = 0; i < local_contact_count; i++) {
            Contact& contact = local_contacts[i];

            // Перечитываем актуальные данные вокселей
            CudaVoxel voxel_a = voxels[contact.voxel_a_idx];
            CudaVoxel voxel_b = voxels[contact.voxel_b_idx];

            // Переживите контактные данные
            contact.relative_velocity = (voxel_a.vx - voxel_b.vx) * contact.normal.x +
                                        (voxel_a.vy - voxel_b.vy) * contact.normal.y +
                                        (voxel_a.vz - voxel_b.vz) * contact.normal.z;

            applyCollisionImpulse(voxel_a, voxel_b, contact, 0.3f);

            if (iter == MAX_CONTACT_ITERATIONS - 1) {
                applyPositionalCorrection(voxel_a, voxel_b, contact);
            }

            voxels[contact.voxel_a_idx] = voxel_a;
            voxels[contact.voxel_b_idx] = voxel_b;
        }
    }
}

// ============= WRAPPER ФУНКЦИИ =============

extern "C" {

void launch_solveCollisions(
    CudaVoxel* d_voxels,
    unsigned int numVoxels,
    unsigned int* d_gridParticleIndex,
    unsigned int* d_cellStart,
    unsigned int* d_cellEnd,
    unsigned int gridSize,
    float cellSize
    ) {
    if (numVoxels == 0) return;

    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;

    // Запускаем итеративное решение
    solveCollisionsIterativeKernel<<<blocks, threads>>>(
        d_voxels,
        numVoxels,
        d_gridParticleIndex,
        d_cellStart,
        d_cellEnd,
        gridSize,
        cellSize,
        nullptr,  // contacts_buffer (не используется в этой версии)
        nullptr   // contact_count
        );

    cudaDeviceSynchronize();
}

}
