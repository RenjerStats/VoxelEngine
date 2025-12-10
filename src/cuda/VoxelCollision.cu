#include "cuda/KernelLauncher.h"
#include "core/Types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <nvtx3/nvToolsExt.h>

__device__ inline float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(const float3& a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ inline float3 operator/(const float3& a, float b) { return make_float3(a.x / b, a.y / b, a.z / b); }
__device__ inline float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float length(const float3& a) { return sqrtf(dot(a, a)); }

// --- CONSTANTS ---
#define VOXEL_SIZE 1.0f
#define EPSILON 1e-5f

__device__ inline int getGridHash(int gridPos_x, int gridPos_y, int gridPos_z, int gridSize) {
    const int p1 = 73856093;
    const int p2 = 19349663;
    const int p3 = 83492791;
    int n = p1 * gridPos_x ^ p2 * gridPos_y ^ p3 * gridPos_z;
    return n & (gridSize - 1);
}

// ------------------------------------------------------------------
// 1. PREDICTION STEP
// x* = x + v * dt + f_ext * dt^2
// ------------------------------------------------------------------
__global__ void predictPositionsKernel(CudaVoxel* voxels, int count, float dt, float gravity) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;
    CudaVoxel v = voxels[idx];

    // Сохраняем предыдущую позицию для Velocity Update в конце шага
    v.oldX = v.x;
    v.oldY = v.y;
    v.oldZ = v.z;

    if (v.mass <= 0.0f) return; // Статика не двигается

    // Применяем внешние силы (Гравитация) к скорости
    // v = v + dt * f_ext
    v.vy += gravity * dt;

    // Предсказываем позицию
    // x* = x + v * dt
    v.x += v.vx * dt;
    v.y += v.vy * dt;
    v.z += v.vz * dt;

    voxels[idx] = v;
}

// ------------------------------------------------------------------
// 2. SOLVER STEP
// Используем метод Якоби: каждый поток считает свои коллизии
// и усредняет результат.
// ------------------------------------------------------------------
__global__ void solveCollisionsPBDKernel(
    CudaVoxel* voxels,
    int numVoxels,
    unsigned int* gridParticleIndex,
    unsigned int* cellStart,
    unsigned int* cellEnd,
    unsigned int gridSize,
    float cellSize,
    float dt
    ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numVoxels) return;

    CudaVoxel me = voxels[idx];
    if (me.mass <= 0.0f) return; // Статика не решается

    float3 pos = make_float3(me.x, me.y, me.z);
    float3 originalPos = make_float3(me.oldX, me.oldY, me.oldZ); // x_prev

    // Накопитель коррекций для усреднения (Averaging)
    float3 totalDelta = make_float3(0, 0, 0);
    int constraintCount = 0;

    int gridX = floorf(me.x / cellSize);
    int gridY = floorf(me.y / cellSize);
    int gridZ = floorf(me.z / cellSize);

    // Поиск соседей (Broad Phase)
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int hash = getGridHash(gridX + x, gridY + y, gridZ + z, gridSize);
                unsigned int start = cellStart[hash];
                unsigned int end = cellEnd[hash];

                for (unsigned int k = start; k < end; k++) {
                    unsigned int neighborIdx = gridParticleIndex[k];
                    if (neighborIdx == idx) continue;

                   CudaVoxel other = voxels[neighborIdx];

                    // --- Narrow Phase (AABB Intersection) ---
                    float3 otherPos = make_float3(other.x, other.y, other.z);
                    float3 diff = pos - otherPos;

                    float x_overlap = VOXEL_SIZE - fabsf(diff.x);
                    float y_overlap = VOXEL_SIZE - fabsf(diff.y);
                    float z_overlap = VOXEL_SIZE - fabsf(diff.z);

                    // Если есть пересечение по всем осям
                    if (x_overlap > EPSILON && y_overlap > EPSILON && z_overlap > EPSILON) {

                        // Находим нормаль коллизии (Minimum Translation Vector)
                        float3 normal = make_float3(0,0,0);
                        float depth = 0.0f;

                        if (x_overlap < y_overlap && x_overlap < z_overlap) {
                            normal = make_float3(diff.x > 0 ? 1 : -1, 0, 0);
                            depth = x_overlap;
                        } else if (y_overlap < z_overlap) {
                            normal = make_float3(0, diff.y > 0 ? 1 : -1, 0);
                            depth = y_overlap;
                        } else {
                            normal = make_float3(0, 0, diff.z > 0 ? 1 : -1);
                            depth = z_overlap;
                        }


                        // PBD Mass weighting
                        float w1 = 1.0f / me.mass;
                        float w2 = (other.mass > 0.0f) ? (1.0f / other.mass) : 0.0f;


                        float wSum = w1 + w2;
                        if (wSum < EPSILON) continue;

                        // --- Contact Constraint solving ---
                        // Коррекция позиции, чтобы разделить объекты
                        float3 deltaX = normal * (depth * (w1 / wSum));

                        // --- Friction ---
                        // Трение в PBD делается на уровне позиций.
                        // 1. Вычисляем смещение относительно предыдущего кадра
                        float3 pos_star_i = pos + deltaX;
                        float3 pos_star_j = otherPos; // Упрощение: считаем соседа зафиксированным (Jacobi style)

                        // 2. Вектор полного перемещения за шаг (относительная скорость * dt)
                        // Формула: (x*_i - x_old_i) - (x*_j - x_old_j)
                        float3 curr_disp = (pos_star_i - originalPos) - (pos_star_j - make_float3(other.oldX, other.oldY, other.oldZ));

                        // 3. Тангенциальная составляющая (проекция скорости на плоскость касания)
                        float disp_normal_projection = dot(curr_disp, normal);
                        float3 disp_tangent = curr_disp - normal * disp_normal_projection; // Убираем нормаль
                        float disp_len = length(disp_tangent);

                        if (disp_len > EPSILON) {
                            // Используем средний коэффициент трения
                            float mu = 0.5f * (me.friction + other.friction);

                            // В PBD сила нормальной реакции пропорциональна глубине проникновения (depth).
                            // Закон Кулона: F_friction <= mu * F_normal  =>  Delta_friction <= mu * depth
                            float max_friction_displacement = mu * depth;

                            float3 frictionDelta;

                            if (disp_len < max_friction_displacement) {
                                // --- Static Friction (Трение покоя) ---
                                frictionDelta = disp_tangent * -1.0f;
                            } else {
                                // --- Kinetic Friction (Трение скольжения) ---
                                frictionDelta = disp_tangent * -(max_friction_displacement / disp_len);
                            }

                            // 4. Применяем трение с учетом веса частицы (Inverse Mass)
                            // w1 = 1/m1, wSum = 1/m1 + 1/m2
                            deltaX = deltaX + frictionDelta * (w1 / wSum);
                        }

                        totalDelta = totalDelta + deltaX;
                        constraintCount++;
                    }
                }
            }
        }
    }

    // Применяем усредненную коррекцию (Averaging / Relaxation)
    if (constraintCount > 0) {
        // Деление на кол-во констрейнтов делает систему стабильнее (Jacobi Averaging)
        // Можно добавить SOR factor (omega) около 1.0 - 1.2 для скорости сходимости
        float omega = 1.2f;
        float factor = omega / (float)constraintCount;

        me.x += totalDelta.x * factor;
        me.y += totalDelta.y * factor;
        me.z += totalDelta.z * factor;


        if (constraintCount>1){
            float diff = length(make_float3(me.x - me.oldX, me.y - me.oldY, me.z - me.oldZ));
            if (diff < 0.001f) {
                me.x = me.oldX;
                me.y = me.oldY;
                me.z = me.oldZ;
            }
        }

        voxels[idx] = me;
    }
}

// ------------------------------------------------------------------
// 3. VELOCITY UPDATE
// v = (x* - x_prev) / dt
// ------------------------------------------------------------------
__global__ void updateVelocitiesKernel(CudaVoxel* voxels, int count, float dt, float damping) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    CudaVoxel v = voxels[idx];
    if (v.mass <= 0.0f) {
        v.vx = 0; v.vy = 0; v.vz = 0;
        voxels[idx] = v;
        return;
    }

    // PBD Velocity update
    // Скорость автоматически включает в себя отскок и внешние силы,
    // так как x изменился в процессе решения констрейнтов.

    float3 newVel;
    newVel.x = (v.x - v.oldX) / dt;
    newVel.y = (v.y - v.oldY) / dt;
    newVel.z = (v.z - v.oldZ) / dt;


    if (newVel.y * v.vy < -EPSILON){
        newVel.y *= 0.0; // предотвращает эффект катапульты у столбиков вокселей
    }

    // Apply Damping (Global)
    newVel = newVel * damping;

    v.vx = newVel.x;
    v.vy = newVel.y;
    v.vz = newVel.z;

    voxels[idx] = v;
}

// --- LAUNCHERS ---

extern "C" {
void launch_predictPositions(CudaVoxel* d_voxels, size_t numVoxels, float dt, float gravity) {
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;
    predictPositionsKernel<<<blocks, threads>>>(d_voxels, numVoxels, dt, gravity);
}

void launch_solveCollisionsPBD(
    CudaVoxel* d_voxels, unsigned int numVoxels,
    unsigned int* d_gridParticleIndex, unsigned int* d_cellStart, unsigned int* d_cellEnd,
    unsigned int gridSize, float cellSize, float dt)
{
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;
    nvtxRangePushA("Frame Render");

    solveCollisionsPBDKernel<<<blocks, threads>>>(
        d_voxels, numVoxels, d_gridParticleIndex, d_cellStart, d_cellEnd,
        gridSize, cellSize, dt
        );

    nvtxRangePop();
}

void launch_updateVelocities(CudaVoxel* d_voxels, size_t numVoxels, float dt, float damping) {
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;
    updateVelocitiesKernel<<<blocks, threads>>>(d_voxels, numVoxels, dt, damping);
}
}
