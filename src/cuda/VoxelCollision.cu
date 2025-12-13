#include "cuda/KernelLauncher.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


__device__ inline float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(const float3& a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ inline float3 operator/(const float3& a, float b) { return make_float3(a.x / b, a.y / b, a.z / b); }
__device__ inline float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float length(const float3& a) { return sqrtf(dot(a, a)); }

// --- CONSTANTS ---
#define VOXEL_SIZE 1.0f
#define EPSILON 1e-5f

__device__ __forceinline__ unsigned int expandBits(unsigned int n)
{
    n = (n ^ (n << 16)) & 0x030000ff;
    n = (n ^ (n <<  8)) & 0x0300f00f;
    n = (n ^ (n <<  4)) & 0x030c30c3;
    n = (n ^ (n <<  2)) & 0x09249249;
    return n;
}

__device__ __forceinline__ unsigned int morton3D(unsigned int x, unsigned int y, unsigned int z)
{
    return (expandBits(z) << 2) | (expandBits(y) << 1) | expandBits(x);
}

__device__ inline int getGridHash(int gridPos_x, int gridPos_y, int gridPos_z, int gridSize) {
    unsigned int x = static_cast<unsigned int>(gridPos_x) & 0xFF;
    unsigned int y = static_cast<unsigned int>(gridPos_y) & 0xFF;
    unsigned int z = static_cast<unsigned int>(gridPos_z) & 0xFF;
    unsigned int mCode = morton3D(x, y, z);
    return mCode & (gridSize - 1);
}

// ------------------------------------------------------------------
// 1. PREDICTION STEP
// x* = x + v * dt + f_ext * dt^2
// ------------------------------------------------------------------

__global__ void predictPositionsKernel(
    float* posX, float* posY, float* posZ,
    float* oldX, float* oldY, float* oldZ,
    float* velX, float* velY, float* velZ,
    const float* mass,
    int count,
    float dt,
    float gravity)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    float m = mass[idx];
    if (m <= 0.0f) return; // Статика

    // Сохраняем предыдущую позицию
    float px = posX[idx];
    float py = posY[idx];
    float pz = posZ[idx];

    oldX[idx] = px;
    oldY[idx] = py;
    oldZ[idx] = pz;

    float vx = velX[idx];
    float vy = velY[idx];
    float vz = velZ[idx];

    // Применяем внешние силы (Гравитация) к скорости
    // v = v + dt * f_ext
    vy += gravity * dt;

    // Предсказываем позицию
    // x* = x + v * dt
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;

    // Запись обратно в глобальную память
    posX[idx] = px;
    posY[idx] = py;
    posZ[idx] = pz;

    velY[idx] = vy;
}

// ------------------------------------------------------------------
// 2. SOLVER STEP
// Используем метод Якоби: каждый поток считает свои коллизии
// и усредняет результат.
// ------------------------------------------------------------------

__global__ void solveCollisionsPBDKernel(
    float* __restrict__ posX, float* __restrict__ posY, float* __restrict__ posZ,
    const float* __restrict__ oldX, const float* __restrict__ oldY, const float* __restrict__ oldZ,
    const float* __restrict__ mass, const float* __restrict__ friction, const int* clusterID,
    unsigned int numVoxels,
    const unsigned int* __restrict__ cellStart,
    const unsigned int* __restrict__ cellEnd,
    unsigned int gridSize,
    float cellSize,
    float dt)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numVoxels) return;

    float myMass = mass[idx];
    if (myMass <= 0.0f) return;

    float px = posX[idx];
    float py = posY[idx];
    float pz = posZ[idx];
    float3 pos = make_float3(px, py, pz);

    float ox = oldX[idx];
    float oy = oldY[idx];
    float oz = oldZ[idx];
    float3 originalPos = make_float3(ox, oy, oz);

    float myFriction = friction[idx];

    // Накопитель коррекций
    float3 totalDelta = make_float3(0, 0, 0);
    int constraintCount = 0;

    int gridX = floorf(px / cellSize);
    int gridY = floorf(py / cellSize);
    int gridZ = floorf(pz / cellSize);

    // Поиск соседей (Broad Phase)
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int hash = getGridHash(gridX + x, gridY + y, gridZ + z, gridSize);
                unsigned int start = __ldg(&cellStart[hash]);

                // Проверка на пустую ячейку
                if (start == 0xFFFFFFFF) continue;

                unsigned int end = __ldg(&cellEnd[hash]);

                for (unsigned int k = start; k < end; k++) {
                    unsigned int neighborIdx = k;
                    if (neighborIdx == idx) continue;
                    if (clusterID[idx] >= 0 && clusterID[idx] == clusterID[neighborIdx]) continue; // Мы можем не разрешать коллизию, если воксели принадлежать одному и тому же кластеру

                    float nx = __ldg(&posX[neighborIdx]);
                    float ny = __ldg(&posY[neighborIdx]);
                    float nz = __ldg(&posZ[neighborIdx]);

                    float otherMass = __ldg(&mass[neighborIdx]);

                    // --- Narrow Phase (AABB Intersection) ---
                    float3 otherPos = make_float3(nx, ny, nz);
                    float3 diff = pos - otherPos;

                    float x_overlap = VOXEL_SIZE - fabsf(diff.x);
                    float y_overlap = VOXEL_SIZE - fabsf(diff.y);
                    float z_overlap = VOXEL_SIZE - fabsf(diff.z);

                    // Если есть пересечение
                    if (x_overlap > EPSILON && y_overlap > EPSILON && z_overlap > EPSILON) {

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
                        float w1 = 1.0f / myMass;
                        float w2 = (otherMass > 0.0f) ? (1.0f / otherMass) : 0.0f;

                        float wSum = w1 + w2;
                        if (wSum < EPSILON) continue;

                        // --- Contact Constraint solving ---
                        float3 deltaX = normal * (depth * (w1 / wSum));

                        // --- Friction ---
                        float3 pos_star_i = pos + deltaX;

                        // Сосед (j)
                        float nj_oldX = oldX[neighborIdx];
                        float nj_oldY = oldY[neighborIdx];
                        float nj_oldZ = oldZ[neighborIdx];
                        float3 pos_star_j = otherPos;

                        // Вектор полного перемещения
                        float3 curr_disp = (pos_star_i - originalPos) - (pos_star_j - make_float3(nj_oldX, nj_oldY, nj_oldZ));

                        float disp_normal_projection = dot(curr_disp, normal);
                        float3 disp_tangent = curr_disp - normal * disp_normal_projection;
                        float disp_len = length(disp_tangent);

                        if (disp_len > EPSILON) {
                            float otherFriction = friction[neighborIdx];
                            float mu = 0.5f * (myFriction + otherFriction);

                            float max_friction_displacement = mu * depth;
                            float3 frictionDelta;

                            if (disp_len < max_friction_displacement) {
                                frictionDelta = disp_tangent * -1.0f;
                            } else {
                                frictionDelta = disp_tangent * -(max_friction_displacement / disp_len);
                            }
                            deltaX = deltaX + frictionDelta * (w1 / wSum);
                        }


                        totalDelta = totalDelta + deltaX;
                        constraintCount++;
                    }
                }
            }
        }
    }

    // Применяем усредненную коррекцию
    if (constraintCount > 0) {
        float omega = 1.2f;
        float factor = omega / (float)constraintCount;

        pos.x += totalDelta.x * factor;
        pos.y += totalDelta.y * factor;
        pos.z += totalDelta.z * factor;

        if (constraintCount > 1) { // засыпание вокселя при наличии 2 и более соседей для избежания желейности системы
            float diff = length(pos - originalPos);
            if (diff < 0.001f) {
                pos = originalPos;
            }
        }

        posX[idx] = pos.x;
        posY[idx] = pos.y;
        posZ[idx] = pos.z;
    }
}

// ------------------------------------------------------------------
// 3. VELOCITY UPDATE
// v = (x* - x_prev) / dt
// ------------------------------------------------------------------
__global__ void updateVelocitiesKernel(
    const float* posX, const float* posY, const float* posZ,
    const float* oldX, const float* oldY, const float* oldZ,
    float* velX, float* velY, float* velZ,
    const float* mass,
    int count,
    float dt,
    float damping)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= count) return;

    if (mass[idx] == 0.0f) {
        velX[idx] = 0;
        velY[idx] = 0;
        velZ[idx] = 0;
        return;
    }

    float px = posX[idx];
    float py = posY[idx];
    float pz = posZ[idx];

    float ox = oldX[idx];
    float oy = oldY[idx];
    float oz = oldZ[idx];

    float3 newVel;
    newVel.x = (px - ox) / dt;
    newVel.y = (py - oy) / dt;
    newVel.z = (pz - oz) / dt;

    float vy_prev = velY[idx];
    if (newVel.y * vy_prev < -EPSILON){ // костыль для избежания эффекта катапульты при падении столбиков вокселей на пол
        newVel.y *= 0.0f;
    }

    newVel = newVel * damping;

    velX[idx] = newVel.x;
    velY[idx] = newVel.y;
    velZ[idx] = newVel.z;
}


// --- LAUNCHERS ---
extern "C" {
void launch_predictPositions(
    float* posX, float* posY, float* posZ,
    float* oldX, float* oldY, float* oldZ,
    float* velX, float* velY, float* velZ,
    float* mass,
    size_t numVoxels,
    float dt,
    float gravity)
{
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;
    predictPositionsKernel<<<blocks, threads>>>(
        posX, posY, posZ,
        oldX, oldY, oldZ,
        velX, velY, velZ,
        mass,
        numVoxels, dt, gravity
        );
}

void launch_solveCollisionsPBD(
    float* posX, float* posY, float* posZ,
    float* oldX, float* oldY, float* oldZ,
    float* mass, float* friction, int* clusterID,
    unsigned int numVoxels,
    unsigned int* d_cellStart,
    unsigned int* d_cellEnd,
    unsigned int gridSize,
    float cellSize,
    float dt)
{
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;

    solveCollisionsPBDKernel<<<blocks, threads>>>(
        posX, posY, posZ,
        oldX, oldY, oldZ,
        mass, friction, clusterID,
        numVoxels,
        d_cellStart, d_cellEnd,
        gridSize, cellSize, dt
        );

}

void launch_updateVelocities(
    float* posX, float* posY, float* posZ,
    float* oldX, float* oldY, float* oldZ,
    float* velX, float* velY, float* velZ,
    float* mass,
    size_t numVoxels,
    float dt,
    float damping)
{
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;
    updateVelocitiesKernel<<<blocks, threads>>>(
        posX, posY, posZ,
        oldX, oldY, oldZ,
        velX, velY, velZ,
        mass,
        numVoxels, dt, damping
        );
}
}
