#include "cuda/KernelLauncher.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

static constexpr float kMaxDeltaSM   = 1.9f;

// --- MATH HELPERS ---

__device__ inline float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(const float3& a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ inline float3 to_float3(const double3& d) {return make_float3((float)d.x, (float)d.y, (float)d.z);}

__device__ inline double3 operator+(const double3& a, const double3& b) { return make_double3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline double3 operator-(const double3& a, const double3& b) { return make_double3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline double3 operator*(const double3& a, double b) { return make_double3(a.x * b, a.y * b, a.z * b); }

__device__ inline void atomicAddDouble3(double3* addressBase, float3 val) {
    atomicAdd(&addressBase->x, (double)val.x);
    atomicAdd(&addressBase->y, (double)val.y);
    atomicAdd(&addressBase->z, (double)val.z);
}

__device__ float3 make3(float x, float y, float z) { return make_float3(x,y,z); }

__device__ float  dot3(const float3& a, const float3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

__device__ float3 cross3(const float3& a, const float3& b) {
    return make3(a.y*b.z - a.z*b.y,
                 a.z*b.x - a.x*b.z,
                 a.x*b.y - a.y*b.x);
}

__device__ float len3(const float3& v) { return sqrtf(dot3(v,v)); }

__device__ float3 normalize3_safe(const float3& v, float eps, bool& ok) {
    float l = len3(v);
    if (!(l > eps) || !isfinite(l)) { ok = false; return make3(0,0,0); }
    ok = true;
    float inv = 1.0f / l;
    return make3(v.x*inv, v.y*inv, v.z*inv);
}

__device__ inline float3 clampDeltaLen(const float3& d, float maxLen) {
    float l2 = d.x*d.x + d.y*d.y + d.z*d.z;
    if (!(l2 > maxLen*maxLen) || !isfinite(l2)) return d;
    float invL = rsqrtf(l2);
    float s = maxLen * invL;
    return make_float3(d.x*s, d.y*s, d.z*s);
}

// --- KERNELS ---

// 1. ИНИЦИАЛИЗАЦИЯ (Запускается 1 раз при старте)
// Рассчитывает исходный центр масс и relative offsets (r_i)
__global__ void initClusterStateKernel(
    const float* posX, const float* posY, const float* posZ,
    const float* mass,
    const int* clusterID,
    float* restOffsetX, float* restOffsetY, float* restOffsetZ,
    double3* clusterCM, float* clusterMass,
    unsigned int numVoxels
    ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numVoxels) return;

    int cID = clusterID[idx];
    if (cID < 0) return;

    float3 p = make_float3(posX[idx], posY[idx], posZ[idx]);

    atomicAddDouble3(&clusterCM[cID], p * mass[idx]);
    atomicAdd(&clusterMass[cID], mass[idx]);
}

// 1.b ИНИЦИАЛИЗАЦИЯ Часть 2 (Расчет Rest Offsets)
__global__ void computeRestOffsetsKernel(
    const float* posX, const float* posY, const float* posZ,
    const int* clusterID,
    float* restOffsetX, float* restOffsetY, float* restOffsetZ,
    const double3* clusterCM, const float* clusterMass,
    unsigned int numVoxels
    ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numVoxels) return;

    int cID = clusterID[idx];
    if (cID < 0) return;

    float mass = clusterMass[cID];
    if (mass <= 0) return;

    float3 cm = to_float3(clusterCM[cID] * (1.0 / mass));
    float3 p = make_float3(posX[idx], posY[idx], posZ[idx]);

    // r_i = x_i - x_cm0
    float3 r = p - cm;

    restOffsetX[idx] = r.x;
    restOffsetY[idx] = r.y;
    restOffsetZ[idx] = r.z;
}

// Шаг 1: Очистка аккумуляторов кластера (CM, Mass, Matrix A)
__global__ void clearClusterAccumulatorsKernel(
    double3* clusterCM, float* clusterMass, double* clusterMatrixA,
    int numClusters
    ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numClusters) return;

    clusterCM[idx] = make_double3(0,0,0);
    clusterMass[idx] = 0.0f;

    int matIdx = idx * 9;
    for(int i=0; i<9; ++i) clusterMatrixA[matIdx + i] = 0.0;
}

// Шаг 2: Расчет ТЕКУЩЕГО центра масс (Atomic Accumulation)
__global__ void calcClusterCMKernel(
    const float* posX, const float* posY, const float* posZ,
    const float* mass, const int* clusterID,
    double3* clusterCM, float* clusterMass,
    unsigned int numVoxels
    ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numVoxels) return;

    int cID = clusterID[idx];
    if (cID < 0) return;

    float m = mass[idx];
    float3 p = make_float3(posX[idx], posY[idx], posZ[idx]);

    atomicAddDouble3(&clusterCM[cID], p * m);
    atomicAdd(&clusterMass[cID], m);
}

// Шаг 3: Расчет Ковариационной Матрицы A (Atomic Accumulation)
__global__ void calcClusterCovarianceKernel(
    const float* posX, const float* posY, const float* posZ,
    const float* mass, const int* clusterID,
    const float* restOffsetX, const float* restOffsetY, const float* restOffsetZ,
    const double3* clusterCM, const float* clusterMass,
    double* clusterMatrixA,
    unsigned int numVoxels
    ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numVoxels) return;

    int cID = clusterID[idx];
    if (cID < 0) return;

    float totalM = clusterMass[cID];


    float3 cm = to_float3(clusterCM[cID] * (1.0 / totalM));
    float3 p = make_float3(posX[idx], posY[idx], posZ[idx]);
    float3 q = p - cm;


    float3 r = make_float3(restOffsetX[idx], restOffsetY[idx], restOffsetZ[idx]);

    float m = mass[idx];

    int matIdx = cID * 9;


    atomicAdd(&clusterMatrixA[matIdx + 0], double(m * q.x * r.x));
    atomicAdd(&clusterMatrixA[matIdx + 1], double(m * q.x * r.y));
    atomicAdd(&clusterMatrixA[matIdx + 2], double(m * q.x * r.z));

    atomicAdd(&clusterMatrixA[matIdx + 3], double(m * q.y * r.x));
    atomicAdd(&clusterMatrixA[matIdx + 4], double(m * q.y * r.y));
    atomicAdd(&clusterMatrixA[matIdx + 5], double(m * q.y * r.z));

    atomicAdd(&clusterMatrixA[matIdx + 6], double(m * q.z * r.x));
    atomicAdd(&clusterMatrixA[matIdx + 7], double(m * q.z * r.y));
    atomicAdd(&clusterMatrixA[matIdx + 8], double(m * q.z * r.z));
}

// Шаг 4: Вычисление Вращения (Polar Decomposition)
__device__ __forceinline__ bool inv3x3_rowmajor(const float M[9], float Minv[9], float epsDet) {
    // adjugate / det
    float a00 = M[0], a01 = M[1], a02 = M[2];
    float a10 = M[3], a11 = M[4], a12 = M[5];
    float a20 = M[6], a21 = M[7], a22 = M[8];

    float c00 =  (a11*a22 - a12*a21);
    float c01 = -(a10*a22 - a12*a20);
    float c02 =  (a10*a21 - a11*a20);

    float det = a00*c00 + a01*c01 + a02*c02;
    if (!isfinite(det) || fabsf(det) <= epsDet) return false;

    float invDet = 1.0f / det;

    Minv[0] = c00 * invDet;
    Minv[1] = (-(a01*a22 - a02*a21)) * invDet;
    Minv[2] = ( (a01*a12 - a02*a11)) * invDet;

    Minv[3] = c01 * invDet;
    Minv[4] = ( (a00*a22 - a02*a20)) * invDet;
    Minv[5] = (-(a00*a12 - a02*a10)) * invDet;

    Minv[6] = c02 * invDet;
    Minv[7] = (-(a00*a21 - a01*a20)) * invDet;
    Minv[8] = ( (a00*a11 - a01*a10)) * invDet;

    return true;
}

__device__ __forceinline__ void writeIdentity(float* out, int matIdx) {
    out[matIdx+0] = 1; out[matIdx+1] = 0; out[matIdx+2] = 0;
    out[matIdx+3] = 0; out[matIdx+4] = 1; out[matIdx+5] = 0;
    out[matIdx+6] = 0; out[matIdx+7] = 0; out[matIdx+8] = 1;
}

__global__ void computeClusterRotationKernel(
    const double* clusterMatrixA,
    float* clusterRot,
    const float* clusterMass,
    unsigned int numClusters,
    float rotStiffness
    ) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numClusters) return;

    const int matIdx = static_cast<int>(idx) * 9;


    float R_old[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        R_old[i] = clusterRot[matIdx + i];
    }

    float traceOld = R_old[0]*R_old[0] + R_old[4]*R_old[4] + R_old[8]*R_old[8];
    if (traceOld < 0.1f) {  // срабатывает на первом кадре, когда R_old не инициализирована
        R_old[0]=1; R_old[1]=0; R_old[2]=0;
        R_old[3]=0; R_old[4]=1; R_old[5]=0;
        R_old[6]=0; R_old[7]=0; R_old[8]=1;
    }

    // 1) Load A
    float totalM = clusterMass[idx];
    float scale = (totalM > 0.000001f) ? (1.0f / totalM) : 1.0f;

    float A[9];
    float frob2 = 0.0f;

    for (int i = 0; i < 9; ++i) {
        double val = clusterMatrixA[matIdx + i];
        float v = (float)(val * (double)scale);

        if (!isfinite(v)) { // если получилась "нехорошая" матрица, оставляем предыдущую
            for(int k=0; k<9; k++) clusterRot[matIdx+k] = R_old[k];
            return;
        }
        A[i] = v;
        frob2 += v*v;
    }

    const float epsA    = 1e-9f;
    const float epsDet  = 1e-9f;
    const float epsNorm = 1e-8f;

    if (!(frob2 > epsA)) {
        for(int k=0; k<9; k++) clusterRot[matIdx+k] = R_old[k];
        return;
    }

    float invFrob = rsqrtf(frob2);
    float R[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) R[i] = A[i] * invFrob;

    for (int iter = 0; iter < 10; ++iter) {
        float Rinv[9];
        if (!inv3x3_rowmajor(R, Rinv, epsDet)) {
            for(int k=0; k<9; k++) clusterRot[matIdx+k] = R_old[k];
            return;
        }

        float Rtinv[9] = {
            Rinv[0], Rinv[3], Rinv[6],
            Rinv[1], Rinv[4], Rinv[7],
            Rinv[2], Rinv[5], Rinv[8]
        };

        for (int i = 0; i < 9; ++i) {
            float v = 0.5f * (R[i] + Rtinv[i]);
            if (!isfinite(v))
            {
                for(int k=0; k<9; k++) clusterRot[matIdx+k] = R_old[k];
                return;
            }
            R[i] = v;
        }
    }

    float R_final[9];

    float clumpMass = totalM;
    float refMass = numClusters/2;
    if (clumpMass < 3)
        clumpMass = 3;
    if (clumpMass > refMass)
        clumpMass = refMass;

    rotStiffness = rotStiffness / (1 + clumpMass / refMass);

    for(int i=0; i<9; ++i) {
        R_final[i] = R_old[i] + rotStiffness * (R[i] - R_old[i]);
    }

    bool ok0, ok1, ok2;
    float3 r0 = make3(R_final[0], R_final[1], R_final[2]);
    float3 r1 = make3(R_final[3], R_final[4], R_final[5]);

    r0 = normalize3_safe(r0, epsNorm, ok0);
    if (!ok0)
    {
        for(int k=0; k<9; k++) clusterRot[matIdx+k] = R_old[k];
        return;
    }

    r1 = r1 - r0 * dot3(r1, r0);
    r1 = normalize3_safe(r1, epsNorm, ok1);
    if (!ok1)
    {
        for(int k=0; k<9; k++) clusterRot[matIdx+k] = R_old[k];
        return;
    }

    float3 r2 = cross3(r0, r1);

    r2 = normalize3_safe(r2, epsNorm, ok2);
    if (!ok2)
    {
        for(int k=0; k<9; k++) clusterRot[matIdx+k] = R_old[k];
        return;
    }

    clusterRot[matIdx+0] = r0.x; clusterRot[matIdx+1] = r0.y; clusterRot[matIdx+2] = r0.z;
    clusterRot[matIdx+3] = r1.x; clusterRot[matIdx+4] = r1.y; clusterRot[matIdx+5] = r1.z;
    clusterRot[matIdx+6] = r2.x; clusterRot[matIdx+7] = r2.y; clusterRot[matIdx+8] = r2.z;
}

// Шаг 5: Применение Shape Matching (PBD Solve)
__global__ void applyShapeMatchingKernel(
    float* posX, float* posY, float* posZ,
    int* clusterID,
    const float* restOffsetX, const float* restOffsetY, const float* restOffsetZ,
    const double3* clusterCM, const float* clusterMass, const float* clusterRot,
    unsigned int numVoxels, float stiffness, float breakLimit,
    int* d_clusterIsBraked
    ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numVoxels) return;

    int cID = clusterID[idx];
    if (cID < 0) return;

    float totalM = clusterMass[cID];
    if (totalM <= 0.0001f) return;

    float3 cm = to_float3(clusterCM[cID] * (1.0 / totalM));
    int matIdx = cID * 9;

    float R[9];
    for(int i=0; i<9; i++) R[i] = clusterRot[matIdx + i];

    float3 r = make_float3(restOffsetX[idx], restOffsetY[idx], restOffsetZ[idx]);

    float3 rotatedR;
    rotatedR.x = R[0]*r.x + R[1]*r.y + R[2]*r.z;
    rotatedR.y = R[3]*r.x + R[4]*r.y + R[5]*r.z;
    rotatedR.z = R[6]*r.x + R[7]*r.y + R[8]*r.z;

    float3 goal = cm + rotatedR;
    float3 curr = make_float3(posX[idx], posY[idx], posZ[idx]);

    float3 delta = goal - curr;

    float len = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;


    if (len > breakLimit) {
        atomicExch(&d_clusterIsBraked[cID], 1);
        return;
    }


    delta = clampDeltaLen(delta, kMaxDeltaSM);
    float3 nextPos = curr + delta * stiffness;

    posX[idx] = nextPos.x;
    posY[idx] = nextPos.y;
    posZ[idx] = nextPos.z;
}

__global__ void detachBrakedClustersKernel(int* clusterID, const int* braked, unsigned int numVoxels) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVoxels) return;
    int c = clusterID[i];
    if (c < 0) return;
    if (braked[c]){
        clusterID[i] = -1;
    }
}

// --- LAUNCHERS ---

extern "C" {

void launch_initClusterState(
    float* posX, float* posY, float* posZ,
    const float* mass,
    int* clusterID,
    float* restOffsetX, float* restOffsetY, float* restOffsetZ,
    double3* clusterCM, float* clusterMass,
    unsigned int numVoxels
    ) {
    int threads = 256;
    int blocks = (numVoxels + threads - 1) / threads;

    cudaMemset(clusterCM, 0, numVoxels * sizeof(double3));
    cudaMemset(clusterMass, 0, numVoxels * sizeof(float));

    initClusterStateKernel<<<blocks, threads>>>(
        posX, posY, posZ,
        mass,
        clusterID,
        restOffsetX, restOffsetY, restOffsetZ,
        clusterCM, clusterMass, numVoxels
        );
    cudaDeviceSynchronize();

    computeRestOffsetsKernel<<<blocks, threads>>>(
        posX, posY, posZ, clusterID,
        restOffsetX, restOffsetY, restOffsetZ,
        clusterCM, clusterMass, numVoxels
        );
}

void launch_shapeMatchingStep(
    float* posX, float* posY, float* posZ,
    const float* mass,
    int* clusterID,
    float* restOffsetX, float* restOffsetY, float* restOffsetZ,
    double3* clusterCM, float* clusterMass, double* clusterMatrixA, float* clusterRot,
    unsigned int numVoxels,
    unsigned int maxClusters,
    float stiffness, float rotStiffness, float breakLimit,
    int* d_clusterIsBraked
    ) {
    int threads = 256;
    int blocksVoxels = (numVoxels + threads - 1) / threads;
    int blocksClusters = (maxClusters + threads - 1) / threads;

    clearClusterAccumulatorsKernel<<<blocksClusters, threads>>>(
        clusterCM, clusterMass, clusterMatrixA, maxClusters
        );

    calcClusterCMKernel<<<blocksVoxels, threads>>>(
        posX, posY, posZ, mass, clusterID,
        clusterCM, clusterMass, numVoxels
        );

    calcClusterCovarianceKernel<<<blocksVoxels, threads>>>(
        posX, posY, posZ, mass, clusterID,
        restOffsetX, restOffsetY, restOffsetZ,
        clusterCM, clusterMass,
        clusterMatrixA, numVoxels
        );

    computeClusterRotationKernel<<<blocksClusters, threads>>>(
        clusterMatrixA, clusterRot, clusterMass, maxClusters, rotStiffness
        );

    applyShapeMatchingKernel<<<blocksVoxels, threads>>>(
        posX, posY, posZ, clusterID,
        restOffsetX, restOffsetY, restOffsetZ,
        clusterCM, clusterMass, clusterRot,
        numVoxels, stiffness,  breakLimit,
        d_clusterIsBraked
        );

    detachBrakedClustersKernel<<<blocksVoxels, threads>>>(
        clusterID, d_clusterIsBraked, numVoxels
        );
}
}



__global__ void initSingleClusterStateKernel(
    const float* posX, const float* posY, const float* posZ,
    const float* mass,
    const int* clusterID,
    double3* clusterCM, float* clusterMass,
    int targetClusterID,
    unsigned int begin, unsigned int end
    ) {
    unsigned int i = begin + (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= end) return;

    int cID = clusterID[i];
    if (cID != targetClusterID) return;
    float m = mass[i];
    if (m <= 0.0f) return;

    float3 p = make_float3(posX[i], posY[i], posZ[i]);
    atomicAddDouble3(&clusterCM[cID], p * m);
    atomicAdd(&clusterMass[cID], m);
}

__global__ void computeSingleClusterRestOffsetsKernel(
    const float* posX, const float* posY, const float* posZ,
    const int* clusterID,
    float* restOffsetX, float* restOffsetY, float* restOffsetZ,
    const double3* clusterCM, const float* clusterMass,
    int targetClusterID,
    unsigned int begin, unsigned int end
    ) {
    unsigned int i = begin + (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= end) return;

    int cID = clusterID[i];
    if (cID != targetClusterID) return;

    float totalM = clusterMass[cID];
    if (totalM <= 0.0f) return;

    float3 cm = to_float3(clusterCM[cID] * (1.0 / totalM));
    float3 p  = make_float3(posX[i], posY[i], posZ[i]);

    float3 r = p - cm;
    restOffsetX[i] = r.x;
    restOffsetY[i] = r.y;
    restOffsetZ[i] = r.z;
}


extern "C" {

void launch_initSingleClusterState(
    const float* posX, const float* posY, const float* posZ,
    const float* mass,
    const int* clusterID,
    float* restOffsetX, float* restOffsetY, float* restOffsetZ,
    double3* clusterCM, float* clusterMass,
    int targetClusterID,
    unsigned int begin, unsigned int count
    ) {
    if (count == 0) return;

    unsigned int end = begin + count;

    cudaMemset(clusterCM   + targetClusterID, 0, sizeof(double3));
    cudaMemset(clusterMass + targetClusterID, 0, sizeof(float));

    int threads = 256;
    int blocks  = (int)((count + threads - 1) / threads);

    initSingleClusterStateKernel<<<blocks, threads>>>(
        posX, posY, posZ, mass, clusterID,
        clusterCM, clusterMass,
        targetClusterID,
        begin, end
        );

    cudaDeviceSynchronize();

    computeSingleClusterRestOffsetsKernel<<<blocks, threads>>>(
        posX, posY, posZ, clusterID,
        restOffsetX, restOffsetY, restOffsetZ,
        clusterCM, clusterMass,
        targetClusterID,
        begin, end
        );
}

}
