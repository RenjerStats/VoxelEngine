#pragma once
#include <cuda_runtime.h>
#include <algorithm>

class ClusterManager {
public:
    ClusterManager() : m_numVoxels(0), m_activeClustersCount(0) {}

    ~ClusterManager() { freeResources(); }

    void initMemory(unsigned int numVoxels) {
        if (m_numVoxels == numVoxels) return;
        freeResources();
        m_numVoxels = numVoxels;
        m_activeClustersCount = 0;

        if (m_numVoxels == 0) return;

        // --- Основные данные ---
        cudaMalloc((void**)&d_voxelClusterID, m_numVoxels * sizeof(int));
        cudaMalloc((void**)&d_restOffsetX, m_numVoxels * sizeof(float));
        cudaMalloc((void**)&d_restOffsetY, m_numVoxels * sizeof(float));
        cudaMalloc((void**)&d_restOffsetZ, m_numVoxels * sizeof(float));

        // --- Sorted Буферы (для перестановки каждый кадр) ---
        cudaMalloc((void**)&d_sortedVoxelClusterID, m_numVoxels * sizeof(int));
        cudaMalloc((void**)&d_sortedRestOffsetX, m_numVoxels * sizeof(float));
        cudaMalloc((void**)&d_sortedRestOffsetY, m_numVoxels * sizeof(float));
        cudaMalloc((void**)&d_sortedRestOffsetZ, m_numVoxels * sizeof(float));

        // --- Данные кластеров (не сортируются, индексируются по ID) ---
        cudaMalloc((void**)&d_clusterCM, m_numVoxels * sizeof(double3));
        cudaMalloc((void**)&d_clusterRot, m_numVoxels * 9 * sizeof(float));
        cudaMalloc((void**)&d_clusterMatrixA, m_numVoxels * 9 * sizeof(double));
        cudaMalloc((void**)&d_clusterMass, m_numVoxels * sizeof(float));
    }

    void resizeMemory(unsigned int newNumVoxels) {

        unsigned int oldNumVoxels = m_numVoxels;

        int* old_voxelClusterID = d_voxelClusterID;
        float* old_restOffsetX = d_restOffsetX;
        float* old_restOffsetY = d_restOffsetY;
        float* old_restOffsetZ = d_restOffsetZ;

        int* old_sortedVoxelClusterID = d_sortedVoxelClusterID;
        float* old_sortedRestOffsetX = d_sortedRestOffsetX;
        float* old_sortedRestOffsetY = d_sortedRestOffsetY;
        float* old_sortedRestOffsetZ = d_sortedRestOffsetZ;

        double3* old_clusterCM = d_clusterCM;
        float* old_clusterRot = d_clusterRot;
        double* old_clusterMatrixA = d_clusterMatrixA;
        float* old_clusterMass = d_clusterMass;

        m_numVoxels = newNumVoxels;

        // Allocate new memory
        cudaMalloc((void**)&d_voxelClusterID, m_numVoxels * sizeof(int));
        cudaMalloc((void**)&d_restOffsetX, m_numVoxels * sizeof(float));
        cudaMalloc((void**)&d_restOffsetY, m_numVoxels * sizeof(float));
        cudaMalloc((void**)&d_restOffsetZ, m_numVoxels * sizeof(float));

        cudaMalloc((void**)&d_sortedVoxelClusterID, m_numVoxels * sizeof(int));
        cudaMalloc((void**)&d_sortedRestOffsetX, m_numVoxels * sizeof(float));
        cudaMalloc((void**)&d_sortedRestOffsetY, m_numVoxels * sizeof(float));
        cudaMalloc((void**)&d_sortedRestOffsetZ, m_numVoxels * sizeof(float));

        cudaMalloc((void**)&d_clusterCM, m_numVoxels * sizeof(double3));
        cudaMalloc((void**)&d_clusterRot, m_numVoxels * 9 * sizeof(float));
        cudaMalloc((void**)&d_clusterMatrixA, m_numVoxels * 9 * sizeof(double));
        cudaMalloc((void**)&d_clusterMass, m_numVoxels * sizeof(float));

        // Copy old data
        if (oldNumVoxels > 0) {
            cudaMemcpy(d_voxelClusterID, old_voxelClusterID,
                       oldNumVoxels * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_restOffsetX, old_restOffsetX,
                       oldNumVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_restOffsetY, old_restOffsetY,
                       oldNumVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_restOffsetZ, old_restOffsetZ,
                       oldNumVoxels * sizeof(float), cudaMemcpyDeviceToDevice);

            cudaMemcpy(d_sortedVoxelClusterID, old_sortedVoxelClusterID,
                       oldNumVoxels * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_sortedRestOffsetX, old_sortedRestOffsetX,
                       oldNumVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_sortedRestOffsetY, old_sortedRestOffsetY,
                       oldNumVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_sortedRestOffsetZ, old_sortedRestOffsetZ,
                       oldNumVoxels * sizeof(float), cudaMemcpyDeviceToDevice);

            cudaMemcpy(d_clusterCM, old_clusterCM,
                       oldNumVoxels * sizeof(double3), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_clusterRot, old_clusterRot,
                       oldNumVoxels * 9 * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_clusterMatrixA, old_clusterMatrixA,
                       oldNumVoxels * 9 * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_clusterMass, old_clusterMass,
                       oldNumVoxels * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        // Free old memory
        cudaFree(old_voxelClusterID);
        cudaFree(old_restOffsetX);
        cudaFree(old_restOffsetY);
        cudaFree(old_restOffsetZ);
        cudaFree(old_sortedVoxelClusterID);
        cudaFree(old_sortedRestOffsetX);
        cudaFree(old_sortedRestOffsetY);
        cudaFree(old_sortedRestOffsetZ);
        cudaFree(old_clusterCM);
        cudaFree(old_clusterRot);
        cudaFree(old_clusterMatrixA);
        cudaFree(old_clusterMass);
    }

    void freeResources() {
        if (d_voxelClusterID) cudaFree(d_voxelClusterID);
        if (d_restOffsetX) cudaFree(d_restOffsetX);
        if (d_restOffsetY) cudaFree(d_restOffsetY);
        if (d_restOffsetZ) cudaFree(d_restOffsetZ);

        if (d_sortedVoxelClusterID) cudaFree(d_sortedVoxelClusterID);
        if (d_sortedRestOffsetX) cudaFree(d_sortedRestOffsetX);
        if (d_sortedRestOffsetY) cudaFree(d_sortedRestOffsetY);
        if (d_sortedRestOffsetZ) cudaFree(d_sortedRestOffsetZ);

        if (d_clusterCM) cudaFree(d_clusterCM);
        if (d_clusterRot) cudaFree(d_clusterRot);
        if (d_clusterMatrixA) cudaFree(d_clusterMatrixA);
        if (d_clusterMass) cudaFree(d_clusterMass);

        d_voxelClusterID = nullptr; d_sortedVoxelClusterID = nullptr;
        d_restOffsetX = nullptr; d_sortedRestOffsetX = nullptr;
        d_restOffsetY = nullptr; d_sortedRestOffsetY = nullptr;
        d_restOffsetZ = nullptr; d_sortedRestOffsetZ = nullptr;

        d_clusterCM = nullptr; d_clusterRot = nullptr;
        d_clusterMatrixA = nullptr; d_clusterMass = nullptr;
    }

    // Swap pointers method (для PhysicsManager)
    void swapSortedBuffers() {
        std::swap(d_voxelClusterID, d_sortedVoxelClusterID);
        std::swap(d_restOffsetX, d_sortedRestOffsetX);
        std::swap(d_restOffsetY, d_sortedRestOffsetY);
        std::swap(d_restOffsetZ, d_sortedRestOffsetZ);
    }

    // Getters
    unsigned int getNumVoxels() const { return m_numVoxels; }

    int* getVoxelClusterID() const { return d_voxelClusterID; }
    int* getSortedVoxelClusterID() const { return d_sortedVoxelClusterID; }

    float* getRestOffsetX() const { return d_restOffsetX; }
    float* getRestOffsetY() const { return d_restOffsetY; }
    float* getRestOffsetZ() const { return d_restOffsetZ; }

    float* getSortedRestOffsetX() const { return d_sortedRestOffsetX; }
    float* getSortedRestOffsetY() const { return d_sortedRestOffsetY; }
    float* getSortedRestOffsetZ() const { return d_sortedRestOffsetZ; }

    double3* getClusterCM() const { return d_clusterCM; }
    float* getClusterRot() const { return d_clusterRot; }
    double* getClusterMatrixA() const { return d_clusterMatrixA; }
    float* getClusterMass() const { return d_clusterMass; }


private:
    unsigned int m_numVoxels;
    unsigned int m_activeClustersCount;

    int* d_voxelClusterID = nullptr;
    int* d_sortedVoxelClusterID = nullptr;


    float* d_restOffsetX = nullptr;
    float* d_sortedRestOffsetX = nullptr;
    float* d_restOffsetY = nullptr;
    float* d_sortedRestOffsetY = nullptr;
    float* d_restOffsetZ = nullptr;
    float* d_sortedRestOffsetZ = nullptr;

    double3* d_clusterCM = nullptr;
    float* d_clusterRot = nullptr;
    double* d_clusterMatrixA = nullptr;
    float* d_clusterMass = nullptr;
};
