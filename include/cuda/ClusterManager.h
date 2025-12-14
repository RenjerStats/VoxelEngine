#pragma once
#include <cuda_runtime.h>
#include <algorithm>

class ClusterManager {
public:
    ClusterManager() : m_numVoxels(0), m_activeClustersCount(0) {}

    ~ClusterManager() { freeResources(); }

    void resize(unsigned int numVoxels) {
        if (m_numVoxels == numVoxels) return;
        freeResources();
        m_numVoxels = numVoxels;
        m_activeClustersCount = 0;

        if (m_numVoxels == 0) return;

        // --- Основные данные ---
        cudaMalloc((void**)&d_voxelClusterID, m_numVoxels * sizeof(int));
        cudaMalloc((void**)&d_voxelConnections, m_numVoxels * sizeof(unsigned char));
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
        cudaMalloc((void**)&d_clusterMatrixA, m_numVoxels * 9 * sizeof(float));
        cudaMalloc((void**)&d_clusterMass, m_numVoxels * sizeof(float));
        cudaMalloc((void**)&d_globalClusterCount, sizeof(unsigned int));
        cudaMalloc((void**)&d_clusterIsDirty, m_numVoxels * sizeof(unsigned int));
    }

    void freeResources() {
        if (d_voxelClusterID) cudaFree(d_voxelClusterID);
        if (d_voxelConnections) cudaFree(d_voxelConnections);
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
        if (d_globalClusterCount) cudaFree(d_globalClusterCount);
        if (d_clusterIsDirty) cudaFree(d_clusterIsDirty);

        d_voxelClusterID = nullptr; d_sortedVoxelClusterID = nullptr;
        d_voxelConnections = nullptr;
        d_restOffsetX = nullptr; d_sortedRestOffsetX = nullptr;
        d_restOffsetY = nullptr; d_sortedRestOffsetY = nullptr;
        d_restOffsetZ = nullptr; d_sortedRestOffsetZ = nullptr;

        d_clusterCM = nullptr; d_clusterRot = nullptr;
        d_clusterMatrixA = nullptr; d_clusterMass = nullptr;
        d_globalClusterCount = nullptr; d_clusterIsDirty = nullptr;
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
    float* getClusterMatrixA() const { return d_clusterMatrixA; }
    float* getClusterMass() const { return d_clusterMass; }

    // Вспомогательный
    unsigned int* getGlobalClusterCountPtr() const { return d_globalClusterCount; }

private:
    unsigned int m_numVoxels;
    unsigned int m_activeClustersCount;

    int* d_voxelClusterID = nullptr;
    int* d_sortedVoxelClusterID = nullptr;

    unsigned char* d_voxelConnections = nullptr;

    float* d_restOffsetX = nullptr;
    float* d_sortedRestOffsetX = nullptr;
    float* d_restOffsetY = nullptr;
    float* d_sortedRestOffsetY = nullptr;
    float* d_restOffsetZ = nullptr;
    float* d_sortedRestOffsetZ = nullptr;

    double3* d_clusterCM = nullptr;
    float* d_clusterRot = nullptr;
    float* d_clusterMatrixA = nullptr;
    float* d_clusterMass = nullptr;
    unsigned int* d_clusterIsDirty = nullptr;
    unsigned int* d_globalClusterCount = nullptr;
};
