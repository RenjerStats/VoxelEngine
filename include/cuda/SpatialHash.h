#pragma once
#include <cuda_runtime.h>

class SpatialHash {
public:
    SpatialHash(unsigned int gridSize = 256*256*64, float cellSize = 1.05f)
        : m_numVoxels(0), m_gridSize(gridSize), m_cellSize(cellSize),
        d_gridParticleHash(nullptr), d_gridParticleIndex(nullptr),
        d_cellStart(nullptr), d_cellEnd(nullptr)
    {
        cudaMalloc((void**)&d_cellStart, m_gridSize * sizeof(unsigned int));
        cudaMalloc((void**)&d_cellEnd, m_gridSize * sizeof(unsigned int));
    }

    ~SpatialHash(){
        if (d_gridParticleHash) cudaFree(d_gridParticleHash);
        if (d_gridParticleIndex) cudaFree(d_gridParticleIndex);
        if (d_cellStart) cudaFree(d_cellStart);
        if (d_cellEnd) cudaFree(d_cellEnd);
    }
    void resize(unsigned int numVoxels){
        if (m_numVoxels == numVoxels) return;

        if (d_gridParticleHash) cudaFree(d_gridParticleHash);
        if (d_gridParticleIndex) cudaFree(d_gridParticleIndex);

        m_numVoxels = numVoxels;

        if (m_numVoxels > 0) {
            cudaMalloc((void**)&d_gridParticleHash, m_numVoxels * sizeof(unsigned int));
            cudaMalloc((void**)&d_gridParticleIndex, m_numVoxels * sizeof(unsigned int));
        }
    }

    unsigned int* getGridParticleHash() const { return d_gridParticleHash; }
    unsigned int* getGridParticleIndex() const { return d_gridParticleIndex; }
    unsigned int* getCellStart() const { return d_cellStart; }
    unsigned int* getCellEnd() const { return d_cellEnd; }

    unsigned int getGridSize() const { return m_gridSize; }
    float getCellSize() const { return m_cellSize; }

private:
    unsigned int m_numVoxels;
    unsigned int m_gridSize;
    float m_cellSize;

    unsigned int* d_gridParticleHash = nullptr;
    unsigned int* d_gridParticleIndex = nullptr;
    unsigned int* d_cellStart = nullptr;
    unsigned int* d_cellEnd = nullptr;
};
