#pragma once
#include <cuda_runtime.h>

class SpatialHash {
public:
    SpatialHash(unsigned int gridSize = 256*256, float cellSize = 1.05f);
    ~SpatialHash();

    // Пересоздание буферов, если изменилось количество частиц
    void resize(unsigned int numVoxels);

    // Геттеры для передачи данных в кернелы (PhysicsManager'у)
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

    // GPU буферы
    unsigned int* d_gridParticleHash = nullptr;
    unsigned int* d_gridParticleIndex = nullptr;
    unsigned int* d_cellStart = nullptr;
    unsigned int* d_cellEnd = nullptr;
};
