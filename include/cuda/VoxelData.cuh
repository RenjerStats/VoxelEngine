#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace CudaPhysics {

// SoA структура для хранения вокселей на GPU
struct VoxelsSoA {
    float* posX;          // X-координаты
    float* posY;          // Y-координаты
    float* posZ;          // Z-координаты
    float* velX;          // Скорости по X
    float* velY;          // Скорости по Y
    float* velZ;          // Скорости по Z
    uint32_t* colorIndex; // Индексы цветов (из палитры)
    float* mass;          // Массы вокселей
    uint32_t count;       // Количество вокселей

    // Конструктор по умолчанию
    VoxelsSoA()
        : posX(nullptr), posY(nullptr), posZ(nullptr)
        , velX(nullptr), velY(nullptr), velZ(nullptr)
        , colorIndex(nullptr), mass(nullptr), count(0) {}
};

// Палитра цветов на GPU (256 цветов RGBA)
struct VoxelPalette {
    uint32_t colors[256]; // Упакованные RGBA цвета (uint32_t)
};

// Менеджер для управления данными вокселей на GPU
class VoxelDataManager {
public:
    VoxelDataManager();
    ~VoxelDataManager();

    // Выделить память на GPU
    bool allocate(uint32_t voxelCount);

    // Загрузить данные вокселей из хоста
    bool uploadVoxels(const float* posX, const float* posY, const float* posZ,
                      const uint32_t* colorIndex, uint32_t count);

    // Загрузить палитру цветов
    bool uploadPalette(const uint32_t* colors);

    // Инициализировать физические параметры (скорости, массы)
    void initializePhysics(float defaultMass = 1.0f);

    // Освободить память
    void release();

    // Получить указатели на данные
    const VoxelsSoA& getDeviceData() const { return m_deviceData; }
    const VoxelPalette& getPalette() const { return m_palette; }
    uint32_t getVoxelCount() const { return m_deviceData.count; }

    // Проверка валидности
    bool isValid() const { return m_deviceData.count > 0 && m_deviceData.posX != nullptr; }

private:
    VoxelsSoA m_deviceData;
    VoxelPalette m_palette;
    bool m_allocated;

    // Запрет копирования
    VoxelDataManager(const VoxelDataManager&) = delete;
    VoxelDataManager& operator=(const VoxelDataManager&) = delete;
};

}
