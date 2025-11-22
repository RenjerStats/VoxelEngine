#pragma once

#include <QVector3D>
#include <QColor>
#include <QString>
#include <functional>

#include "ogt_vox.h"

namespace VoxIO {

struct Voxel {
    uint32_t x, y, z;
    uint8_t colorIndex;
    QColor color;

    QVector3D position() const {
        return QVector3D(x, y, z);
    }
};

class VoxModel {
public:
    VoxModel(const ogt_vox_model* model, const ogt_vox_palette& palette);

    uint32_t sizeX() const { return m_model->size_x; }
    uint32_t sizeY() const { return m_model->size_y; }
    uint32_t sizeZ() const { return m_model->size_z; }

    // Получить цвет воксела по индексу
    uint8_t getVoxelIndex(uint32_t x, uint32_t y, uint32_t z) const;
    QColor getVoxelColor(uint32_t x, uint32_t y, uint32_t z) const;

    // Проверка: вокселъ заполнен?
    bool isVoxelSolid(uint32_t x, uint32_t y, uint32_t z) const;

    // Обход ВСЕХ вокселей (включая пустые)
    void forEachVoxel(std::function<void(uint32_t x, uint32_t y, uint32_t z, uint8_t colorIndex)> callback) const;

    // Обход ТОЛЬКО ЗАПОЛНЕННЫХ вокселей (рекомендуется!)
    void forEachSolidVoxel(std::function<void(const Voxel& voxel)> callback) const;

    // Обход по слоям (Z) - полезно для послойного рендеринга
    void forEachLayer(std::function<void(uint32_t z, const std::vector<Voxel>& voxels)> callback) const;

    // Получить все заполненные вокселы сразу (для дальнейшей обработки)
    std::vector<Voxel> getSolidVoxels() const;

    // Получить палитру
    const ogt_vox_palette& palette() const { return m_palette; }

private:
    const ogt_vox_model* m_model;
    ogt_vox_palette m_palette;
};

// Обертка для сцены
class VoxScene {
public:
    VoxScene(const ogt_vox_scene* scene);
    ~VoxScene();

    // Запретить копирование
    VoxScene(const VoxScene&) = delete;
    VoxScene& operator=(const VoxScene&) = delete;

    // Получить количество моделей
    uint32_t modelCount() const { return m_scene->num_models; }

    // Получить модель по индексу
    VoxModel getModel(uint32_t index) const;

    // Получить количество инстансов (размещений моделей)
    uint32_t instanceCount() const { return m_scene->num_instances; }

    // Получить инстанс
    const ogt_vox_instance& getInstance(uint32_t index) const;

    // Получить палитру сцены
    const ogt_vox_palette& palette() const { return m_scene->palette; }

    // Версия файла
    uint32_t fileVersion() const { return m_scene->file_version; }

private:
    const ogt_vox_scene* m_scene;
};

}
