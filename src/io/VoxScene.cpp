#include "io/VoxScene.h"
#include <QDebug>

namespace VoxIO {

VoxModel::VoxModel(const ogt_vox_model* model, const ogt_vox_palette& palette)
    : m_model(model)
    , m_palette(palette){}

uint8_t VoxModel::getVoxelIndex(uint32_t x, uint32_t y, uint32_t z) const
{
    if (x >= m_model->size_x || y >= m_model->size_y || z >= m_model->size_z) {
        return 0;
    }

    uint32_t voxelIndex = x + (y * m_model->size_x) + (z * m_model->size_x * m_model->size_y);
    return m_model->voxel_data[voxelIndex];
}

QColor VoxModel::getVoxelColor(uint32_t x, uint32_t y, uint32_t z) const
{
    uint8_t colorIndex = getVoxelIndex(x, y, z);

    if (colorIndex == 0) {
        return QColor(); // Пустой воксель - невалидный цвет
    }

    const ogt_vox_rgba& rgba = m_palette.color[colorIndex];
    return QColor(rgba.r, rgba.g, rgba.b, rgba.a);
}

bool VoxModel::isVoxelSolid(uint32_t x, uint32_t y, uint32_t z) const
{
    return getVoxelIndex(x, y, z) != 0;
}

// МЕТОД 1: Обход всех вокселей (медленный, если много пустых)
void VoxModel::forEachVoxel(std::function<void(uint32_t, uint32_t, uint32_t, uint8_t)> callback) const
{
    for (uint32_t z = 0; z < m_model->size_z; ++z) {
        for (uint32_t y = 0; y < m_model->size_y; ++y) {
            for (uint32_t x = 0; x < m_model->size_x; ++x) {
                uint8_t colorIndex = getVoxelIndex(x, y, z);
                callback(x, y, z, colorIndex);
            }
        }
    }
}

// МЕТОД 2: Обход только заполненных вокселей (ОПТИМИЗИРОВАННЫЙ!)
void VoxModel::forEachSolidVoxel(std::function<void(const Voxel&)> callback) const
{
    const uint32_t totalVoxels = m_model->size_x * m_model->size_y * m_model->size_z;

    // Прямой проход по массиву данных - очень быстро!
    for (uint32_t voxelIndex = 0; voxelIndex < totalVoxels; ++voxelIndex) {
        uint8_t colorIndex = m_model->voxel_data[voxelIndex];

        // colorIndex == 0 означает пустой воксель - пропускаем
        if (colorIndex == 0) {
            continue;
        }

        uint32_t x = voxelIndex % m_model->size_x;
        uint32_t y = (voxelIndex / m_model->size_x) % m_model->size_y;
        uint32_t z = voxelIndex / (m_model->size_x * m_model->size_y);

        // Создаем структуру Voxel
        Voxel voxel;
        voxel.x = x;
        voxel.y = y;
        voxel.z = z;
        voxel.colorIndex = colorIndex;

        const ogt_vox_rgba& rgba = m_palette.color[colorIndex];
        voxel.color = QColor(rgba.r, rgba.g, rgba.b, rgba.a);

        callback(voxel);
    }
}

// МЕТОД 3: Обход по слоям (полезно для визуализации по слоям)
void VoxModel::forEachLayer(std::function<void(uint32_t, const std::vector<Voxel>&)> callback) const
{
    for (uint32_t z = 0; z < m_model->size_z; ++z) {
        std::vector<Voxel> layerVoxels;

        for (uint32_t y = 0; y < m_model->size_y; ++y) {
            for (uint32_t x = 0; x < m_model->size_x; ++x) {
                uint8_t colorIndex = getVoxelIndex(x, y, z);

                if (colorIndex != 0) {
                    Voxel voxel;
                    voxel.x = x;
                    voxel.y = y;
                    voxel.z = z;
                    voxel.colorIndex = colorIndex;

                    const ogt_vox_rgba& rgba = m_palette.color[colorIndex];
                    voxel.color = QColor(rgba.r, rgba.g, rgba.b, rgba.a);

                    layerVoxels.push_back(voxel);
                }
            }
        }

        if (!layerVoxels.empty()) {
            callback(z, layerVoxels);
        }
    }
}

// МЕТОД 4: Получить все заполненные вокселы в вектор
std::vector<Voxel> VoxModel::getSolidVoxels() const
{
    std::vector<Voxel> voxels;

    // Резервируем примерную память (предполагаем ~30% заполненность)
    voxels.reserve((m_model->size_x * m_model->size_y * m_model->size_z) / 3);

    forEachSolidVoxel([&voxels](const Voxel& voxel) {
        voxels.push_back(voxel);
    });

    return voxels;
}

VoxScene::VoxScene(const ogt_vox_scene* scene)
    : m_scene(scene)
{
    if (!m_scene) {
        qWarning() << "VoxScene: scene is null!";
    }
}

VoxScene::~VoxScene()
{
    if (m_scene) {
        ogt_vox_destroy_scene(m_scene);
    }
}

VoxModel VoxScene::getModel(uint32_t index) const
{
    if (index >= m_scene->num_models) {
        qWarning() << "VoxScene: model index" << index << "out of bounds!";
        return VoxModel(nullptr, m_scene->palette);
    }

    return VoxModel(m_scene->models[index], m_scene->palette);
}

const ogt_vox_instance& VoxScene::getInstance(uint32_t index) const
{
    if (index >= m_scene->num_instances) {
        qWarning() << "VoxScene: instance index" << index << "out of bounds!";
        static ogt_vox_instance dummy{};
        return dummy;
    }

    return m_scene->instances[index];
}

}
