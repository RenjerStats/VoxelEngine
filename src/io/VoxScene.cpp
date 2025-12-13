#include "io/VoxScene.h"
#include "core/Types.h"
#include <QDebug>

namespace VoxIO {

VoxModel::VoxModel(const ogt_vox_model* model, const ogt_vox_palette& palette)
    : model(model)
    , palette(palette){}

uint8_t VoxModel::getVoxelIndex(uint32_t x, uint32_t y, uint32_t z) const
{
    if (x >= model->size_x || y >= model->size_y || z >= model->size_z) {
        return 0;
    }

    uint32_t voxelIndex = x + (y * model->size_x) + (z * model->size_x * model->size_y);
    return model->voxel_data[voxelIndex];
}

QColor VoxModel::getVoxelColor(uint32_t x, uint32_t y, uint32_t z) const
{
    uint8_t colorIndex = getVoxelIndex(x, y, z);

    if (colorIndex == 0) {
        return QColor();
    }

    const ogt_vox_rgba& rgba = palette.color[colorIndex];
    return QColor(rgba.r, rgba.g, rgba.b, rgba.a);
}

bool VoxModel::isVoxelSolid(uint32_t x, uint32_t y, uint32_t z) const
{
    return getVoxelIndex(x, y, z) != 0;
}

void VoxModel::forEachSolidVoxel(std::function<void(const Voxel&)> callback) const
{
    const uint32_t totalVoxels = model->size_x * model->size_y * model->size_z;

    for (uint32_t voxelIndex = 0; voxelIndex < totalVoxels; ++voxelIndex) {
        uint8_t colorIndex = model->voxel_data[voxelIndex];

        if (colorIndex == 0) {
            continue;
        }

        uint32_t x = voxelIndex % model->size_x;
        uint32_t y = (voxelIndex / model->size_x) % model->size_y;
        uint32_t z = voxelIndex / (model->size_x * model->size_y);

        Voxel voxel;
        voxel.x = y;
        voxel.y = z;
        voxel.z = x; // Меняем местами, из-за разницы в системах .vox и OpenGL
        voxel.colorIndex = colorIndex;

        const ogt_vox_rgba& rgba = palette.color[colorIndex];
        voxel.color = QColor(rgba.r, rgba.g, rgba.b, rgba.a);

        callback(voxel);
    }
}


std::vector<Voxel> VoxModel::getSolidVoxels() const
{
    std::vector<Voxel> voxels;

    voxels.reserve((model->size_x * model->size_y * model->size_z) / 3);

    forEachSolidVoxel([&voxels](const Voxel& voxel) {
        voxels.push_back(voxel);
    });

    return voxels;
}

std::vector<RenderVoxel> VoxModel::getCudaVoxels() const
{
    std::vector<RenderVoxel> cudaVoxels;
    cudaVoxels.reserve((model->size_x * model->size_y * model->size_z) / 3);

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();

    float maxX = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();

    forEachSolidVoxel([&](const Voxel& v) {
        RenderVoxel cv;
        cv.x = (float)v.x;
        cv.y = (float)v.y;
        cv.z = (float)v.z;
        cv.colorID = (float)v.colorIndex;

        cudaVoxels.push_back(cv);

        minX = std::min(minX, cv.x);
        maxX = std::max(maxX, cv.x);
        minY = std::min(minY, cv.y);
        minZ = std::min(minZ, cv.z);
        maxZ = std::max(maxZ, cv.z);
    });


    float diffX = maxX-minX;
    float diffZ = maxZ-minZ;
    for (int x = minX-(int)(diffX*0.4); x < maxX+(int)(diffX*0.4); ++x) {
        for (int z = minZ-(int)(diffZ*0.4); z < maxZ+(int)(diffZ*0.4); ++z) {
            RenderVoxel cv;
            cv.x = (float)x;
            cv.y = -1;
            cv.z = (float)z;
            cv.colorID = 255; // зарезервированный цвет для неподвижных вокселей, нарпимер для пола

            cudaVoxels.push_back(cv);
        }
    }

    return cudaVoxels;
}




VoxScene::VoxScene(const ogt_vox_scene* scene)
    : scene(scene)
{
    if (!scene) {
        qWarning() << "VoxScene: scene is null!";
    }
}

VoxScene::~VoxScene()
{
    if (scene) {
        ogt_vox_destroy_scene(scene);
    }
}

VoxModel VoxScene::getModel(uint32_t index) const
{
    if (index >= scene->num_models) {
        qWarning() << "VoxScene: model index" << index << "out of bounds!";
        return VoxModel(nullptr, scene->palette);
    }

    return VoxModel(scene->models[index], scene->palette);
}

const ogt_vox_instance& VoxScene::getInstance(uint32_t index) const
{
    if (index >= scene->num_instances) {
        qWarning() << "VoxScene: instance index" << index << "out of bounds!";
        static ogt_vox_instance dummy{};
        return dummy;
    }

    return scene->instances[index];
}

}
