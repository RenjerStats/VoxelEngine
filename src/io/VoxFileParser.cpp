#include <QDebug>
#include <QFile>

#define OGT_VOX_IMPLEMENTATION
#include "io/VoxFileParser.h"

#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cstring>

namespace VoxIO {

struct Vec3Hash {
    std::size_t operator()(const std::tuple<int32_t, int32_t, int32_t>& v) const {
        std::size_t h1 = std::hash<int32_t>{}(std::get<0>(v));
        std::size_t h2 = std::hash<int32_t>{}(std::get<1>(v));
        std::size_t h3 = std::hash<int32_t>{}(std::get<2>(v));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

std::unique_ptr<VoxScene> VoxFileParser::load(const QString& filePath)
{
    return loadWithFlags(filePath, 0);
}

std::unique_ptr<VoxScene> VoxFileParser::loadWithFlags(const QString& filePath, uint32_t readFlags)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Failed to open file:" << filePath;
        return nullptr;
    }

    QByteArray fileData = file.readAll();
    file.close();

    if (fileData.isEmpty()) {
        qWarning() << "File is empty:" << filePath;
        return nullptr;
    }

    qDebug() << "Loaded" << fileData.size() << "bytes from" << filePath;

    const uint8_t* buffer = reinterpret_cast<const uint8_t*>(fileData.constData());
    uint32_t bufferSize = static_cast<uint32_t>(fileData.size());

    const ogt_vox_scene* scene = nullptr;

    if (readFlags == 0) {
        scene = ogt_vox_read_scene(buffer, bufferSize);
    } else {
        scene = ogt_vox_read_scene_with_flags(buffer, bufferSize, readFlags);
    }

    if (!scene) {
        qWarning() << "Failed to parse .vox file:" << filePath;
        return nullptr;
    }

    for (uint32_t i = 0; i < scene->num_models; ++i) {
        const ogt_vox_model* model = scene->models[i];
        qDebug() << "  Model" << i << ":"
                 << model->size_x << "x"
                 << model->size_y << "x"
                 << model->size_z;
    }

    return std::make_unique<VoxScene>(scene);
}

bool VoxFileParser::save(const QString& filePath, const std::vector<RenderVoxel>& voxelsToSave, const ogt_vox_palette& palette) {
    if (voxelsToSave.empty()) {
        qWarning() << "Cannot save empty voxel data";
        return false;
    }

    std::unordered_map<std::tuple<int32_t, int32_t, int32_t>, uint8_t, Vec3Hash> uniqueVoxels;
    int32_t minX = std::numeric_limits<int32_t>::max();
    int32_t minY = std::numeric_limits<int32_t>::max();
    int32_t minZ = std::numeric_limits<int32_t>::max();
    int32_t maxX = std::numeric_limits<int32_t>::lowest();
    int32_t maxY = std::numeric_limits<int32_t>::lowest();
    int32_t maxZ = std::numeric_limits<int32_t>::lowest();

    for (const RenderVoxel& voxel : voxelsToSave) {
        if (voxel.colorID == 255) {
            continue;
        }

        int32_t x = static_cast<int32_t>(std::floor(voxel.x));
        int32_t y = static_cast<int32_t>(std::floor(voxel.y));
        int32_t z = static_cast<int32_t>(std::floor(voxel.z));

        if (std::abs(x) >= 256 || std::abs(y) >= 256 || std::abs(z) >= 256) {
            continue;
        }

        auto key = std::make_tuple(x, y, z);
        if (uniqueVoxels.find(key) != uniqueVoxels.end()) {
            continue;
        }

        uint8_t colorIndex = static_cast<uint8_t>(std::clamp(static_cast<int>(voxel.colorID), 1, 255));
        uniqueVoxels[key] = colorIndex;
        minX = std::min(minX, x);
        minY = std::min(minY, y);
        minZ = std::min(minZ, z);
        maxX = std::max(maxX, x);
        maxY = std::max(maxY, y);
        maxZ = std::max(maxZ, z);
    }

    if (uniqueVoxels.empty()) {
        qWarning() << "No valid voxels to save after filtering";
        return false;
    }

    minX = std::max(minX, 0);
    minY = std::max(minY, 0);
    minZ = std::max(minZ, 0);
    maxX = std::min(maxX, 255);
    maxY = std::min(maxY, 255);
    maxZ = std::min(maxZ, 255);

    uint32_t sizeX = static_cast<uint32_t>(maxX - minX + 1);
    uint32_t sizeY = static_cast<uint32_t>(maxZ - minZ + 1);
    uint32_t sizeZ = static_cast<uint32_t>(maxY - minY + 1);

    sizeX = std::min(sizeX, uint32_t(256));
    sizeY = std::min(sizeY, uint32_t(256));
    sizeZ = std::min(sizeZ, uint32_t(256));

    const uint32_t voxelCount = sizeX * sizeY * sizeZ;
    std::vector<uint8_t> voxelData(voxelCount, 0);

    int voxelsWritten = 0;
    for (const auto& [key, colorIndex] : uniqueVoxels) {
        int32_t worldX = std::get<0>(key);
        int32_t worldY = std::get<1>(key);
        int32_t worldZ = std::get<2>(key);

        int32_t localX = worldX - minX;
        int32_t localY = worldY - minY;
        int32_t localZ = worldZ - minZ;

        uint32_t voxX = static_cast<uint32_t>(localX);
        uint32_t voxY = static_cast<uint32_t>(localZ);
        uint32_t voxZ = static_cast<uint32_t>(localY);

        if (voxX < sizeX && voxY < sizeY && voxZ < sizeZ) {
            uint32_t index = voxX + (voxY * sizeX) + (voxZ * sizeX * sizeY);

            if (index < voxelCount) {
                voxelData[index] = colorIndex;
                voxelsWritten++;
            }
        }
    }

    qDebug() << "Voxels written to array:" << voxelsWritten;

    ogt_vox_model model;
    model.size_x = sizeX;
    model.size_y = sizeY;
    model.size_z = sizeZ;
    model.voxel_data = voxelData.data();
    model.voxel_hash = 0;

    const ogt_vox_model* models[1];
    models[0] = &model;

    ogt_vox_layer layer;
    std::memset(&layer, 0, sizeof(layer));
    layer.name = nullptr;
    layer.hidden = false;
    layer.color = {255, 255, 255, 255};

    ogt_vox_group root_group;
    std::memset(&root_group, 0, sizeof(root_group));
    root_group.name = nullptr;
    root_group.hidden = false;
    root_group.layer_index = 0;
    root_group.parent_group_index = k_invalid_group_index;

    root_group.transform.m00 = 1.0f;
    root_group.transform.m11 = 1.0f;
    root_group.transform.m22 = 1.0f;
    root_group.transform.m33 = 1.0f;
    root_group.transform.m01 = root_group.transform.m02 = root_group.transform.m03 = 0.0f;
    root_group.transform.m10 = root_group.transform.m12 = root_group.transform.m13 = 0.0f;
    root_group.transform.m20 = root_group.transform.m21 = root_group.transform.m23 = 0.0f;
    root_group.transform.m30 = root_group.transform.m31 = root_group.transform.m32 = 0.0f;

    root_group.transform_anim.keyframes = nullptr;
    root_group.transform_anim.num_keyframes = 0;
    root_group.transform_anim.loop = false;

    ogt_vox_instance instance;
    std::memset(&instance, 0, sizeof(instance));
    instance.name = nullptr;
    instance.model_index = 0;
    instance.layer_index = 0;
    instance.group_index = 0;
    instance.hidden = false;

    instance.transform.m00 = 1.0f;
    instance.transform.m11 = 1.0f;
    instance.transform.m22 = 1.0f;
    instance.transform.m33 = 1.0f;
    instance.transform.m01 = instance.transform.m02 = instance.transform.m03 = 0.0f;
    instance.transform.m10 = instance.transform.m12 = instance.transform.m13 = 0.0f;
    instance.transform.m20 = instance.transform.m21 = instance.transform.m23 = 0.0f;
    instance.transform.m30 = instance.transform.m31 = instance.transform.m32 = 0.0f;

    instance.transform_anim.keyframes = nullptr;
    instance.transform_anim.num_keyframes = 0;
    instance.transform_anim.loop = false;
    instance.model_anim.keyframes = nullptr;
    instance.model_anim.num_keyframes = 0;
    instance.model_anim.loop = false;

    ogt_vox_scene scene;
    std::memset(&scene, 0, sizeof(scene));

    scene.num_models = 1;
    scene.models = models;

    scene.num_instances = 1;
    scene.instances = &instance;

    scene.num_layers = 1;
    scene.layers = &layer;

    scene.num_groups = 1;
    scene.groups = &root_group;

    scene.palette = palette;

    scene.num_cameras = 0;
    scene.cameras = nullptr;
    scene.sun = nullptr;

    uint32_t bufferSize = 0;
    uint8_t* buffer = ogt_vox_write_scene(&scene, &bufferSize);

    if (!buffer || bufferSize == 0) {
        qWarning() << "Failed to write scene to buffer";
        return false;
    }

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        qWarning() << "Failed to open file for writing:" << filePath;
        ogt_vox_free(buffer);
        return false;
    }

    qint64 written = file.write(reinterpret_cast<const char*>(buffer), bufferSize);
    file.close();
    ogt_vox_free(buffer);

    if (written != bufferSize) {
        qWarning() << "Failed to write all data to file";
        return false;
    }

    qDebug() << "Successfully saved" << written << "bytes to" << filePath;
    return true;
}

}
