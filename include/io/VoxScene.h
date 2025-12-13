#pragma once

#include <QVector3D>
#include <QColor>
#include <QString>
#include <functional>

#include "ogt_vox.h"

struct RenderVoxel;

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

    uint32_t sizeX() const { return model->size_x; }
    uint32_t sizeY() const { return model->size_y; }
    uint32_t sizeZ() const { return model->size_z; }

    uint8_t getVoxelIndex(uint32_t x, uint32_t y, uint32_t z) const;
    QColor getVoxelColor(uint32_t x, uint32_t y, uint32_t z) const;

    bool isVoxelSolid(uint32_t x, uint32_t y, uint32_t z) const;

    void forEachSolidVoxel(std::function<void(const Voxel& voxel)> callback) const;

    std::vector<Voxel> getSolidVoxels() const;

    std::vector<RenderVoxel> getCudaVoxels() const;

    const ogt_vox_palette& getPalette() const { return palette; }

private:
    const ogt_vox_model* model;
    ogt_vox_palette palette;
};

class VoxScene {
public:
    VoxScene(const ogt_vox_scene* scene);
    ~VoxScene();

    VoxScene(const VoxScene&) = delete;
    VoxScene& operator=(const VoxScene&) = delete;

    uint32_t modelCount() const { return scene->num_models; }

    VoxModel getModel(uint32_t index) const;

    uint32_t instanceCount() const { return scene->num_instances; }

    const ogt_vox_instance& getInstance(uint32_t index) const;

    const ogt_vox_palette& palette() const { return scene->palette; }

    uint32_t fileVersion() const { return scene->file_version; }

private:
    const ogt_vox_scene* scene;
};

}
