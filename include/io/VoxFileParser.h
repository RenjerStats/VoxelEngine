#pragma once

#include <QString>
#include <QFile>
#include <memory>
#include "VoxScene.h"
#include "core/Types.h"


namespace VoxIO {
class VoxFileParser {
public:
    static std::unique_ptr<VoxScene> load(const QString& filePath);

    static std::unique_ptr<VoxScene> loadWithFlags(const QString& filePath, uint32_t readFlags);

    static bool save(const QString& filePath, const std::vector<RenderVoxel>& voxelsToSave, const ogt_vox_palette& palette);

private:
    VoxFileParser() = default;
};

}
