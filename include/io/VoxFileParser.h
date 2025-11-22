#pragma once

#include <QString>
#include <QFile>
#include <memory>
#include "VoxScene.h"

namespace VoxIO {

class VoxFileParser {
public:
    // Загрузить .vox файл
    static std::unique_ptr<VoxScene> load(const QString& filePath);

    // Загрузить с дополнительными флагами
    // Флаги из ogt_vox.h:
    // - k_read_scene_flags_groups: читать информацию о группах
    // - k_read_scene_flags_keyframes: читать анимацию
    // - k_read_scene_flags_keep_empty_models_instances: не удалять пустые модели
    static std::unique_ptr<VoxScene> loadWithFlags(const QString& filePath, uint32_t readFlags);

    // Сохранить сцену в .vox файл (если понадобится)
    static bool save(const QString& filePath, const VoxScene& scene);

private:
    VoxFileParser() = default;
};

}
