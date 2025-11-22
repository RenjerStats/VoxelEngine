#include <QDebug>

#define OGT_VOX_IMPLEMENTATION
#include "io/VoxFileParser.h"

namespace VoxIO {

std::unique_ptr<VoxScene> VoxFileParser::load(const QString& filePath)
{
    return loadWithFlags(filePath, 0); // Флаги по умолчанию
}

std::unique_ptr<VoxScene> VoxFileParser::loadWithFlags(const QString& filePath, uint32_t readFlags)
{
    // Шаг 1: Открыть файл
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        qWarning() << "Failed to open file:" << filePath;
        return nullptr;
    }

    // Шаг 2: Прочитать все данные в буфер
    QByteArray fileData = file.readAll();
    file.close();

    if (fileData.isEmpty()) {
        qWarning() << "File is empty:" << filePath;
        return nullptr;
    }

    qDebug() << "Loaded" << fileData.size() << "bytes from" << filePath;

    // Шаг 3: Парсинг через ogt_vox
    const uint8_t* buffer = reinterpret_cast<const uint8_t*>(fileData.constData());
    uint32_t bufferSize = static_cast<uint32_t>(fileData.size());

    const ogt_vox_scene* scene = nullptr;

    if (readFlags == 0) {
        // Обычная загрузка
        scene = ogt_vox_read_scene(buffer, bufferSize);
    } else {
        // Загрузка с флагами
        scene = ogt_vox_read_scene_with_flags(buffer, bufferSize, readFlags);
    }

    if (!scene) {
        qWarning() << "Failed to parse .vox file:" << filePath;
        return nullptr;
    }

    // Шаг 4: Информация о загруженной сцене
    qDebug() << "=== VOX Scene Info ===";
    qDebug() << "File version:" << scene->file_version;
    qDebug() << "Models:" << scene->num_models;
    qDebug() << "Instances:" << scene->num_instances;
    qDebug() << "Layers:" << scene->num_layers;
    qDebug() << "Groups:" << scene->num_groups;

    for (uint32_t i = 0; i < scene->num_models; ++i) {
        const ogt_vox_model* model = scene->models[i];
        qDebug() << "  Model" << i << ":"
                 << model->size_x << "x"
                 << model->size_y << "x"
                 << model->size_z;
    }

    // Шаг 5: Создаем обертку VoxScene
    return std::make_unique<VoxScene>(scene);
}

bool VoxFileParser::save(const QString& filePath, const VoxScene& scene)
{
    // Получаем указатель на ogt_vox_scene из VoxScene
    // Для этого нужно добавить метод в VoxScene для доступа к внутреннему указателю

    // TODO: Реализовать сохранение, если потребуется
    qWarning() << "Save not implemented yet";
    return false;
}

}
