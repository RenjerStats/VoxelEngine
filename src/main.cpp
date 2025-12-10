#include <QDebug>
#include <vector>
#define NSIGHT_DEBUG

#ifdef NSIGHT_DEBUG
// --- HEADLESS MODE ---
#include "physics/PhysicsManager.h"
#include "io/VoxFileParser.h"
#include "io/VoxScene.h"
#include "core/Types.h"
#include <cuda_runtime.h>
#include <iostream>

using namespace VoxIO;

int main(int argc, char *argv[]) {
    // В этом режиме мы НЕ создаем QApplication, чтобы избежать инициализации OpenGL контекста

    std::cout << ">>> NSIGHT COMPUTE DEBUG MODE ENABLED <<<" << std::endl;
    std::cout << ">>> NO GUI / NO OPENGL INTEROP <<<" << std::endl;

    // 1. Загрузка данных (путь хардкодим или берем из argv)
    QString scenePath = "assets/test4.vox";
    std::cout << "Loading scene: " << scenePath.toStdString() << std::endl;

    auto scene = VoxFileParser::load(scenePath);
    if (!scene || scene->modelCount() == 0) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    // Извлекаем воксели (как в VoxelWindow::loadScene)
    VoxModel model = scene->getModel(0);
    std::vector<CudaVoxel> hostVoxels = model.getCudaVoxels();
    std::cout << "Loaded voxels: " << hostVoxels.size() << std::endl;

    // 2. Инициализация физики
    PhysicsManager physics(60, hostVoxels.size());

    // ВАЖНО: Используем новый метод загрузки, минуя VBO
    physics.uploadVoxelsToGPU(hostVoxels);
    physics.initSumulation();

    // 3. Цикл симуляции
    // Запускаем фиксированное количество кадров, чтобы профайлер мог собрать статистику
    int totalFrames = 100;
    std::cout << "Starting simulation for " << totalFrames << " frames..." << std::endl;

    for (int i = 0; i < totalFrames; ++i) {
        // Параметры: speed = 1.0, stability = 1.0 (можно менять для нагрузки)
        physics.updatePhysics(1.0f, 1.0f);

        cudaDeviceSynchronize();

        if (i % 10 == 0) {
            std::cout << "Frame " << i << " complete." << std::endl;
        }
    }

    std::cout << "Simulation finished. Exiting." << std::endl;
    return 0;
}

#else
// --- STANDARD MODE ---
#include <QApplication>
#include "ui/MainWindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    return app.exec();
}
#endif
