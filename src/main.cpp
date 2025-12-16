//#define NSIGHT_DEBUG

#ifdef NSIGHT_DEBUG
// --- HEADLESS MODE ---
#include "physics/PhysicsManager.h"
#include "io/VoxFileParser.h"
#include "io/VoxScene.h"
#include "core/Types.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <QDebug>

using namespace VoxIO;

int main(int argc, char *argv[]) {
    std::cout << ">>> NSIGHT COMPUTE DEBUG MODE ENABLED <<<" << std::endl;
    std::cout << ">>> NO GUI / NO OPENGL INTEROP <<<" << std::endl;

    QString scenePath = "assets/test5.vox";
    std::cout << "Loading scene: " << scenePath.toStdString() << std::endl;

    auto scene = VoxFileParser::load(scenePath);
    if (!scene || scene->modelCount() == 0) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    VoxModel model = scene->getModel(0);
    std::vector<RenderVoxel> hostVoxels = model.getCudaVoxels();
    std::cout << "Loaded voxels: " << hostVoxels.size() << std::endl;

    PhysicsManager physics(60, hostVoxels.size());

    physics.uploadVoxelsToGPU(hostVoxels);

    int totalFrames = 600;
    std::cout << "Starting simulation for " << totalFrames << " frames..." << std::endl;

    for (int i = 0; i < totalFrames; ++i) {
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
#include <QHBoxLayout>
#include <QWidget>
#include "ui/VoxelWindow.h"
#include "ui/ControlPanel.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QWidget* mainWindow = new QWidget();
    mainWindow->setWindowTitle("Voxel Engine CUDA Physics");
    mainWindow->resize(1280, 720);

    VoxelWindow* voxelWindow = new VoxelWindow();

    QWidget* container = QWidget::createWindowContainer(voxelWindow);
    container->setMinimumSize(800, 600);
    container->setFocusPolicy(Qt::StrongFocus);

    ControlPanel* controlPanel = new ControlPanel(voxelWindow);
    controlPanel->setFixedWidth(300);

    QHBoxLayout* layout = new QHBoxLayout(mainWindow);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(container, 1);
    layout->addWidget(controlPanel, 0);

    voxelWindow->setScenePath("assets/monu2.vox");

    mainWindow->show();

    return app.exec();
}
#endif
