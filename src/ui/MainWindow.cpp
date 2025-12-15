#include "ui/MainWindow.h"
#include "ui/VoxelWindow.h"

#include <QWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QSlider>
#include <QLabel>
#include <QStatusBar>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("Voxel Engine (Qt6 + CUDA)");
    resize(1280, 720);
    setupUI();
}

// Вспомогательная функция для создания слайдера с подписью
void addSliderControl(QVBoxLayout* layout, const QString& name, int min, int max, int val,
                      std::function<void(int)> onChange, double scaleFactor = 1.0) {

    QLabel* label = new QLabel(QString("%1: %2").arg(name).arg(val * scaleFactor), layout->parentWidget());
    QSlider* slider = new QSlider(Qt::Horizontal, layout->parentWidget());
    slider->setRange(min, max);
    slider->setValue(val);

    QObject::connect(slider, &QSlider::valueChanged, [label, name, scaleFactor, onChange](int value) {
        // Обновляем текст метки
        double realValue = value * scaleFactor;
        label->setText(QString("%1: %2").arg(name).arg(realValue, 0, 'f', 2));
        // Вызываем функцию обновления
        onChange(value);
    });

    layout->addWidget(label);
    layout->addWidget(slider);
    layout->addSpacing(5);
}

void MainWindow::setupUI() {
    QWidget* centralWidget = new QWidget(this);
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // --- Панель управления ---
    QWidget* controlPanel = new QWidget(this);
    controlPanel->setFixedWidth(300);
    controlPanel->setStyleSheet("background-color: #2D2D30; color: #CCCCCC; QSlider { height: 20px; }");

    QVBoxLayout* panelLayout = new QVBoxLayout(controlPanel);
    panelLayout->setAlignment(Qt::AlignTop);

    panelLayout->addWidget(new QLabel("<b>Scene Settings</b>", controlPanel));
    panelLayout->addSpacing(10);

    // Создаем окно рендеринга (оно нужно нам для замыканий в лямбдах)
    voxelWindow = new VoxelWindow();
    voxelWindow->setScenePath("assets/test6.vox");

    // --- 1. Слайдер FOV (Field of View) ---
    // Диапазон: 10..120 градусов
    addSliderControl(panelLayout, "FOV", 10, 120, 45, [this](int v){
        voxelWindow->setFOV((float)v);
    });

    // --- 2. Дистанция до модели ---
    // Диапазон: 1..128
    addSliderControl(panelLayout, "distance to model", 5, 256, 100, [this](int v){
        voxelWindow->setDistance((float)v);
    });

    // Разделитель
    panelLayout->addSpacing(15);
    panelLayout->addWidget(new QLabel("<b>Lighting (Direction)</b>", controlPanel));

    // --- 3. Освещение (X, Y, Z) ---


    addSliderControl(panelLayout, "Light X", -100, 100, 50, [this](int v){
        voxelWindow->setLightDirX(v);
    }, 0.01);
    addSliderControl(panelLayout, "Light Y", -100, 100, 100, [this](int v){
        voxelWindow->setLightDirY(v);
    }, 0.01);
    addSliderControl(panelLayout, "Light Z", -100, 100, 30, [this](int v){
        voxelWindow->setLightDirZ(v);
    }, 0.01);

    // Разделитель
    panelLayout->addSpacing(15);
    panelLayout->addWidget(new QLabel("<b>Camera Position</b>", controlPanel));

    // --- 4. Вращение камеры (X, Y, Z) ---

    addSliderControl(panelLayout, "Cam X", -180, 180, 180, [this](int v){
        voxelWindow->setCameraRotationX(v);
    });
    addSliderControl(panelLayout, "Cam Y", -180, 180, -45, [this](int v){
        voxelWindow->setCameraRotationY(v);
    });
    addSliderControl(panelLayout, "Cam Z", -180, 180, 180, [this](int v){
        voxelWindow->setCameraRotationZ(v);
    });

    // Разделитель
    panelLayout->addSpacing(15);
    panelLayout->addWidget(new QLabel("<b>Physics control</b>", controlPanel));

    QPushButton* resetButton = new QPushButton("Reset Simulation", controlPanel);
    panelLayout->addWidget(resetButton);

    connect(resetButton, &QPushButton::clicked, [this](){
        voxelWindow->resetSimulation();
    });


    panelLayout->addStretch();

    // Оборачиваем VoxelWindow
    QWidget* renderContainer = QWidget::createWindowContainer(voxelWindow);
    renderContainer->setMinimumSize(800, 600);
    renderContainer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    renderContainer->setFocusPolicy(Qt::StrongFocus);

    // Сборка лайаута
    mainLayout->addWidget(controlPanel);
    mainLayout->addWidget(renderContainer);

    setCentralWidget(centralWidget);
    statusBar()->showMessage("Engine Ready.");
}
