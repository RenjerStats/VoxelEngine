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

void MainWindow::setupUI() {
    // 1. Главный контейнер
    QWidget* centralWidget = new QWidget(this);
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);
    mainLayout->setContentsMargins(0, 0, 0, 0); // Убираем отступы по краям
    mainLayout->setSpacing(0);

    // 2. Панель управления (Слева)
    QWidget* controlPanel = new QWidget(this);
    controlPanel->setFixedWidth(250); // Фиксированная ширина
    controlPanel->setStyleSheet("background-color: #2D2D30; color: #CCCCCC;"); // Темная тема

    QVBoxLayout* panelLayout = new QVBoxLayout(controlPanel);

    // Элементы управления
    QPushButton* btnReset = new QPushButton("Reset Simulation", controlPanel);
    QLabel* lblGravity = new QLabel("Gravity:", controlPanel);
    QSlider* sldGravity = new QSlider(Qt::Horizontal, controlPanel);

    // Добавляем их на панель
    panelLayout->addWidget(new QLabel("<b>Controls</b>", controlPanel));
    panelLayout->addSpacing(10);
    panelLayout->addWidget(btnReset);
    panelLayout->addSpacing(20);
    panelLayout->addWidget(lblGravity);
    panelLayout->addWidget(sldGravity);
    panelLayout->addStretch(); // Прижать всё вверх

    // 3. Окно рендеринга (Справа)
    // Создаем нативное окно
    voxelWindow = new VoxelWindow();
    voxelWindow->setScenePath("assets/test_scene.vox");

    // Оборачиваем его в виджет-контейнер
    QWidget* renderContainer = QWidget::createWindowContainer(voxelWindow);

    // Важные настройки фокуса и размера
    renderContainer->setMinimumSize(800, 600);
    renderContainer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    renderContainer->setFocusPolicy(Qt::StrongFocus); // Чтобы ловить клики и клавиши

    // 4. Сборка лайаута
    mainLayout->addWidget(controlPanel);
    mainLayout->addWidget(renderContainer); // Займет всё оставшееся место

    setCentralWidget(centralWidget);
    statusBar()->showMessage("Engine Ready. CUDA initialized.");

    // Пример соединения сигналов (понадобится позже)
    // connect(btnReset, &QPushButton::clicked, [this](){ /* reset logic */ });
}
