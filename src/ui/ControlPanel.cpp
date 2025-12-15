#include "ui/ControlPanel.h"
#include "ui/VoxelWindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>

ControlPanel::ControlPanel(VoxelWindow* voxelWindow, QWidget* parent)
    : QWidget(parent), m_voxelWindow(voxelWindow)
{
    setupUI();
    connect(this, &ControlPanel::signalTogglePause, [this](bool paused){
        if(paused) m_voxelWindow->setPaused(true);
        else m_voxelWindow->setPaused(false);
    });

    connect(this, &ControlPanel::signalResetSimulation, m_voxelWindow, &VoxelWindow::resetSimulation);

    connect(this, &ControlPanel::signalSpawnObject, [this](int type, float velocity, int size){
        if (type == 0) {
            m_voxelWindow->spawnSphereFromCamera(velocity, size);
        } else {
            m_voxelWindow->spawnCubeFromCamera(velocity, size);
        }
    });
}

void ControlPanel::setupUI() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    mainLayout->setSpacing(15);

    QGroupBox* groupSim = new QGroupBox("Simulation Control", this);
    QVBoxLayout* simLayout = new QVBoxLayout(groupSim);

    btnPause = new QPushButton("Pause", this);
    btnPause->setCheckable(true);

    btnReset = new QPushButton("Reset Scene", this);

    simLayout->addWidget(btnPause);
    simLayout->addWidget(btnReset);
    mainLayout->addWidget(groupSim);

    QGroupBox* groupSpawn = new QGroupBox("Spawn Objects", this);
    QFormLayout* spawnLayout = new QFormLayout(groupSpawn);

    comboSpawnType = new QComboBox(this);
    comboSpawnType->addItem("Sphere");
    comboSpawnType->addItem("Cube");

    spinSize = new QSpinBox(this);
    spinSize->setRange(1, 100);
    spinSize->setValue(10);
    spinSize->setSuffix(" vox");

    spinVelocity = new QDoubleSpinBox(this);
    spinVelocity->setRange(0.0, 500.0);
    spinVelocity->setValue(50.0);
    spinVelocity->setSingleStep(5.0);

    btnSpawn = new QPushButton("Shoot Object", this);
    btnSpawn->setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;");

    spawnLayout->addRow("Type:", comboSpawnType);
    spawnLayout->addRow("Size:", spinSize);
    spawnLayout->addRow("Speed:", spinVelocity);
    spawnLayout->addWidget(btnSpawn);

    mainLayout->addWidget(groupSpawn);

    mainLayout->addStretch();

    connect(btnPause, &QPushButton::clicked, this, &ControlPanel::onPauseClicked);
    connect(btnReset, &QPushButton::clicked, this, &ControlPanel::signalResetSimulation);
    connect(btnSpawn, &QPushButton::clicked, this, &ControlPanel::onSpawnClicked);
}

void ControlPanel::onPauseClicked() {
    isPaused = btnPause->isChecked();
    btnPause->setText(isPaused ? "Resume" : "Pause");
    Q_EMIT signalTogglePause(isPaused);
}

void ControlPanel::onSpawnClicked() {
    int type = comboSpawnType->currentIndex(); // 0 = Sphere, 1 = Cube
    int size = spinSize->value();
    float velocity = static_cast<float>(spinVelocity->value());

    Q_EMIT signalSpawnObject(type, velocity, size);
}
