#include "ui/ControlPanel.h"
#include "ui/VoxelWindow.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QStandardPaths>

ControlPanel::ControlPanel(VoxelWindow* voxelWindow, QWidget* parent)
    : QWidget(parent), m_voxelWindow(voxelWindow)
{
    setupUI();

    connect(this, &ControlPanel::signalTogglePause, [this](bool paused) {
        m_voxelWindow->setPaused(paused);
    });
    connect(this, &ControlPanel::signalDownloadScene, [this](QString path) {
        m_voxelWindow->loadScene(path);
    });
    connect(this, &ControlPanel::signalSaveScene, [this](QString path) {
        m_voxelWindow->saveScene(path);
    });


    connect(this, &ControlPanel::signalResetSimulation, m_voxelWindow, &VoxelWindow::resetSimulation);

    connect(this, &ControlPanel::signalSpawnObject, [this](int type, float velocity, int size) {
        if (type == 0) {
            m_voxelWindow->spawnSphereFromCamera(velocity, size);
        } else {
            m_voxelWindow->spawnCubeFromCamera(velocity, size);
        }
    });

    connect(m_voxelWindow, &VoxelWindow::spawnRequested, this, &ControlPanel::onSpawnClicked);

    connect(this, &ControlPanel::signalFOVChanged, m_voxelWindow, &VoxelWindow::setFOV);

    connect(this, &ControlPanel::signalDistanceChanged,m_voxelWindow, &VoxelWindow::setDistance);

    connect(this, &ControlPanel::signalHeightChanged,m_voxelWindow, &VoxelWindow::setHeight);
}

void ControlPanel::setupUI() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(10, 10, 10, 10);
    mainLayout->setSpacing(15);

    QGroupBox* groupFiles = new QGroupBox("Управление файлами", this);
    QVBoxLayout* filesLayout = new QVBoxLayout(groupFiles);
    filesLayout->setSpacing(8);

    btnDownload = new QPushButton("📁 Загрузить сцену", this);
    btnDownload->setMinimumHeight(35);
    btnDownload->setStyleSheet(
        "QPushButton { "
        "    background-color: #2196F3; "
        "    color: white; "
        "    font-weight: bold; "
        "    border-radius: 4px; "
        "    padding: 8px; "
        "}"
        "QPushButton:hover { background-color: #1976D2; }"
        "QPushButton:pressed { background-color: #1565C0; }"
        );

    btnSave = new QPushButton("💾 Сохранить сцену", this);
    btnSave->setMinimumHeight(35);
    btnSave->setStyleSheet(
        "QPushButton { "
        "    background-color: #FF9800; "
        "    color: white; "
        "    font-weight: bold; "
        "    border-radius: 4px; "
        "    padding: 8px; "
        "}"
        "QPushButton:hover { background-color: #F57C00; }"
        "QPushButton:pressed { background-color: #E65100; }"
        );

    filesLayout->addWidget(btnDownload);
    filesLayout->addWidget(btnSave);

    mainLayout->addWidget(groupFiles);

    QGroupBox* groupSim = new QGroupBox("Управление симуляцией", this);
    QVBoxLayout* simLayout = new QVBoxLayout(groupSim);
    simLayout->setSpacing(8);

    btnPause = new QPushButton("Пауза", this);
    btnPause->setCheckable(true);
    btnPause->setMinimumHeight(35);

    btnReset = new QPushButton("Сбросить сцену", this);
    btnReset->setMinimumHeight(35);

    simLayout->addWidget(btnPause);
    simLayout->addWidget(btnReset);

    mainLayout->addWidget(groupSim);

    QGroupBox* groupCamera = new QGroupBox("Параметры камеры", this);
    QVBoxLayout* cameraLayout = new QVBoxLayout(groupCamera);
    cameraLayout->setSpacing(12);

    cameraLayout->addWidget(createSliderControl("Угол обзора (FOV):", 30, 120, 45,
                                                &sliderFOV, &lblFOVValue, "°"));

    cameraLayout->addWidget(createSliderControl("Дистанция:", 50, 500, 200,
                                                &sliderDistance, &lblDistanceValue));

    cameraLayout->addWidget(createSliderControl("Высота:", -200, 300, 0,
                                                &sliderHeight, &lblHeightValue));

    mainLayout->addWidget(groupCamera);

    QGroupBox* groupSpawn = new QGroupBox("Создание объектов", this);
    QVBoxLayout* spawnLayout = new QVBoxLayout(groupSpawn);
    spawnLayout->setSpacing(12);

    QHBoxLayout* typeLayout = new QHBoxLayout();
    QLabel* lblType = new QLabel("Тип:", this);
    lblType->setMinimumWidth(80);
    comboSpawnType = new QComboBox(this);
    comboSpawnType->addItem("Сфера");
    comboSpawnType->addItem("Куб");
    typeLayout->addWidget(lblType);
    typeLayout->addWidget(comboSpawnType, 1);
    spawnLayout->addLayout(typeLayout);

    spawnLayout->addWidget(createSliderControl("Размер:", 3, 40, 10,
                                               &sliderSize, &lblSizeValue, " vox"));

    spawnLayout->addWidget(createSliderControl("Скорость:", 30, 300, 100,
                                               &sliderVelocity, &lblVelocityValue));

    btnSpawn = new QPushButton("Выстрелить объектом", this);
    btnSpawn->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; "
                            "font-weight: bold; border-radius: 4px; padding: 8px; }"
                            "QPushButton:hover { background-color: #45a049; }"
                            "QPushButton:pressed { background-color: #3d8b40; }");
    btnSpawn->setMinimumHeight(40);
    spawnLayout->addWidget(btnSpawn);

    mainLayout->addWidget(groupSpawn);

    mainLayout->addStretch();

    connect(btnDownload, &QPushButton::clicked, this, &ControlPanel::onDownloadClicked);
    connect(btnSave, &QPushButton::clicked, this, &ControlPanel::onSaveClicked);
    connect(btnPause, &QPushButton::clicked, this, &ControlPanel::onPauseClicked);
    connect(btnReset, &QPushButton::clicked, this, &ControlPanel::signalResetSimulation);
    connect(btnSpawn, &QPushButton::clicked, this, &ControlPanel::onSpawnClicked);

    connect(sliderSize, &QSlider::valueChanged, this, &ControlPanel::onSizeSliderChanged);
    connect(sliderVelocity, &QSlider::valueChanged, this, &ControlPanel::onVelocitySliderChanged);
    connect(sliderFOV, &QSlider::valueChanged, this, &ControlPanel::onFOVSliderChanged);
    connect(sliderDistance, &QSlider::valueChanged, this, &ControlPanel::onDistanceSliderChanged);
    connect(sliderHeight, &QSlider::valueChanged, this, &ControlPanel::onHeightSliderChanged);
}

QWidget* ControlPanel::createSliderControl(const QString& label, int min, int max,
                                           int defaultValue, QSlider** slider,
                                           QLabel** valueLabel, const QString& suffix) {
    QWidget* widget = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(widget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);

    QHBoxLayout* topLayout = new QHBoxLayout();
    QLabel* lblName = new QLabel(label, this);
    lblName->setStyleSheet("font-weight: bold;");

    *valueLabel = new QLabel(QString::number(defaultValue) + suffix, this);
    (*valueLabel)->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    (*valueLabel)->setMinimumWidth(60);
    (*valueLabel)->setStyleSheet("color: #2196F3; font-weight: bold;");

    topLayout->addWidget(lblName);
    topLayout->addStretch();
    topLayout->addWidget(*valueLabel);

    *slider = new QSlider(Qt::Horizontal, this);
    (*slider)->setRange(min, max);
    (*slider)->setValue(defaultValue);
    (*slider)->setMinimumHeight(25);

    layout->addLayout(topLayout);
    layout->addWidget(*slider);

    return widget;
}

void ControlPanel::onDownloadClicked() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Загрузить воксельную сцену",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
        "MagicaVoxel Files (*.vox);;All Files (*)"
        );

    if (!fileName.isEmpty()) {
        Q_EMIT signalDownloadScene(fileName);
    }
}

void ControlPanel::onSaveClicked() {
    QString fileName = QFileDialog::getSaveFileName(
        this,
        "Сохранить воксельную сцену",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) + "/scene.vox",
        "MagicaVoxel Files (*.vox);;All Files (*)"
        );

    if (!fileName.isEmpty()) {
        if (!fileName.endsWith(".vox", Qt::CaseInsensitive)) {
            fileName += ".vox";
        }
        Q_EMIT signalSaveScene(fileName);
    }
}



void ControlPanel::onPauseClicked() {
    isPaused = btnPause->isChecked();
    btnPause->setText(isPaused ? "Продолжить" : "Пауза");
    Q_EMIT signalTogglePause(isPaused);
}

void ControlPanel::onSpawnClicked() {
    int type = comboSpawnType->currentIndex(); // 0 = Сфера, 1 = Куб
    int size = sliderSize->value();
    float velocity = static_cast<float>(sliderVelocity->value());
    Q_EMIT signalSpawnObject(type, velocity, size);
}

void ControlPanel::onSizeSliderChanged(int value) {
    lblSizeValue->setText(QString::number(value) + " vox");
}

void ControlPanel::onVelocitySliderChanged(int value) {
    lblVelocityValue->setText(QString::number(value));
}

void ControlPanel::onFOVSliderChanged(int value) {
    lblFOVValue->setText(QString::number(value) + "°");
    Q_EMIT signalFOVChanged(static_cast<float>(value));
}

void ControlPanel::onDistanceSliderChanged(int value) {
    lblDistanceValue->setText(QString::number(value));
    Q_EMIT signalDistanceChanged(static_cast<float>(value));
}

void ControlPanel::onHeightSliderChanged(int value) {
    lblHeightValue->setText(QString::number(value));
    Q_EMIT signalHeightChanged(static_cast<float>(value));
}
