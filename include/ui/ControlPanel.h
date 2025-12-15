#pragma once

#include <QWidget>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QComboBox>
#include <QLabel>
#include <QGroupBox>

class VoxelWindow; // Forward declaration

class ControlPanel : public QWidget {
    Q_OBJECT

public:
    explicit ControlPanel(VoxelWindow* voxelWindow, QWidget* parent = nullptr);

Q_SIGNALS:
    void signalResetSimulation();
    void signalTogglePause(bool isPaused);

    void signalSpawnObject(int type, float velocity, int size);

private Q_SLOTS:
    void onSpawnClicked();
    void onPauseClicked();

private:
    void setupUI();

    VoxelWindow* m_voxelWindow;

    QPushButton* btnPause;
    QPushButton* btnReset;

    QComboBox* comboSpawnType;
    QSpinBox* spinSize;
    QDoubleSpinBox* spinVelocity;
    QPushButton* btnSpawn;

    bool isPaused = false;
};
