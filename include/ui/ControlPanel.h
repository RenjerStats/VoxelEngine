#pragma once

#include <QWidget>
#include <QPushButton>
#include <QComboBox>
#include <QSlider>
#include <QLabel>
#include <QGroupBox>
#include <QVBoxLayout>

class VoxelWindow;

class ControlPanel : public QWidget {
    Q_OBJECT

public:
    explicit ControlPanel(VoxelWindow* voxelWindow, QWidget* parent = nullptr);

Q_SIGNALS:
    void signalResetSimulation();
    void signalTogglePause(bool isPaused);
    void signalDownloadScene(QString path);
    void signalSaveScene(QString path);
    void signalSpawnObject(int type, float velocity, int size);
    void signalFOVChanged(float fov);
    void signalDistanceChanged(float distance);
    void signalHeightChanged(float height);

private Q_SLOTS:
    void onSpawnClicked();
    void onPauseClicked();
    void onDownloadClicked();
    void onSaveClicked();
    void onSizeSliderChanged(int value);
    void onVelocitySliderChanged(int value);
    void onFOVSliderChanged(int value);
    void onDistanceSliderChanged(int value);
    void onHeightSliderChanged(int value);

private:
    void setupUI();
    QWidget* createSliderControl(const QString& label, int min, int max, int defaultValue,
                                 QSlider** slider, QLabel** valueLabel, const QString& suffix = "");

    VoxelWindow* m_voxelWindow;

    QPushButton* btnPause;
    QPushButton* btnReset;

    QPushButton* btnDownload;
    QPushButton* btnSave;

    QComboBox* comboSpawnType;
    QSlider* sliderSize;
    QLabel* lblSizeValue;
    QSlider* sliderVelocity;
    QLabel* lblVelocityValue;
    QPushButton* btnSpawn;

    QSlider* sliderFOV;
    QLabel* lblFOVValue;
    QSlider* sliderDistance;
    QLabel* lblDistanceValue;
    QSlider* sliderHeight;
    QLabel* lblHeightValue;

    bool isPaused = false;
};
