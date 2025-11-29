#pragma once

#include <QMainWindow>

class VoxelWindow;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override = default;

private:
    void setupUI();

    VoxelWindow* voxelWindow;
};
