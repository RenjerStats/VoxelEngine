#pragma once

#include <QMainWindow>

class GLWidget;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override = default;

private:
    void setupUI();

    GLWidget* m_glWidget;
};
