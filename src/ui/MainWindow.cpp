#include "ui/MainWindow.h"
#include "ui/GLWidget.h"
#include <QHBoxLayout>
#include <QLabel>
#include <QStatusBar>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("Voxel Physics Engine - OpenGL Test");
    resize(1280, 720);
    setupUI();
}

void MainWindow::setupUI() {
    // Центральный виджет
    QWidget* centralWidget = new QWidget(this);
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);

    // OpenGL виджет занимает всё пространство
    m_glWidget = new GLWidget(this);
    m_glWidget->setMinimumSize(800, 600);
    mainLayout->addWidget(m_glWidget);

    setCentralWidget(centralWidget);

    // Статус бар с подсказкой
    statusBar()->showMessage("OpenGL Widget Initialized");
}
