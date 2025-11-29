#pragma once

#include <QOpenGLWindow>
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLTexture>
#include <QTimer>
#include <vector>

struct CudaVoxel;

class VoxelWindow : public QOpenGLWindow, protected QOpenGLFunctions_4_5_Core
{Q_OBJECT

public:
    // Конструктор и деструктор
    VoxelWindow(QWindow* parent = nullptr);
    ~VoxelWindow() override;


    // Метод для установки пути к файлу (используется для загрузки)
    void setScenePath(const QString& path) { scenePath = path; }

protected:
    // Переопределение методов QOpenGLWindow
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void loadScene();


private:
    // Графические объекты
    QOpenGLShaderProgram program;
    QOpenGLBuffer vbo;
    QOpenGLVertexArrayObject vao;
    QOpenGLTexture* paletteTexture = nullptr;

    // Управление временем и анимацией
    QTimer* timer = nullptr;
    float time;

    // Данные сцены
    std::vector<CudaVoxel> hostCudaVoxels;
    int voxelCount = 0;

    // Путь к файлу сцены
    QString scenePath;
};
