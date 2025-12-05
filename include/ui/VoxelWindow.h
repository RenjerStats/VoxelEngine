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


    void setFOV(float val);
    void setDistance(float val);
    void setLightDirX(float z);
    void setLightDirY(float y);
    void setLightDirZ(float z);
    void setCameraRotationX(float x);
    void setCameraRotationY(float y);
    void setCameraRotationZ(float z);

protected:
    // Переопределение методов QOpenGLWindow
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void loadScene();


private:
    void calculateCenterOfModel();

    // Графические объекты
    QOpenGLShaderProgram program;
    QOpenGLBuffer vbo;
    QOpenGLVertexArrayObject vao;
    QOpenGLTexture* paletteTexture = nullptr;

    QTimer* timer = nullptr;

    // Данные сцены
    std::vector<CudaVoxel> hostCudaVoxels;
    int voxelCount = 0;

    // Путь к файлу сцены
    QString scenePath;

    // --- Параметры рендеринга  ---
    QVector3D sceneCenter = QVector3D(0, 0, 0); // Центр сцены
    float distanceToModel;
    float m_fov;
    QVector3D m_lightDir;
    QVector3D m_cameraRotation;
};
