#pragma once

#include "physics/PhysicsManager.h"

#include <QOpenGLWindow>
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLTexture>
#include <QTimer>
#include <vector>

struct RenderVoxel;

class VoxelWindow : public QOpenGLWindow, protected QOpenGLFunctions_4_5_Core
{Q_OBJECT

public:
    // Конструктор и деструктор
    VoxelWindow(QWindow* parent = nullptr);
    ~VoxelWindow() override;


    // Метод для установки пути к файлу (используется для загрузки)
    void setScenePath(const QString& path) { scenePath = path; }
    void resetSimulation();


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
    void initShadowFBO();

    // Графические объекты
    QOpenGLShaderProgram program;       // Основной шейдер (свет + цвет)
    QOpenGLShaderProgram shadowProgram; // Теневой шейдер (только глубина)

    // Shadow Mapping ресурсы
    GLuint depthMapFBO = 0;
    GLuint depthMapTexture = 0;
    const unsigned int SHADOW_WIDTH = 2048, SHADOW_HEIGHT = 2048; // Разрешение тени
    float lightBoxScale;

    QOpenGLBuffer vbo;
    QOpenGLVertexArrayObject vao;
    QOpenGLTexture* paletteTexture = nullptr;

    QTimer* timer = nullptr;

    // Данные сцены
    std::vector<RenderVoxel> hostCudaVoxels;
    int voxelCount = 0;
    QString scenePath;

    // Параметры
    QVector3D sceneCenter;
    float distanceToModel;
    float m_fov;
    QVector3D m_lightDir;
    QVector3D m_cameraRotation;


    // физика
    PhysicsManager physicsManager;
};
