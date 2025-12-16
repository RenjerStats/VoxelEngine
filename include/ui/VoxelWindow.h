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
    VoxelWindow(QWindow* parent = nullptr);
    ~VoxelWindow() override;


    void setScenePath(const QString& path) { scenePath = path; }


    void setFOV(float val);
    void setDistance(float val);
    void setLightDirX(float z);
    void setLightDirY(float y);
    void setLightDirZ(float z);
    void setCameraRotationX(float x);
    void setCameraRotationY(float y);
    void setCameraRotationZ(float z);

public Q_SLOTS:
    void setPaused(bool p);
    void spawnSphereFromCamera(float velocityMagnitude, unsigned size);
    void spawnCubeFromCamera(float velocityMagnitude, unsigned size);
    void resetSimulation();

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void loadScene();


private:
    void calculateCenterOfModel();
    void initShadowFBO();

    QOpenGLShaderProgram program;
    QOpenGLShaderProgram shadowProgram;

    GLuint depthMapFBO = 0;
    GLuint depthMapTexture = 0;
    const unsigned int SHADOW_WIDTH = 2048, SHADOW_HEIGHT = 2048;
    float lightBoxScale;

    QOpenGLBuffer vbo;
    QOpenGLVertexArrayObject vao;
    QOpenGLTexture* paletteTexture = nullptr;

    QTimer* timer = nullptr;

    std::vector<RenderVoxel> hostCudaVoxels;
    QString scenePath;

    QVector3D sceneCenter;
    float distanceToModel;
    float m_fov;
    QVector3D m_lightDir;
    QVector3D m_cameraRotation;



    PhysicsManager physicsManager;
    void onPhysicsMemoryResize(unsigned int newMaxVoxels, unsigned int activeVoxels);
};
