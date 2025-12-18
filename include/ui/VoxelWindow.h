#pragma once

#include "io/ogt_vox.h"
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
class InputHandler;

class VoxelWindow : public QOpenGLWindow, protected QOpenGLFunctions_4_5_Core
{Q_OBJECT

public:
    VoxelWindow(QWindow* parent = nullptr);
    ~VoxelWindow() override;

    void setFOV(float val);
    void setDistance(float val);
    void setHeight(float val);

    void setLightDirX(float x);
    void setLightDirY(float y);
    void setLightDirZ(float z);
    void setCameraRotationX(float x);
    void setCameraRotationY(float y);


public: Q_SIGNALS:
    void spawnRequested();

public Q_SLOTS:
    void setPaused(bool p);
    void spawnSphereFromCamera(float velocityMagnitude, unsigned size);
    void spawnCubeFromCamera(float velocityMagnitude, unsigned size);
    void resetSimulation();
    void loadScene(QString path);
    void saveScene(QString path);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;


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
    ogt_vox_palette voxPalette;

    QVector3D sceneCenter;
    float distanceToModel;
    float cameraHeight;
    float fov;
    QVector3D lightDir;
    QVector3D cameraRotation;

    InputHandler* inputHandler = nullptr;

    bool physicsPaused = false;
    PhysicsManager physicsManager;
    void onPhysicsMemoryResize(unsigned int newMaxVoxels, unsigned int activeVoxels);
};
