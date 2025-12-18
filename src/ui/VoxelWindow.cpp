#include "ui/VoxelWindow.h"
#include "io/VoxFileParser.h"
#include "io/VoxScene.h"
#include "core/Types.h"
#include "ui/InputHandler.h"

#include <QDebug>
#include <QtMath>

using namespace VoxIO;
using namespace std;


VoxelWindow::VoxelWindow(QWindow* parent)
    : QOpenGLWindow(NoPartialUpdate, parent),
    fov(45.0f),
    lightDir(0.5f, 1.0f, 0.3f),
    cameraRotation(0.0f, 0.0f, 0.0f),
    distanceToModel(100),
    sceneCenter(0.0f, 0.0f, 0.0f),
    lightBoxScale(100),
    cameraHeight(0)
{
    QSurfaceFormat format;
    format.setVersion(4, 5);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setSamples(4);
    setFormat(format);

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &VoxelWindow::requestUpdate);
    timer->start(16);
}

VoxelWindow::~VoxelWindow() {
    makeCurrent();


    vbo.destroy();
    vao.destroy();
    if (paletteTexture) {
        paletteTexture->destroy();
        delete paletteTexture;
    }
    if (depthMapFBO) glDeleteFramebuffers(1, &depthMapFBO);
    if (depthMapTexture) glDeleteTextures(1, &depthMapTexture);

    physicsManager.freeResources();

    doneCurrent();
}

void VoxelWindow::initializeGL() {
    if (!initializeOpenGLFunctions()) {
        qCritical() << "Failed to initialize OpenGL functions!";
        return;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    if (!program.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/voxel.vert") ||
        !program.addShaderFromSourceFile(QOpenGLShader::Geometry, ":/shaders/voxel.geom") ||
        !program.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/voxel.frag") ||
        !program.link()) {
        qCritical() << "Shader error:" << program.log();
        return;
    }

    if (!shadowProgram.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/voxel.vert") ||
        !shadowProgram.addShaderFromSourceFile(QOpenGLShader::Geometry, ":/shaders/voxel.geom") ||
        !shadowProgram.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/shadow.frag") ||
        !shadowProgram.link()) {
        qCritical() << "Shadow Shader error:" << shadowProgram.log();
    }

    vao.create();
    vbo.create();

    vao.bind();
    vbo.bind();

    int stride = sizeof(RenderVoxel);

    program.enableAttributeArray(0);
    program.setAttributeBuffer(0, GL_FLOAT, 0, 3, stride);

    program.enableAttributeArray(1);
    program.setAttributeBuffer(1, GL_FLOAT, 12, 1, stride);

    vbo.release();
    vao.release();

    initShadowFBO();

    inputHandler = new InputHandler(this);

    loadScene("assets/AncientTemple.vox");

    qDebug() << "System Initialized. GPU:" << (const char*)glGetString(GL_RENDERER);
}

void VoxelWindow::loadScene(QString path){
    qDebug() << "Loading scene from:" << path;
    std::unique_ptr<VoxScene> scene;
    try {
        scene = VoxFileParser::load(path);
    } catch (...) {
        qWarning() << "Failed to load scene!";
    }

    if (!scene) {
        qWarning() << "Failed to load scene!";
        return;
    }

    hostCudaVoxels.clear();

    const ogt_vox_palette& pal = scene->palette();
    voxPalette = pal; // сохраняем для использования в save scene
    QImage palImage(256, 1, QImage::Format_RGBA8888);
    for (int i = 0; i < 256; ++i) {
        palImage.setPixelColor(i, 0, QColor(pal.color[i].r, pal.color[i].g, pal.color[i].b, pal.color[i].a));
    }

    if (scene->modelCount() > 0) {
        VoxModel model = scene->getModel(0);
        hostCudaVoxels = model.getCudaVoxels();
        calculateCenterOfModel();
    } else {
        hostCudaVoxels.clear();
        sceneCenter = QVector3D(0, 0, 0);
    }

    unsigned int voxelCount = hostCudaVoxels.size();
    qDebug() << "Loaded voxels:" << voxelCount;

    unsigned int maxVoxels = voxelCount * 2;
    if (maxVoxels < 1024) maxVoxels = 1024;


    vbo.bind();
    vbo.allocate(maxVoxels * sizeof(RenderVoxel));
    if (!hostCudaVoxels.empty()) {
        vbo.write(0, hostCudaVoxels.data(), voxelCount * sizeof(RenderVoxel));
    }
    vbo.release();

    if (paletteTexture) {
        paletteTexture->destroy();
        delete paletteTexture;
        paletteTexture = nullptr;
    }

    paletteTexture = new QOpenGLTexture(palImage);
    paletteTexture->setMinificationFilter(QOpenGLTexture::Nearest);
    paletteTexture->setMagnificationFilter(QOpenGLTexture::Nearest);
    paletteTexture->setWrapMode(QOpenGLTexture::ClampToEdge);

    physicsManager = PhysicsManager(60, maxVoxels);
    physicsManager.connectToOpenGL(vbo.bufferId(), voxelCount);
    physicsManager.setVoxelCallback([this](unsigned int newMax, unsigned int active) {
        this->onPhysicsMemoryResize(newMax, active);
    });

    requestUpdate();
}

void VoxelWindow::saveScene(QString path){
    unsigned int voxelCount = physicsManager.getActiveVoxels();
    std::vector<RenderVoxel> buffer;
    buffer.clear();
    buffer.resize(voxelCount);
    vbo.bind();
    vbo.read(0, buffer.data(), voxelCount * sizeof(RenderVoxel));
    vbo.release();
    try {
        bool success = VoxFileParser::save(path, buffer, voxPalette);
    } catch (...) {
        qWarning() << "Failed to save scene!";
    }
}

void VoxelWindow::mousePressEvent(QMouseEvent* event) {
    inputHandler->mousePressEvent(event);
}

void VoxelWindow::mouseMoveEvent(QMouseEvent* event) {
    inputHandler->mouseMoveEvent(event);
}

void VoxelWindow::mouseReleaseEvent(QMouseEvent* event) {
    inputHandler->mouseReleaseEvent(event);
}

void VoxelWindow::keyPressEvent(QKeyEvent* event) {
    if (!event) return;

    if (event->key() == Qt::Key_Space) {
        Q_EMIT spawnRequested();
        event->accept();
    } else {
        QOpenGLWindow::keyPressEvent(event);
    }
}

void VoxelWindow::setPaused(bool paused) {
    physicsPaused = paused;
}

void VoxelWindow::initShadowFBO() {
    glGenFramebuffers(1, &depthMapFBO);

    glGenTextures(1, &depthMapTexture);
    glBindTexture(GL_TEXTURE_2D, depthMapTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                 SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMapTexture, 0);

    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void VoxelWindow::onPhysicsMemoryResize(unsigned int newMaxVoxels, unsigned int activeVoxels)
{
    makeCurrent();
    qDebug() << "Resizing VBO to max voxels:" << newMaxVoxels;

    vbo.bind();
    vbo.allocate(newMaxVoxels * sizeof(RenderVoxel));
    vbo.release();

    physicsManager.updateGLResource(vbo.bufferId());

    doneCurrent();
}

void VoxelWindow::calculateCenterOfModel()
{
    if (!hostCudaVoxels.empty()) {
        float minX = std::numeric_limits<float>::max();
        float minY = std::numeric_limits<float>::max();
        float minZ = std::numeric_limits<float>::max();

        float maxX = std::numeric_limits<float>::lowest();
        float maxY = std::numeric_limits<float>::lowest();
        float maxZ = std::numeric_limits<float>::lowest();

        for (const auto& v : hostCudaVoxels) {
            if (v.x < minX) minX = v.x;
            if (v.y < minY) minY = v.y;
            if (v.z < minZ) minZ = v.z;

            if (v.x > maxX) maxX = v.x;
            if (v.y > maxY) maxY = v.y;
            if (v.z > maxZ) maxZ = v.z;
        }

        sceneCenter = QVector3D(
            (minX + maxX) / 2.0f,
            (minY + maxY) / 2.0f,
            (minZ + maxZ) / 2.0f
            );

        float sizeX = maxX - minX;
        float sizeY = maxY - minY;

        float maxDim = std::max({sizeX, sizeY});
        distanceToModel = maxDim * 1.2f;
        lightBoxScale = distanceToModel;

        qDebug() << "Scene Center Calculated:" << sceneCenter;
    }
}

void VoxelWindow::resetSimulation(){
    if (hostCudaVoxels.empty()) return;

    qDebug() << "Resetting simulation...";

    physicsManager.freeResources();

    vbo.bind();
    vbo.write(0, hostCudaVoxels.data(), hostCudaVoxels.size() * sizeof(RenderVoxel));
    vbo.release();

    unsigned int voxelCount = hostCudaVoxels.size();
    unsigned int maxVoxels = voxelCount * 2;
    if (maxVoxels < 1024) maxVoxels = 1024;

    physicsManager = PhysicsManager(60, maxVoxels);
    physicsManager.connectToOpenGL(vbo.bufferId(), voxelCount);

    requestUpdate();
}


void VoxelWindow::resizeGL(int w, int h) {}

void VoxelWindow::setFOV(float val) { fov = val; }
void VoxelWindow::setDistance(float val) { distanceToModel = val; }
void VoxelWindow::setHeight(float val) { cameraHeight = val; }
void VoxelWindow::setLightDirX(float x) { lightDir.setX(x); }
void VoxelWindow::setLightDirY(float y) { lightDir.setY(y); }
void VoxelWindow::setLightDirZ(float z) { lightDir.setZ(-z); }
void VoxelWindow::setCameraRotationX(float x) { cameraRotation.setX(x); }
void VoxelWindow::setCameraRotationY(float y) { cameraRotation.setY(y); }

void VoxelWindow::paintGL() {
    if (!physicsPaused) physicsManager.updatePhysics(1, 1);

    unsigned int voxelCount = physicsManager.getActiveVoxels();
    if (voxelCount == 0) {
        glClearColor(0.1f, 0.15f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return;
    }


    float orthoSize = lightBoxScale;
    QMatrix4x4 lightProjection;
    lightProjection.ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 1.0f, lightBoxScale * 3.0f);


    QVector3D lightPos = sceneCenter + (lightDir.normalized() * distanceToModel * 1.5f);

    QMatrix4x4 lightView;
    lightView.lookAt(lightPos, sceneCenter, QVector3D(0, 1, 0));

    QMatrix4x4 lightSpaceMatrix = lightProjection * lightView;

    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glClear(GL_DEPTH_BUFFER_BIT);

    shadowProgram.bind();
    shadowProgram.setUniformValue("proj", lightProjection);
    shadowProgram.setUniformValue("view", lightView);
    shadowProgram.setUniformValue("voxelSize", 1.0f);

    vao.bind();
    glDrawArrays(GL_POINTS, 0, voxelCount);
    vao.release();
    shadowProgram.release();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    glViewport(0, 0, width() * devicePixelRatio(), height() * devicePixelRatio());
    glClearColor(0.1f, 0.15f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    program.bind();
    vao.bind();

    QMatrix4x4 view, proj;
    float nearPlane = 0.1f;
    float farPlane = 5000.0f;
    proj.perspective(fov, (float)width() / height(), nearPlane, farPlane);

    float yaw = qDegreesToRadians(cameraRotation.y());
    float pitch = qDegreesToRadians(cameraRotation.x());
    float camX = distanceToModel * cos(pitch) * sin(yaw);
    float camY = distanceToModel * sin(pitch);
    float camZ = distanceToModel * cos(pitch) * cos(yaw);

    QVector3D thiftedCenter = sceneCenter + QVector3D(0, cameraHeight, 0);
    QVector3D orbitCamPos = thiftedCenter + QVector3D(camX, camY, camZ);

    view.lookAt(orbitCamPos, thiftedCenter, QVector3D(0, 1, 0));

    program.setUniformValue("proj", proj);
    program.setUniformValue("view", view);
    program.setUniformValue("voxelSize", 1.0f);
    program.setUniformValue("lightDir", lightDir);
    program.setUniformValue("viewPos", orbitCamPos);
    program.setUniformValue("shininess", 32.0f);

    program.setUniformValue("lightSpaceMatrix", lightSpaceMatrix);

    if (paletteTexture) {
        glActiveTexture(GL_TEXTURE0);
        paletteTexture->bind();
        program.setUniformValue("uPalette", 0);
    }

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depthMapTexture);
    program.setUniformValue("shadowMap", 1);

    glDrawArrays(GL_POINTS, 0, voxelCount);

    if (paletteTexture) paletteTexture->release();
    glBindTexture(GL_TEXTURE_2D, 0);
    vao.release();
    program.release();
}

void VoxelWindow::spawnSphereFromCamera(float velocityMagnitude, unsigned size) {
    float yaw = qDegreesToRadians(cameraRotation.y());
    float pitch = qDegreesToRadians(cameraRotation.x());

    float camX = distanceToModel * cos(pitch) * sin(yaw);
    float camY = distanceToModel * sin(pitch);
    float camZ = distanceToModel * cos(pitch) * cos(yaw);

    QVector3D thiftedCenter = sceneCenter + QVector3D(0, cameraHeight, 0);
    QVector3D cameraPosition = thiftedCenter + QVector3D(camX, camY, camZ);

    QVector3D direction = (thiftedCenter - cameraPosition).normalized();


    float vx = direction.x() * velocityMagnitude;
    float vy = direction.y() * velocityMagnitude;
    float vz = direction.z() * velocityMagnitude;

    int randomColor = 1 + (rand() % 255);

    QVector3D spawnPos = cameraPosition + direction * (distanceToModel * 0.2f);
    physicsManager.spawnSphere(spawnPos.x(), spawnPos.y(), spawnPos.z(), size, vx, vy, vz, randomColor);
}

void VoxelWindow::spawnCubeFromCamera(float velocityMagnitude, unsigned size) {
    float yaw = qDegreesToRadians(cameraRotation.y());
    float pitch = qDegreesToRadians(cameraRotation.x());

    float camX = distanceToModel * cos(pitch) * sin(yaw);
    float camY = distanceToModel * sin(pitch);
    float camZ = distanceToModel * cos(pitch) * cos(yaw);


    QVector3D thiftedCenter = sceneCenter + QVector3D(0, cameraHeight, 0);
    QVector3D cameraPosition = thiftedCenter + QVector3D(camX, camY, camZ);

    QVector3D direction = (thiftedCenter - cameraPosition).normalized();

    float vx = direction.x() * velocityMagnitude;
    float vy = direction.y() * velocityMagnitude;
    float vz = direction.z() * velocityMagnitude;

    int randomColor = 1 + (rand() % 255);

    QVector3D spawnPos = cameraPosition + direction * (distanceToModel * 0.2f);
    physicsManager.spawnCube(spawnPos.x(), spawnPos.y(), spawnPos.z(), size, vx, vy, vz, randomColor);
}
