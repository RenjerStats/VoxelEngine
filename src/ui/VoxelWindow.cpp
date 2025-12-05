#include "ui/VoxelWindow.h"
#include "io/VoxFileParser.h"
#include "io/VoxScene.h"
#include "core/Types.h"

#include <QDebug>
#include <QtMath>

using namespace VoxIO;
using namespace std;

VoxelWindow::VoxelWindow(QWindow* parent)
    : QOpenGLWindow(NoPartialUpdate, parent)
    , m_fov(45.0f)
    , m_lightDir(0.5f, 1.0f, 0.3f)
    , m_cameraRotation(180, -45, 180)
    , distanceToModel(100)
{
    // Настройка формата OpenGL
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

    // Очистка OpenGL
    vbo.destroy();
    vao.destroy();
    if (paletteTexture) {
        paletteTexture->destroy(); // Освобождаем GL ресурс
        delete paletteTexture;     // Освобождаем память CPU
    }

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
    glClearColor(0.1f, 0.15f, 0.2f, 1.0f);

    // --- 1. Шейдеры ---
    if (!program.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/voxel.vert") ||
        !program.addShaderFromSourceFile(QOpenGLShader::Geometry, ":/shaders/voxel.geom") ||
        !program.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/voxel.frag") ||
        !program.link()) {
        qCritical() << "Shader error:" << program.log();
        return;
    }

    // --- 2. Инициализация буферов (ОБЯЗАТЕЛЬНО ЗДЕСЬ) ---
    vao.create();
    vbo.create();

    vao.bind();
    vbo.bind();

    // Настраиваем layout один раз. Данные зальем позже, но формат опишем сейчас.
    int stride = sizeof(CudaVoxel);

    // Attribute 0: Position (offset 0)
    program.enableAttributeArray(0);
    program.setAttributeBuffer(0, GL_FLOAT, 0, 3, stride);

    // Attribute 1: ColorID (offset 36)
    program.enableAttributeArray(1);
    program.setAttributeBuffer(1, GL_FLOAT, 36, 1, stride);

    vbo.release();
    vao.release();

    // --- 3. Загрузка сцены ---
    if (!scenePath.isEmpty()) {
        loadScene();
        scenePath.clear();
    }

    qDebug() << "VoxelWindow Initialized. GPU:" << (const char*)glGetString(GL_RENDERER);
}

void VoxelWindow::loadScene() {
    qDebug() << "Loading scene from:" << scenePath;

    // 1. Парсинг .vox
    auto scene = VoxFileParser::load(scenePath);
    if (!scene) {
        qWarning() << "Failed to load scene!";
        return;
    }

    hostCudaVoxels.clear();

    // Подготовка палитры (CPU)
    const ogt_vox_palette& pal = scene->palette();
    QImage palImage(256, 1, QImage::Format_RGBA8888);
    for (int i = 0; i < 256; ++i) {
        palImage.setPixelColor(i, 0, QColor(pal.color[i].r, pal.color[i].g, pal.color[i].b, pal.color[i].a));
    }

    // 2. Загрузка вокселей
    if (scene->modelCount() > 0) {
        VoxModel model = scene->getModel(0);
        hostCudaVoxels = model.getCudaVoxels();
        calculateCenterOfModel();
    } else {
        hostCudaVoxels.clear();
        sceneCenter = QVector3D(0, 0, 0);
    }

    voxelCount = hostCudaVoxels.size();
    qDebug() << "Loaded voxels:" << voxelCount;

    // Загружаем данные в УЖЕ СОЗДАННЫЙ VBO
    vbo.bind();
    vbo.allocate(hostCudaVoxels.data(), voxelCount * sizeof(CudaVoxel));
    vbo.release();

    // Палитра: Очищаем старую
    if (paletteTexture) {
        paletteTexture->destroy(); // Освобождаем GL-ресурсы
        delete paletteTexture;
        paletteTexture = nullptr;
    }

    // Создаём новую текстуру АВТОМАТИЧЕСКИ из QImage
    paletteTexture = new QOpenGLTexture(palImage);

    // Настройки (ПОСЛЕ создания)
    paletteTexture->setMinificationFilter(QOpenGLTexture::Nearest);
    paletteTexture->setMagnificationFilter(QOpenGLTexture::Nearest);
    paletteTexture->setWrapMode(QOpenGLTexture::ClampToEdge);

    // Запускаем перерисовку
    requestUpdate();
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

        // Центр = середина между мин. и макс. границами
        sceneCenter = QVector3D(
            (minX + maxX) / 2.0f,
            (minY + maxY) / 2.0f,
            (minZ + maxZ) / 2.0f
            );

        // 2. Вычисляем размер модели (диагональ bounding box)
        float sizeX = maxX - minX;
        float sizeY = maxY - minY;

        // Берем максимальную сторону и умножаем на 1.5, чтобы модель влезла в экран
        float maxDim = std::max({sizeX, sizeY});
        distanceToModel = maxDim * 1.2f;

        qDebug() << "Scene Center Calculated:" << sceneCenter;
    }
}


void VoxelWindow::resizeGL(int w, int h) {
    // Обновляем матрицу проекции в шейдере, если нужно
}

void VoxelWindow::setFOV(float val) { m_fov = val; } // update вызывается таймером, но можно добавить update()
void VoxelWindow::setDistance(float val) { distanceToModel = val; }
void VoxelWindow::setLightDirX(float x) { m_lightDir.setX(x); }
void VoxelWindow::setLightDirY(float y) { m_lightDir.setY(y); }
void VoxelWindow::setLightDirZ(float z) { m_lightDir.setZ(z); }
void VoxelWindow::setCameraRotationX(float x) { m_cameraRotation.setX(x); }
void VoxelWindow::setCameraRotationY(float y) { m_cameraRotation.setY(y); }
void VoxelWindow::setCameraRotationZ(float z) { m_cameraRotation.setZ(z); }

void VoxelWindow::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (voxelCount == 0) return;

    program.bind();
    vao.bind();

    QMatrix4x4 view, proj;
    view.setToIdentity();
    float nearPlane = 0.1f;
    float farPlane = 5000.0f; // Запас дальности

    proj.perspective(m_fov, (float)width() / height(), nearPlane, farPlane);
    program.setUniformValue("proj", proj);

    // --- РАСЧЕТ ОРБИТЫ ---
    // Вращаемся вокруг Y. Используем полярные координаты.
    float camX = sceneCenter.x() +  distanceToModel;
    float camZ = sceneCenter.z() + distanceToModel;
    // Поднимаем камеру чуть выше центра (на половину дистанции)
    float camY = sceneCenter.y() + (distanceToModel * 0.3f);

    QVector3D orbitCamPos(camX, camY, camZ);

    view.lookAt(
        orbitCamPos,    // Откуда (летаем)
        sceneCenter,    // Куда (центр модели)
        QVector3D(0, 1, 0)
        );

    view.translate(sceneCenter);
    view.rotate(m_cameraRotation.y(), 0, 1, 0); // Вращение относительно y
    view.rotate(m_cameraRotation.x(), 1, 0, 0); // Вращение относительно x
    view.rotate(m_cameraRotation.z(), 0, 0, 1); // Вращение относительно z
    view.translate(-sceneCenter);

    program.setUniformValue("view", view);

    // Размер точки
    program.setUniformValue("voxelSize", 1.0f);

    // Свет
    program.setUniformValue("lightDir", m_lightDir);
    program.setUniformValue("shininess", 32.0f);

    // ВАЖНО: Позиция глаза для бликов должна совпадать с реальной камерой
    program.setUniformValue("viewPos", orbitCamPos);

    if (paletteTexture) {
        glActiveTexture(GL_TEXTURE0);
        paletteTexture->bind();
        program.setUniformValue("uPalette", 0);
    }

    glDrawArrays(GL_POINTS, 0, voxelCount);

    if (paletteTexture) paletteTexture->release();
    vao.release();
    program.release();
}
