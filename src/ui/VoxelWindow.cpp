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
    , sceneCenter(0.0f, 0.0f, 0.0f)
    , lightBoxScale(100)
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
    timer->start(16); // Обновляем сцену каждые 16мс (~60 FPS)
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

    // --- 1. Шейдеры цвета и базового освещения ---
    if (!program.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/voxel.vert") ||
        !program.addShaderFromSourceFile(QOpenGLShader::Geometry, ":/shaders/voxel.geom") ||
        !program.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/voxel.frag") ||
        !program.link()) {
        qCritical() << "Shader error:" << program.log();
        return;
    }

        // --- 1. Шейдеры теней ---
    if (!shadowProgram.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/voxel.vert") ||
        !shadowProgram.addShaderFromSourceFile(QOpenGLShader::Geometry, ":/shaders/voxel.geom") ||
        !shadowProgram.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/shadow.frag") ||
        !shadowProgram.link()) {
        qCritical() << "Shadow Shader error:" << shadowProgram.log();
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

    initShadowFBO();

    // --- 3. Загрузка сцены ---
    if (!scenePath.isEmpty()) {
        loadScene();
        scenePath.clear();
    }

    qDebug() << "VoxelWindow Initialized. GPU:" << (const char*)glGetString(GL_RENDERER);
}

void VoxelWindow::initShadowFBO() {
    // 1. Создаем Framebuffer
    glGenFramebuffers(1, &depthMapFBO);

    // 2. Создаем текстуру глубины
    glGenTextures(1, &depthMapTexture);
    glBindTexture(GL_TEXTURE_2D, depthMapTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                 SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

    // Параметры фильтрации (важно для теней)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Clamp to Border (чтобы за пределами карты теней не было теней)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    // 3. Прикрепляем текстуру к FBO
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMapTexture, 0);

    // Указываем, что нам не нужны цвета (только глубина)
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

    physicsManager = PhysicsManager(60, voxelCount);
    physicsManager.registerVoxelSharedBuffer(vbo.bufferId());
    physicsManager.initSumulation();


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
        lightBoxScale = distanceToModel;

        qDebug() << "Scene Center Calculated:" << sceneCenter;
    }
}

void VoxelWindow::resetSimulation(){
    if (hostCudaVoxels.empty()) return;

    qDebug() << "Resetting simulation...";

    physicsManager.freeResources();

    vbo.bind();
    vbo.write(0, hostCudaVoxels.data(), hostCudaVoxels.size() * sizeof(CudaVoxel));
    vbo.release();

    physicsManager.registerVoxelSharedBuffer(vbo.bufferId());

    requestUpdate();
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
    if (voxelCount == 0) {
        // Просто чистим экран если пусто
        glClearColor(0.1f, 0.15f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        return;
    }

    physicsManager.updatePhysics(1, 0.99);

    // ===========================
    // ШАГ 0: Расчет матриц света
    // ===========================
    // Создаем ортогональную матрицу, охватывающую сцену
    // Размер коробки зависит от distanceToModel (размера модели)
    float orthoSize = lightBoxScale; // Подберите коэффициент по вкусу
    QMatrix4x4 lightProjection;
    lightProjection.ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 1.0f, lightBoxScale * 3.0f);

    // "Виртуальная" позиция солнца. Сдвигаем от центра сцены навстречу вектору света
    // m_lightDir у вас (0.5, 1.0, 0.3) -> свет бьет В ЭТУ сторону.
    // Значит источник находится в противоположной стороне (sceneCenter - m_lightDir * dist)
    // НО в voxel.frag вы используете lightDir как направление НА свет?
    // Обычно lightDir - это направление ОТ источника. Если у вас Phong, проверьте это.
    // Допустим m_lightDir - это вектор падения света.
    // Позиция камеры света:
    QVector3D lightPos = sceneCenter + (m_lightDir.normalized() * distanceToModel * 1.5f);

    QMatrix4x4 lightView;
    lightView.lookAt(lightPos, sceneCenter, QVector3D(0, 1, 0));

    QMatrix4x4 lightSpaceMatrix = lightProjection * lightView;


    // ===========================
    // ШАГ 1: Shadow Pass (Рендер в текстуру)
    // ===========================
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glClear(GL_DEPTH_BUFFER_BIT);

    shadowProgram.bind();
    shadowProgram.setUniformValue("proj", lightProjection);
    shadowProgram.setUniformValue("view", lightView);
    shadowProgram.setUniformValue("voxelSize", 1.0f); // Размер вокселя

    vao.bind();
    // Voxel.geom используется и здесь! Он сгенерирует кубы с точки зрения света.
    // Поскольку shadow.frag пустой, запишется только глубина.
    glDrawArrays(GL_POINTS, 0, voxelCount);
    vao.release();
    shadowProgram.release();

    glBindFramebuffer(GL_FRAMEBUFFER, 0); // Вернулись к экрану


    // ===========================
    // ШАГ 2: Render Pass (Обычный рендер)
    // ===========================
    glViewport(0, 0, width() * devicePixelRatio(), height() * devicePixelRatio()); // Вернули вьюпорт
    glClearColor(0.1f, 0.15f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    program.bind();
    vao.bind();

    // Матрицы камеры (как было раньше)
    QMatrix4x4 view, proj;
    float nearPlane = 0.1f;
    float farPlane = 5000.0f;
    proj.perspective(m_fov, (float)width() / height(), nearPlane, farPlane);

    float camX = sceneCenter.x() + distanceToModel;
    float camZ = sceneCenter.z() + distanceToModel;
    float camY = sceneCenter.y() + (distanceToModel * 0.3f);
    QVector3D orbitCamPos(camX, camY, camZ);

    view.lookAt(orbitCamPos, sceneCenter, QVector3D(0, 1, 0));
    // Применяем вращение мыши
    view.translate(sceneCenter);
    view.rotate(m_cameraRotation.y(), 0, 1, 0);
    view.rotate(m_cameraRotation.x(), 1, 0, 0);
    view.rotate(m_cameraRotation.z(), 0, 0, 1);
    view.translate(-sceneCenter);

    // Передаем юниформы
    program.setUniformValue("proj", proj);
    program.setUniformValue("view", view);
    program.setUniformValue("voxelSize", 1.0f);
    program.setUniformValue("lightDir", m_lightDir); // Вектор НА свет
    program.setUniformValue("viewPos", orbitCamPos); // Для бликов (приблизительно)
    program.setUniformValue("shininess", 32.0f);

    // ---> ВАЖНО: Передаем матрицу света для расчета проекции тени
    program.setUniformValue("lightSpaceMatrix", lightSpaceMatrix);

    // Текстура палитры (Unit 0)
    if (paletteTexture) {
        glActiveTexture(GL_TEXTURE0);
        paletteTexture->bind();
        program.setUniformValue("uPalette", 0);
    }

    // Текстура тени (Unit 1)
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depthMapTexture);
    program.setUniformValue("shadowMap", 1);

    glDrawArrays(GL_POINTS, 0, voxelCount);

    if (paletteTexture) paletteTexture->release();
    glBindTexture(GL_TEXTURE_2D, 0); // Unbind shadow map
    vao.release();
    program.release();
}
