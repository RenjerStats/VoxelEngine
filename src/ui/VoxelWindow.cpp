#include "ui/VoxelWindow.h"
#include "io/VoxFileParser.h"
#include "io/VoxScene.h"
#include "core/Types.h"

#include <QDebug>

using namespace VoxIO;
using namespace std;

VoxelWindow::VoxelWindow(QWindow* parent)
    : QOpenGLWindow(NoPartialUpdate, parent)
    , time(0.0f)
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
    } else {
        hostCudaVoxels.clear();
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
    paletteTexture = new QOpenGLTexture(palImage); // Всё в одном вызове!

    // Настройки (ПОСЛЕ создания)
    paletteTexture->setMinificationFilter(QOpenGLTexture::Nearest);
    paletteTexture->setMagnificationFilter(QOpenGLTexture::Nearest);
    paletteTexture->setWrapMode(QOpenGLTexture::ClampToEdge);

    // Запускаем перерисовку
    requestUpdate();
}

void VoxelWindow::resizeGL(int w, int h) {
    // Обновляем матрицу проекции в шейдере, если нужно
}

void VoxelWindow::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (voxelCount == 0) return;

    program.bind();
    vao.bind();

    // Матрицы камеры
    QMatrix4x4 view, proj;
    proj.perspective(45.0f, (float)width() / height(), 0.1f, 1000.0f);
    view.lookAt(
        QVector3D(30, 30, 30),  // Позиция камеры
        QVector3D(0, 0, 0),     // Смотрим на центр
        QVector3D(0, 1, 0)      // Вектор вверх
        );
    view.rotate(time * 10.0f, 0, 1, 0); // Медленное вращение

    program.setUniformValue("view", view);
    program.setUniformValue("proj", proj);
    program.setUniformValue("voxelSize", 1.0f); // Размер воксела

    QVector3D lightDir = QVector3D(0.5f, 1.0f, 0.3f).normalized();
    program.setUniformValue("lightDir", lightDir);
    program.setUniformValue("shininess", 32.0f); // Если добавили в шейдер

    // Позиция камеры для освещения
    QVector3D camPos = QVector3D(30, 30, 30);
    program.setUniformValue("viewPos", camPos);

    // Палитра
    if (paletteTexture) {
        glActiveTexture(GL_TEXTURE0);
        paletteTexture->bind();
        program.setUniformValue("uPalette", 0);
    }

    // Рисуем (GL_POINTS, т.к. Geometry Shader превратит их в кубы)
    glDrawArrays(GL_POINTS, 0, voxelCount);

    if (paletteTexture) paletteTexture->release();
    vao.release();
    program.release();

    time += 0.016f;
}
