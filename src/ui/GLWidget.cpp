#include "ui/GLWidget.h"
#include <QDebug>

GLWidget::GLWidget(QWidget* parent)
    : QOpenGLWidget(parent)
    , m_time(0.0f)
{
    // Устанавливаем формат OpenGL контекста
    QSurfaceFormat format;
    //format.setVersion(2, 1);                    // OpenGL 4.5
    format.setProfile(QSurfaceFormat::CoreProfile); // Core Profile
    format.setDepthBufferSize(24);              // 24-bit depth buffer
    format.setSamples(4);                       // 4x MSAA (antialiasing)
    setFormat(format);

    // Таймер для постоянной перерисовки (~60 FPS)
    m_timer = new QTimer(this);
    connect(m_timer, &QTimer::timeout, this, QOverload<>::of(&GLWidget::update));
    m_timer->start(16); // ~60 FPS (16ms)
}

GLWidget::~GLWidget() {
    // Очистка ресурсов OpenGL
    makeCurrent();  // Активируем контекст перед очисткой
    // TODO: здесь будем удалять VBO, VAO, шейдеры
    doneCurrent();  // Деактивируем контекст
}

void GLWidget::initializeGL() {
    // Инициализируем функции OpenGL
    if (!initializeOpenGLFunctions()) {
        qCritical() << "Failed to initialize OpenGL functions!";
        return;
    }

    // Выводим информацию о видеокарте
    qDebug() << "OpenGL Version:" << (const char*)glGetString(GL_VERSION);
    qDebug() << "GLSL Version:" << (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
    qDebug() << "Renderer:" << (const char*)glGetString(GL_RENDERER);

    // Настройка OpenGL состояния
    glEnable(GL_DEPTH_TEST);           // Включаем тест глубины
    glDepthFunc(GL_LESS);              // Пиксель рисуется, если ближе
    glEnable(GL_CULL_FACE);            // Включаем отсечение задних граней
    glCullFace(GL_BACK);               // Отсекаем задние грани

    // Цвет фона (тёмно-синий)
    glClearColor(0.1f, 0.15f, 0.2f, 1.0f);

    qDebug() << "OpenGL initialized successfully!";
}

void GLWidget::resizeGL(int w, int h) {
    // Устанавливаем viewport на весь виджет
    glViewport(0, 0, w, h);

    qDebug() << "Resize:" << w << "x" << h;
}

void GLWidget::paintGL() {
    // Очищаем буферы цвета и глубины
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Анимация цвета фона (плавное изменение)
    m_time += 0.016f; // ~60 FPS
    float r = 0.1f + 0.05f * sin(m_time);
    float g = 0.15f + 0.05f * sin(m_time * 0.7f);
    float b = 0.2f + 0.1f * sin(m_time * 0.5f);
    glClearColor(r, g, b, 1.0f);

    // TODO: здесь будем рисовать вокселы
}
