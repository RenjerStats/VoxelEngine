#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_5_Core>
#include <QTimer>

class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core {
    Q_OBJECT

public:
    explicit GLWidget(QWidget* parent = nullptr);
    ~GLWidget() override;

protected:
    // Три главных метода OpenGL lifecycle
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

private:
    QTimer* m_timer;      // Таймер для постоянной перерисовки
    float m_time;         // Время для анимации
};
