#pragma once

#include <QObject>
#include <QPoint>
#include <QVector3D>
#include <QMouseEvent>

class VoxelWindow;

class MouseInputHandler : public QObject {
    Q_OBJECT

public:
    explicit MouseInputHandler(VoxelWindow* voxelWindow, QObject* parent = nullptr);
    ~MouseInputHandler() = default;

    void mousePressEvent(QMouseEvent* event);
    void mouseMoveEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);

private:
    VoxelWindow* m_voxelWindow = nullptr;

    QPoint m_lastMousePos;
    bool m_isLeftButtonPressed = false;
    bool m_isRightButtonPressed = false;

    float m_cameraYaw = 0.0f;
    float m_cameraPitch = 0.0f;
    float m_cameraRoll = 0.0f;


    float m_lightYaw = 0.0f;
    float m_lightPitch = 0.0f;

    float m_cameraSensitivity = 0.5f;
    float m_lightSensitivity = 0.1f;

    void updateCameraRotation();
    void updateLightDirection();
    void clampAngles();
};
