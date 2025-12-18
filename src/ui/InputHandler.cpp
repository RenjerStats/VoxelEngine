#include "ui/InputHandler.h"
#include "ui/VoxelWindow.h"

#include <QMouseEvent>
#include <QDebug>
#include <cmath>

InputHandler::InputHandler(VoxelWindow* voxelWindow, QObject* parent)
    : QObject(parent), m_voxelWindow(voxelWindow)
{
    m_cameraYaw = 45.0f;
    m_cameraPitch = 30.0f;

    updateCameraRotation();
}

void InputHandler::mousePressEvent(QMouseEvent* event)
{
    if (!event) return;

    m_lastMousePos = event->pos();

    if (event->button() == Qt::LeftButton) {
        m_isLeftButtonPressed = true;
    }

    if (event->button() == Qt::RightButton) {
        m_isRightButtonPressed = true;
    }
}

void InputHandler::mouseMoveEvent(QMouseEvent* event)
{
    if (!event || !m_voxelWindow) return;

    QPoint currentPos = event->pos();
    QPoint delta = currentPos - m_lastMousePos;

    if (delta.manhattanLength() > 100) {
        m_lastMousePos = currentPos;
        return;
    }

    if (m_isLeftButtonPressed) {
        m_cameraYaw -= delta.x() * m_cameraSensitivity;
        m_cameraPitch += delta.y() * m_cameraSensitivity;

        if (m_cameraYaw < 0.0f) m_cameraYaw += 360.0f;
        if (m_cameraYaw >= 360.0f) m_cameraYaw -= 360.0f;
        m_cameraPitch = std::clamp(m_cameraPitch, -89.0f, 89.0f);

        updateCameraRotation();
    }

    if (m_isRightButtonPressed) {
        m_lightYaw -= delta.x() * m_lightSensitivity;
        m_lightPitch += delta.y() * m_lightSensitivity;

        if (m_lightYaw < 0.0f) m_lightYaw += 360.0f;
        if (m_lightYaw > 360.0f) m_lightYaw -= 360.0f;

        m_lightPitch = std::clamp(m_lightPitch, -89.0f, 89.0f);

        updateLightDirection();
    }

    m_lastMousePos = currentPos;
}

void InputHandler::mouseReleaseEvent(QMouseEvent* event)
{
    if (!event) return;

    if (event->button() == Qt::LeftButton) {
        m_isLeftButtonPressed = false;
    }

    if (event->button() == Qt::RightButton) {
        m_isRightButtonPressed = false;
    }
}

void InputHandler::updateCameraRotation()
{
    m_voxelWindow->setCameraRotationX(m_cameraPitch);
    m_voxelWindow->setCameraRotationY(m_cameraYaw);
}

void InputHandler::updateLightDirection()
{
    float yawRad = qDegreesToRadians(m_lightYaw);
    float pitchRad = qDegreesToRadians(m_lightPitch);

    float x = std::cos(pitchRad) * std::sin(yawRad);
    float y = std::sin(pitchRad);
    float z = std::cos(pitchRad) * std::cos(yawRad);

    m_voxelWindow->setLightDirX(x);
    m_voxelWindow->setLightDirY(y);
    m_voxelWindow->setLightDirZ(z);
}
