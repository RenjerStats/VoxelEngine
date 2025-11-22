#include <QApplication>
#include <QSurfaceFormat>
#include <QDebug>
#include <QDir>

#include "ui/MainWindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow window;
    window.show();

    return app.exec();
}
