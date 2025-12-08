cmake_minimum_required(VERSION 3.16)
project(VoxelEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets OpenGLWidgets)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

add_definitions(-DQT_NO_KEYWORDS) # пришлось использовать, так как KEYWORDS конфликтуют с библиотекой ogt_vox

set(HEADERS
    include/Application.h
    # Core
    include/core/Types.h
    include/core/Config.h
    include/core/Logger.h
    # IO
    include/io/VoxFileParser.h
    include/io/VoxScene.h
    include/io/ogt_vox.h
    # Physics
    include/physics/PhysicsManager.h
    # CUDA
    include/cuda/KernelLauncher.h
    include/cuda/CudaInitializer.h
    include/cuda/SpatialHash.h
    # Rendering
    include/rendering/VoxelRenderer.h
    include/rendering/GLCudaInterop.h
    include/rendering/Camera.h
    include/rendering/ShaderProgram.h
    # UI
    include/ui/MainWindow.h
    include/ui/VoxelWindow.h
    include/ui/ControlPanel.h
)

set(SOURCES
    src/main.cpp
    src/Application.cpp
    # Core
    src/core/Types.cpp
    src/core/Logger.cpp
    # IO
    src/io/VoxFileParser.cpp
    src/io/VoxScene.cpp
    # Physics
    src/physics/PhysicsManager.cpp
    # Rendering
    src/rendering/VoxelRenderer.cpp
    src/rendering/GLCudaInterop.cpp
    src/rendering/Camera.cpp
    src/rendering/ShaderProgram.cpp
    # UI
    src/ui/MainWindow.cpp
    src/ui/VoxelWindow.cpp
    src/ui/ControlPanel.cpp
)

set(CUDA_SOURCES
    src/cuda/Gravity.cu
    src/cuda/SpatialHash.cu
    src/cuda/FloorCollision.cu
    src/cuda/VoxelCollision.cu
    src/cuda/VoxelSortByY.cu
    src/cuda/CudaInitializer.cu
)

set(RESOURCES
    shaders/shaders.qrc
)

add_executable(VoxelEngine
    ${HEADERS}
    ${SOURCES}
    ${CUDA_SOURCES}
    ${RESOURCES}

)

target_include_directories(VoxelEngine PRIVATE include)

target_link_libraries(VoxelEngine
    Qt6::Core
    Qt6::Widgets
    Qt6::OpenGLWidgets
    CUDA::cudart
    CUDA::cuda_driver
    OpenGL::GL
)

set_target_properties(VoxelEngine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;86"
)

# Автоматически создаёт иерархию по структуре папок
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADERS}
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PREFIX "Source Files"
    FILES ${SOURCES} ${CUDA_SOURCES}
)

source_group("Resources" FILES ${RESOURCES})

# Копируем папку assets в папку, где лежит .exe
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
