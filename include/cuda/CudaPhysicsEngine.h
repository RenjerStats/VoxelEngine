#pragma once

extern "C" {
void cuda_registerGLBuffer(unsigned int vboID);
void cuda_runSimulation(float dt, size_t numVoxels);
void cuda_cleanup();
}
