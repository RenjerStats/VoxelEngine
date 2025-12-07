#pragma once

class cudaGraphicsResource;

extern "C" {
void cuda_registerGLBuffer(unsigned int vboID, cudaGraphicsResource** pp_resource);
void cuda_cleanup(cudaGraphicsResource* cuda_vbo_resource);
}
