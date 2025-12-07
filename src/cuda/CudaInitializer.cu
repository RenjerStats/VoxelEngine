#include "cuda/CudaInitializer.h"
#include "core/Types.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <GL/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>



extern "C" {
void cuda_registerGLBuffer(unsigned int vboID, cudaGraphicsResource** pp_resource) {
    if (*pp_resource != nullptr) {
        cudaGraphicsUnregisterResource(*pp_resource);
        *pp_resource = nullptr;
    }

    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        pp_resource,
        vboID,
        cudaGraphicsMapFlagsWriteDiscard
        );

    if (err != cudaSuccess) {
        printf("CUDA Register Error: %s\n", cudaGetErrorString(err));
    }
}


void cuda_cleanup(cudaGraphicsResource* cuda_vbo_resource) {
    if (cuda_vbo_resource) {
        cudaGraphicsUnregisterResource(cuda_vbo_resource);
    }
}
}
