#include "cuproc.h"

/**
 * @brief Construct a new Cuda Process:: Cuda Process object
 * 
 */
CudaProcess::CudaProcess()
{

}

/**
 * @brief Destroy the Cuda Process:: Cuda Process object
 * 
 */
CudaProcess::~CudaProcess()
{

}


/**
 * @brief 
 * 
 * @param fd 
 */
void CudaProcess::setFd(int fd)
{
    this->fd = fd;
}

void* CudaProcess::get_img_ptr()
{
    image = NvEGLImageFromFd(NULL,fd);
    cudaFree(0);
    status = cuGraphicsEGLRegisterImage(&pResource, image,CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (status != CUDA_SUCCESS) {
        printf("cuGraphicsEGLRegisterImage failed in : %d, cuda process stop\n",status);
        return NULL;
    }
    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
    if (status != CUDA_SUCCESS) {
        printf("cuGraphicsSubResourceGetMappedArray failed\n");
        return NULL;
    }
    return eglFrame.frame.pPitch[0];
}

void CudaProcess::freeImage()
{
    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS) {
        printf("cudaNoSingalCreate failed after memcpy\n");
    }

    status = cuGraphicsUnregisterResource(pResource);
    if (status != CUDA_SUCCESS) {
        printf("cudaNoSingalCreate cuGraphicsEGLUnRegisterResource failed: %d\n", status);
    }
}

