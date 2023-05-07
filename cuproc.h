#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "nvbuf_utils.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cudaEGL.h"

using namespace std;

class CudaProcess
{
public:
    CudaProcess();
    ~CudaProcess();
    void* get_img_ptr();
    void freeImage();
    void setFd(int fd);
private:
    int fd;
    EGLImageKHR image;
    CUresult status;
    CUeglFrame eglFrame;
    CUgraphicsResource pResource = NULL;
};