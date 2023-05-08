#ifndef _CUDA_DEINTERLACE_H_
#define _CUDA_DEINTERLACE_H_
#include "cudaUtility.h"

cudaError_t cudaDeinterlace(void* input_cur, void* output, size_t width, size_t height);

#endif
