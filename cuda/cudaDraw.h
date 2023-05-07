
#ifndef __CUDA_DRAW_H__
#define __CUDA_DRAW_H__

#include "cudaUtility.h"
#include "imageFormat.h"



/**
 * cudaDrawLine
 * @ingroup drawing
 */
cudaError_t cudaDrawLine( void* input, void* output, size_t width, size_t height, imageFormat format, 
					 int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 );
	
/**
 * cudaDrawLine
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawLine( T* input, T* output, size_t width, size_t height, 
				 	 int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 )	
{ 
	return cudaDrawLine(input, output, width, height, imageFormatFromType<T>(), x1, y1, x2, y2, color, line_width); 
}

/**
 * cudaDrawLine (in-place)
 * @ingroup drawing
 */
inline cudaError_t cudaDrawLine( void* image, size_t width, size_t height, imageFormat format, 
						   int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 )
{
	return cudaDrawLine(image, image, width, height, format, x1, y1, x2, y2, color, line_width);
}					
	
/**
 * cudaDrawLine (in-place)
 * @ingroup drawing
 */
template<typename T> 
cudaError_t cudaDrawLine( T* image, size_t width, size_t height, 
				 	 int x1, int y1, int x2, int y2, const float4& color, float line_width=1.0 )	
{ 
	return cudaDrawLine(image, width, height, imageFormatFromType<T>(), x1, y1, x2, y2, color, line_width); 
}	



cudaError_t cudaDeinterlace( void* input_cur, void* output, size_t width, size_t height);
#endif