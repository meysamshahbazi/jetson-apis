
#include "cudaDeinterlace.h"



__global__ void gpuDeinterlace( unsigned char* input_cur, unsigned char* output, size_t width, size_t height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int pitch = 1536; //1536;// this is very important and not necesserly is equal to 2*width !!

	// if(input_cur[720*2+2] == 0)
	// 	output[2*y*pitch + x] =  static_cast<unsigned char>( ( (int) input_cur[y*pitch+x] + (int) output[(2*y+1)*pitch + x] )/2 ); 

	// else if(input_cur[720*2+2] == 255)
	// 	output[(2*y+1)*pitch + x] =  static_cast<unsigned char>( ( (int) input_cur[y*pitch+x] + (int) output[2*y*pitch + x] )/2) ;

	// if(input_cur[720*2+2] == 0)
	// 	output[2*y*pitch + x] =  static_cast<unsigned char>(  input_cur[y*pitch+x]); 

	// else if(input_cur[720*2+2] == 255)
	// 	output[(2*y+1)*pitch + x] =  static_cast<unsigned char>( input_cur[y*pitch+x] ) ;
	
	// if( x == 0 && y ==0 ) {
	// 	printf("field 0:%d \t",	input_cur[738*2+0] );
	// 	printf(" | 1:%d \t",	input_cur[721*2+1] );
	// 	printf(" | 2:%d \t",	input_cur[721*2+2] );
	// 	printf(" | 3:%d \n",	input_cur[721*2+3] );
	// }

	
	
	unsigned char field_flag[1];//

	field_flag[0] = input_cur[721*2+2];

	if(field_flag[0] == 0){
		output[2*y*pitch + x] = static_cast<unsigned char>(input_cur[y*pitch+x] ); 
		// output[(2*y+1)*pitch + x] = static_cast<unsigned char>( ( (int) input_cur[y*pitch+x] + (int) output[(2*y+1)*pitch + x] )/2) ;
	}
	else if(field_flag[0] == 255) {
		output[(2*y+1)*pitch + x] =  static_cast<unsigned char>(input_cur[y*pitch+x] );
		// output[(2*y)*pitch + x] = static_cast<unsigned char>( ( (int) input_cur[y*pitch+x] + (int) output[(2*y)*pitch + x] )/2) ;
	}

}

cudaError_t cudaDeinterlace( void* input_cur, void* output, size_t width, size_t height)
{	
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(2*720,blockDim.x), iDivUp(288,blockDim.y)); // 736 * 288
	gpuDeinterlace<<<gridDim, blockDim>>>((unsigned char *) input_cur, 
										(unsigned char *) output,width,height);// TODO: add pitch

	return cudaGetLastError();
}





