cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

# set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(jetosn_api VERSION 0.0.0 LANGUAGES C CXX)

set(CUDA_PATH /usr/local/cuda-10.2)
set(TEGRA_ARMABI aarch64-linux-gnu)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.2)

find_package(Threads)
find_package(CUDA REQUIRED)
# find_package(CUDNN REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(TensorRT REQUIRED)


set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER /usr/bin/g++)

include_directories(
    include
    cuda
    image
    threads
	${CUDA_PATH}/include
	/usr/include/${TEGRA_ARMABI}
	/usr/include/libdrm
    )


file(GLOB CLASS_SRC "classes/*.cpp" "cuda/*.cpp" "cuda/*.cu" "image/*.cpp threads/*.cpp" "*.cpp")

link_directories(	
            ${CUDA_PATH}/lib64
            /usr/lib/${TEGRA_ARMABI}
            /usr/lib/${TEGRA_ARMABI}/tegra
        )

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_62,code=sm_62)


SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -Wunused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-write-strings")	
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# TARGETS --------------------------------------------------------------------------------------------------------------

cuda_add_executable(${PROJECT_NAME} main.cpp ${CLASS_SRC} )
set(CMAKE_CXX_STANDARD 14)
add_compile_options(${CMAKE_MFC_FLAG} )
target_link_libraries(${PROJECT_NAME}  	-lpthread -lv4l2 -lEGL -lGLESv2 -lX11 -lnvbuf_utils -lnvjpeg -lnvosd -ldrm -lcuda -lcudart ${CUDA_nppicc_LIBRARY})

