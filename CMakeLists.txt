cmake_minimum_required(VERSION 3.5)

project(xiaoleiReId)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-10.0/include)
link_directories(/usr/local/cuda-10.0//lib64)
# tensorrt
# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(/home/cxl/tensorRt/TensorRT-7.0.0.11/include)
link_directories(/home/cxl/tensorRt/TensorRT-7.0.0.11/lib/)

# set(OpenCV_DIR usr/local/share/opencv4)
find_package(OpenCV 4.1 REQUIRED)
# include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS} /usr/local/include/opencv4)
# find_package("/usr/local/include/opencv4/opencv2")
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -mavx2 -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})
#cuda_add_library(leaky ${PROJECT_SOURCE_DIR}/leaky.cu)
# cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
# target_link_libraries(yololayer nvinfer cudart ${OpenCV_LIBS})
# target_link_libraries(yololayer nvinfer cudart "/usr/local/lib/opencv4.1/")
aux_source_directory(. SOURCEFILE)
add_executable(xiaoleiReId ${SOURCEFILE})
# add_executable(xiaoleiReId ${PROJECT_SOURCE_DIR}/logger.cpp ${PROJECT_SOURCE_DIR}/myTensorRT.cpp ${PROJECT_SOURCE_DIR}/logger.h ${PROJECT_SOURCE_DIR}/common.h   ${PROJECT_SOURCE_DIR}/argsParser.h ${PROJECT_SOURCE_DIR}/buffers.h ${PROJECT_SOURCE_DIR}/half.h  ${PROJECT_SOURCE_DIR}/scanfFile.cpp)
target_link_libraries(xiaoleiReId nvinfer)
target_link_libraries(xiaoleiReId cudart)
target_link_libraries(xiaoleiReId nvcaffe_parser)
# target_link_libraries(xiaoleiReId yololayer)
target_link_libraries(xiaoleiReId ${OpenCV_LIBS})
# target_link_libraries(xiaoleiReId "/usr/local/lib/opencv4.1/")
target_link_libraries(${PROJECT_NAME} libdlib.a)
add_definitions(-O0 -pthread)

