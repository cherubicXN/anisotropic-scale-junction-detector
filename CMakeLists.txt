
cmake_minimum_required (VERSION 2.6)
project (ASJDetector)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
AUX_SOURCE_DIRECTORY(./sources SRC_LIST)

message(${SRC_LIST})

add_executable(ASJDetector ${SRC_LIST})
target_link_libraries(ASJDetector ${OpenCV_LIBS})