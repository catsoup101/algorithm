cmake_minimum_required(VERSION 3.10)
project(BezierCurve)

find_package(OpenCV REQUIRED)

include(CTest)
include(CPack)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "C:\\Sofware\\opencv\\build\\install") 
set(OpenCV_DIR "C:\\Sofware\\oopencv\\build\\include") 
set(OpenCV_DIR "C:\\Program Files (x86)\\Eigen3\\share\\eigen3\\cmake")

include_directories("C:/Sofware/opencv/build/include")
include_directories("C:/Sofware/opencv/build/install") 
include_directories("C:/Program Files (x86)/Eigen3/include")

add_executable(BezierCurve main.cpp)

target_link_libraries(BezierCurve ${OpenCV_LIBRARIES})
