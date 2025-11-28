cmake_minimum_required(VERSION 3.10)
project(cam_server)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ARAVIS REQUIRED aravis-0.8)
pkg_check_modules(ZMQ REQUIRED libzmq)

include_directories(
    ${ARAVIS_INCLUDE_DIRS}
    ${ZMQ_INCLUDE_DIRS}
)

link_directories(
    ${ARAVIS_LIBRARY_DIRS}
    ${ZMQ_LIBRARY_DIRS}
)

add_executable(cam_server cam_server.cpp)

target_link_libraries(cam_server
    ${ARAVIS_LIBRARIES}
    ${ZMQ_LIBRARIES}
)
