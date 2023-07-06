cmake_minimum_required(VERSION 3.25)
project(bvh_mixed_new)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS -O3)

add_subdirectory(bvh)
add_executable(bvh_mixed_new main.cpp marker.hpp)
target_link_libraries(bvh_mixed_new PUBLIC bvh mpfr)
