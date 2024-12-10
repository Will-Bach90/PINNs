cmake_minimum_required(VERSION 3.15)
project(PINNProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

add_subdirectory(src)
# add_subdirectory(testing)