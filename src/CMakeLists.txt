set(CMAKE_CXX_STANDARD 17)

file(GLOB SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

file(GLOB HDR_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/incluce/*.h"
)

add_library(pinn_lib ${SRC_FILES})
target_include_directories(pinn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(pinn src/main.cpp src/node.cpp)
target_link_libraries(pinn PRIVATE pinn_lib)