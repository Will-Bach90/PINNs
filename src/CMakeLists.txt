file(GLOB SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(pinn_lib ${SRC_FILES})
target_include_directories(pinn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(pinn src/main.cpp)
target_link_libraries(pinn PRIVATE pinn_lib)