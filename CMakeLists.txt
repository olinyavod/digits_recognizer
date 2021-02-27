cmake_minimum_required (VERSION 3.8)

project (digits_recogmizer)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

add_executable(main WIN32 "src/main.cpp"  "include/neural_network.hpp")

target_link_libraries(main ${SDL2_LIBRARIES})


