find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

add_executable(app WIN32 "src/main.cpp"  "include/neural_network.hpp")

target_link_libraries(app ${SDL2_LIBRARIES})