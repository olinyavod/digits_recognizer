find_package(SDL2 REQUIRED)
find_package(SDL2IMAGE REQUIRED SDL2_image>=2.0.0)

include_directories(${SDL2_INCLUDE_DIRS} ${SDL2IMAGE_INCLUDE_DIRS})

add_executable(app WIN32 "src/main.cpp"  "include/neural_network.hpp" "include/main.hpp")

target_link_libraries(app ${SDL2_LIBRARIES} ${SDL2IMAGE_LIBRARIES})