cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp common.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")

add_executable(iris iris.cpp common.cpp)
target_link_libraries(iris "${TORCH_LIBRARIES}")

add_executable(mnist mnist-cnn.cpp common.cpp)
target_link_libraries(mnist "${TORCH_LIBRARIES}")

add_executable(names names.cpp common.cpp)
target_link_libraries(names "${TORCH_LIBRARIES}")

add_subdirectory(arclas)