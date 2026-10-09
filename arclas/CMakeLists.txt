add_executable(arclas arclas.cpp ../common.cpp)
target_link_libraries(arclas "${TORCH_LIBRARIES}")

add_executable(arclas-dataset arclas-dataset.cpp)
target_link_libraries(arclas-dataset "${TORCH_LIBRARIES}" "${OPENCV_LIBRARIES}")

find_package(PkgConfig)
pkg_check_modules(GTKMM REQUIRED IMPORTED_TARGET gtkmm-4.0)

link_directories(${GTKMM_LIBRARY_DIRS})
include_directories(${GTKMM_INCLUDE_DIRS})
add_executable(gtk-arclas gtk-arclas.cpp)
target_link_libraries(gtk-arclas PRIVATE "${TORCH_LIBRARIES}" PkgConfig::GTKMM)
