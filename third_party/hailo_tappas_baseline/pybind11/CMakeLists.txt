cmake_minimum_required(VERSION 3.18)

project(hailo_tappas_baseline CXX)

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
    message(FATAL_ERROR
        "GCC 11+ required (HailoRT C++ API uses C++20 features). "
        "Found ${CMAKE_CXX_COMPILER_VERSION} at ${CMAKE_CXX_COMPILER}.\n"
        "Install: sudo apt install gcc-11 g++-11\n"
        "Reconfigure: -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(UPSTREAM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../upstream)

# ---- Required system packages ------------------------------------------------

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ---- xtl + xtensor (header-only, mirror upstream's ExternalProject pattern) --

include(ExternalProject)
set(EXT_PREFIX ${CMAKE_BINARY_DIR}/external)
ExternalProject_Add(xtl-baseline
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    GIT_SHALLOW TRUE
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXT_PREFIX}
)
ExternalProject_Add(xtensor-baseline
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    GIT_SHALLOW TRUE
    DEPENDS xtl-baseline
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${EXT_PREFIX}
        -Dxtl_DIR=${EXT_PREFIX}/share/cmake/xtl/
)

# ---- Pybind11 module ---------------------------------------------------------

pybind11_add_module(hailo_tappas_baseline
    hailo_pybind.cpp
    ${UPSTREAM_DIR}/instance_segmentation/instance_seg_postprocess.cpp
    ${UPSTREAM_DIR}/common/toolbox.cpp
)

# Match upstream's permissive warning posture (their headers trigger several).
target_compile_options(hailo_tappas_baseline PRIVATE
    -O3 -DNDEBUG -Wall
    -Wno-reorder -Wno-ignored-qualifiers -Wno-unused-local-typedefs
    -Wno-parentheses -Wno-array-bounds -Wno-unused-value
    -Wno-conversion -Wno-unused-parameter -Wno-missing-field-initializers
    -Wno-sign-compare -Wno-uninitialized
)

add_dependencies(hailo_tappas_baseline xtensor-baseline)

target_include_directories(hailo_tappas_baseline PRIVATE
    ${EXT_PREFIX}/include
    ${UPSTREAM_DIR}/instance_segmentation
    ${UPSTREAM_DIR}/common
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(hailo_tappas_baseline PRIVATE
    HailoRT::libhailort
    Threads::Threads
    ${OpenCV_LIBS}
)

# Install next to the source tree by default so an in-tree development install
# (`pip install -e .`) of ara2-validator can pick the .so up via package_data.
# Override with -DCMAKE_INSTALL_PREFIX=... if shipping system-wide.
install(TARGETS hailo_tappas_baseline
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
