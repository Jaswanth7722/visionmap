cmake_minimum_required(VERSION 3.16)
project(ps26053_lidar_mapping VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler definitions for cross-platform and MinGW compatibility
if(MINGW)
    add_compile_definitions(_stdcall=__stdcall)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
# 1. Eigen3
find_package(Eigen3 QUIET)
if(TARGET Eigen3::Eigen)
    set(EIGEN_TARGET Eigen3::Eigen)
    message(STATUS "Found Eigen3::Eigen target")
else()
    find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Dense PATHS "C:/msys64/mingw64/include/eigen3" "/usr/include/eigen3")
    if(EIGEN3_INCLUDE_DIR)
        set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
        message(STATUS "Found Eigen3 headers: ${EIGEN3_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "Eigen3 not found!")
    endif()
endif()

# 2. yaml-cpp
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    find_path(YAML_CPP_INCLUDE_DIR NAMES yaml-cpp/yaml.h PATHS "C:/msys64/mingw64/include" "/usr/include")
    find_library(YAML_CPP_LIBRARY NAMES yaml-cpp libyaml-cpp PATHS "C:/msys64/mingw64/lib" "/usr/lib")
    if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARY)
        message(STATUS "Found yaml-cpp: ${YAML_CPP_LIBRARY}")
    else()
        message(FATAL_ERROR "yaml-cpp not found!")
    endif()
else()
    set(YAML_CPP_LIBRARY yaml-cpp)
endif()

# 3. ONNX Runtime C++
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS ${ONNXRUNTIME_LIB_DIR} NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_LIBRARY)
    # Search system
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
endif()
message(STATUS "ONNX Runtime C++ Library: ${ONNXRUNTIME_LIBRARY}")

# -----------------------------------------------------------------------------
# Core Library: ps26053_core
# -----------------------------------------------------------------------------
set(CORE_SOURCES
    cpp/src/io/lidar_io.cpp
    cpp/src/preprocessing/point_cloud_ops.cpp
    cpp/src/resolution/resolution_policy.cpp
    cpp/src/mapping/quadtree.cpp
    cpp/src/mapping/grid_25d.cpp
    cpp/src/inference/onnx_engine.cpp
    cpp/src/tracking/kalman_tracker.cpp
    cpp/src/runtime/pipeline.cpp
)

add_library(ps26053_core STATIC ${CORE_SOURCES})

target_include_directories(ps26053_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    "C:/msys64/mingw64/include/eigen3"
    ${EIGEN3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(ps26053_core PUBLIC
    ${EIGEN_TARGET}
    ${YAML_CPP_LIBRARY}
    ${ONNXRUNTIME_LIBRARY}
)

# Helper function to copy onnxruntime.dll to binary directory on Windows
function(copy_onnx_dll target_name)
    if(WIN32 AND EXISTS "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
            $<TARGET_FILE_DIR:${target_name}>
        )
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Applications
# -----------------------------------------------------------------------------
add_executable(lidar_mapper cpp/apps/lidar_mapper.cpp)
target_link_libraries(lidar_mapper PRIVATE ps26053_core)
copy_onnx_dll(lidar_mapper)

add_executable(benchmark cpp/apps/benchmark.cpp)
target_link_libraries(benchmark PRIVATE ps26053_core)
copy_onnx_dll(benchmark)

add_executable(live_dashboard_server cpp/apps/live_dashboard_server.cpp)
target_link_libraries(live_dashboard_server PRIVATE ps26053_core ws2_32 psapi)
copy_onnx_dll(live_dashboard_server)

# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------
enable_testing()

set(TEST_TARGETS
    test_resolution
    test_quadtree
    test_projection
    test_tracking
    test_temporal_fusion
    test_boundary_alignment
)

foreach(test_name ${TEST_TARGETS})
    add_executable(${test_name} tests/cpp/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE ps26053_core)
    copy_onnx_dll(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
