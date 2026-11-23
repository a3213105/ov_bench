cmake_minimum_required(VERSION 3.16)
project(ov_model_benchmarks VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(nlohmann_json 3.6 REQUIRED)

add_library(ov_bench_core src/core.cpp src/stability.cpp)
target_include_directories(ov_bench_core PUBLIC include)
target_link_libraries(ov_bench_core PUBLIC openvino::runtime nlohmann_json::nlohmann_json OpenSSL::Crypto)

add_library(ov_bench_app src/application.cpp src/image.cpp src/random_accuracy.cpp)
target_include_directories(ov_bench_app PUBLIC include)
target_link_libraries(ov_bench_app PUBLIC ov_bench_core PNG::PNG JPEG::JPEG)

add_executable(ov_bench src/main.cpp tests/random_accuracy_core.cpp)
target_link_libraries(ov_bench PRIVATE ov_bench_app)
target_compile_definitions(ov_bench PRIVATE
    OVBENCH_REFERENCE_WORKER="${CMAKE_CURRENT_SOURCE_DIR}/tools/random_reference_worker.py")
foreach(target ov_bench_core ov_bench_app ov_bench)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endforeach()

option(OVBENCH_BUILD_MEMORY "Build the native OV/ONNX Runtime memory diagnostic" OFF)
if(OVBENCH_BUILD_MEMORY)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime libonnxruntime.so.1.23.2 REQUIRED)
    add_executable(pred_memory src/memory.cpp)
    target_include_directories(pred_memory PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(pred_memory PRIVATE ov_bench_app "${ONNXRUNTIME_LIBRARY}")
    target_compile_options(pred_memory PRIVATE -Wall -Wextra -Wpedantic)
endif()
