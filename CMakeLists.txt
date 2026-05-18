cmake_minimum_required(VERSION 3.20)
project(tensor_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
    -fvisibility=hidden
)
add_compile_options($<$<CONFIG:Release>:-O3>)
add_compile_options($<$<CONFIG:Debug>:-O0>)
add_compile_options($<$<CONFIG:Debug>:-g>)

set(ENGINE_SOURCES
    src/allocator.cpp
    src/storage.cpp
    src/tensor_impl.cpp
    src/broadcast.cpp
    src/prng.cpp
    src/ops/constructors.cpp
    src/ops/random.cpp
    src/ops/unary.cpp
    src/ops/binary.cpp
    src/ops/cast.cpp
    src/ops/gather.cpp
    src/ops/reduce.cpp
    src/ops/matmul.cpp
    src/ops/inplace.cpp
    src/ffi.cpp
)

# Engine sources compiled once into an object library. Two consumers:
#   - the public shared library tensor_engine, exposing only the C ABI;
#   - test executables, which need C++ internals (hidden symbols) and
#     therefore link the object files directly rather than the .dylib.
add_library(tensor_engine_objs OBJECT ${ENGINE_SOURCES})
target_include_directories(tensor_engine_objs
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_library(tensor_engine SHARED $<TARGET_OBJECTS:tensor_engine_objs>)
target_link_libraries(tensor_engine PUBLIC tensor_engine_objs)

# Land the .so/.dylib next to the Python package so ctypes can find it.
set_target_properties(tensor_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tensor_autograd
)

enable_testing()
add_subdirectory(tests/cpp)

# Python tests run via stdlib unittest (no pytest dependency). They import
# the package from the repo root and load the freshly-built shared library
# that the engine target dropped into tensor_autograd/. The CMake test
# implicitly depends on the engine being built first.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME python_tests
        COMMAND ${Python3_EXECUTABLE} -m unittest discover -v
                -s ${CMAKE_SOURCE_DIR}/tests/python
                -t ${CMAKE_SOURCE_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()
