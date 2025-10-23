cmake_minimum_required(VERSION 3.12)
project(MyProject)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O3")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(REPLACE "/Ob2" "/Ob1 /Gy" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX /std:c++17 /arch:AVX2 /source-charset:utf-8")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -march=native")
endif()


# 自动检测目标架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    set(TARGET_ARCH "arm")
    message(STATUS "Building for ARM architecture")
else()
    set(TARGET_ARCH "x86")
    message(STATUS "Building for x86 architecture")
endif()

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加子目录
add_subdirectory(third_party/tokenizers-cpp)
add_subdirectory(common)
add_subdirectory(ops)
add_subdirectory(examples)