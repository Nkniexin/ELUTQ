# 查找所有源文件
file(GLOB_RECURSE COMMON_SOURCES "src/*.cpp")
file(GLOB_RECURSE COMMON_HEADERS "include/*.h")

# 创建静态库
add_library(common STATIC ${COMMON_SOURCES} ${COMMON_HEADERS})

# 设置包含目录
target_include_directories(common PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 设置编译器选项
target_compile_options(common PRIVATE -Wall -Wextra)