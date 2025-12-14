# 查找models目录下的源文件
file(GLOB_RECURSE MODEL_SOURCES "models/*.cpp")
file(GLOB_RECURSE MODEL_HEADERS "models/*.h")

# 添加可执行文件
add_executable(main 
    main.cpp
    ${MODEL_SOURCES}
)

# 设置包含目录
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}           # 当前目录
    ${CMAKE_CURRENT_SOURCE_DIR}/models    # models头文件
    ${CMAKE_SOURCE_DIR}/common/include    # common头文件
    ${CMAKE_SOURCE_DIR}/ops/include       # ops公共头文件
    ${CMAKE_SOURCE_DIR}/ops/include/${TARGET_ARCH}  # 架构特定头文件
)

# 链接库
target_link_libraries(main PRIVATE ops common tokenizers_cpp)

# 设置编译器选项
target_compile_options(main PRIVATE -Wall -Wextra)

# 传递架构定义
target_compile_definitions(main PRIVATE 
    ARCH_${TARGET_ARCH}=1
    MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models"
)