# ops/CMakeLists.txt

# 源文件收集
file(GLOB OPS_SOURCES 
    "src/${TARGET_ARCH}/*.cpp"
    "src/shared/*.cpp"
)

# 创建静态库
add_library(ops STATIC ${OPS_SOURCES})

# 包含目录
target_include_directories(ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/${TARGET_ARCH}
    ${CMAKE_SOURCE_DIR}/common/include
)

# 链接依赖
target_link_libraries(ops PRIVATE common)

# 架构特定配置
if(TARGET_ARCH STREQUAL "arm")
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        # Apple Silicon优化
        target_compile_options(ops PRIVATE
            -march=armv8.5-a+fp16+dotprod+fp16fml
        )
    else()
        # 通用ARM配置
        target_compile_options(ops PRIVATE
            -march=native
        )
    endif()
    
    target_compile_definitions(ops PRIVATE
        ARM_NEON_ENABLED=1
        ARM_FP16_ENABLED=1
    )
elseif(TARGET_ARCH STREQUAL "x86")
    target_compile_options(ops PRIVATE
        -mavx2
        -mf16c
    )
endif()

# 通用定义
target_compile_definitions(ops PRIVATE
    ARCH_${TARGET_ARCH}=1
    FP16_SUPPORTED=1
)