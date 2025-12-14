#pragma once
#include <common.h>
#include <cstddef> 


inline void * Allocate_Memory(size_t size){

    void* ptr = aligned_alloc(size,64);

    if(ptr == NULL) {
        std::cout << "workspace allocation failed" << std::endl;
        exit(-1);
    }

    return ptr;

}