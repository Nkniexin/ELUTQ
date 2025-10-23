#pragma once
#include <arm_nnstructures.h>
#include <common.h>
#include <fileutils.h>
#include <json11.hpp>
#include <Allocate.h>
#include <arm_t_mac_kernal.h>


enum class Parameter_type {
    PARAM_TYPE_LINEAR,
    PARAM_TYPE_EMBEDDING,
    PARAM_TYPE_RMS,
    PARAM_TYPE_LM_HEAD
};

weight_tensor* load_Parameter(const std::string & path,Parameter_type type ,size_t size,uint8_t* workplace,int n_thread);