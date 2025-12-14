#pragma once
#include <fstream>
#include <iostream>
#include <filesystem>
#include "json11.hpp"

int read_from_file(void* buffer, int size, const char* filename);
int write_to_file(void* buffer, int size, const char* filename);
json11::Json Get_json(const std::string& path);
std::string LoadBytesFromFile(const std::string& path);
