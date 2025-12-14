#include <fileutils.h>



int read_from_file(void* buffer, int size, const char* filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (file) {
        file.read(reinterpret_cast<char*>(buffer), size);
        file.close();
        return 0;
    }
    return 1;
}

int write_to_file(void* buffer, int size, const char* filename)
{
    std::ofstream file(filename, std::ios::binary);

    if (file) {
        file.write(reinterpret_cast<char*>(buffer), size);
        file.close();
        return 0;
    }
    std::cout<<" not write successfully!"<<std::endl;
    
    return 1;
}

json11::Json Get_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string err;

    json11::Json json = json11::Json::parse(buffer.str(),err);

    if (!err.empty()) {
            throw std::runtime_error("JSON 解析错误: " + err);
    }

    return json;
}

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}
