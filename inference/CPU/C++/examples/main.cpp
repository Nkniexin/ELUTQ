#include<llama.h>
#include<tokenizers_cpp.h>
#include<string>
#include<sstream>
#include<fstream>

struct Message {
    std::string role;
    std::string content;
};

std::string build_prompt(const std::vector<Message>& messages, bool add_generation_prompt) {
    std::string prompt;

    if (messages.empty()) return "";

    // if (messages[0].role != "system") {
    //     prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    // }

    for (const auto& msg : messages) {
        prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        prompt += "<|im_start|>assistant\n";
    }

    return prompt;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << " <Prompt_Path> " <<std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    std::string prompt_file_path = std::string(argv[2]);
    int n_threads = 4; // Example thread count, can be adjusted
    llama model(model_path, n_threads); // Initialize the model with 4-bit quantization

    // int input_num = 256;
    int generate_num = 50;

    // int32_t* inputs = (int32_t*)malloc(input_num * sizeof(int32_t));
    // read_from_file(inputs, input_num * sizeof(int32_t), (std::string("llama2_input.bin")).c_str());
    
    std::string tokenizer_path = std::string(model_path) + "/tokenizer.json";

    auto blob = LoadBytesFromFile(tokenizer_path);
    auto tok = tokenizers::Tokenizer::FromBlobJSON(blob);
    
    std::ifstream file(prompt_file_path);

    std::string content((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());

    std::string text = content;

    std::vector<Message> messages = {
        {"user", text}
    };

    std::string prompt = build_prompt(messages, true);

    // std::vector<int> input_tokens(inputs, inputs + input_num);

    std::vector<int> input_tokens = tok->Encode(prompt);


    std::cout<<"token_ids size: "<<input_tokens.size()<<std::endl;                                                           

	std::vector<int> position_ids;
	for (int i = 0; i < input_tokens.size(); i++) position_ids.push_back(i);
    
    // Example input tokens and position IDs
    // std::vector<int> input_tokens = {108386};
    // std::vector<int> position_ids = {0};

    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    int position = position_ids.size();
    std::vector<int>res;
    for(int i= 0;i < generate_num ;i++){

        model.forward(input_tokens, position_ids);
        res.clear();
        res.push_back(model.get_response());
        // std::cout<<res.back()<<std::endl;
        if(i == 0){
            end = std::chrono::steady_clock::now();
            std::chrono::duration<double> perfill_elapsed = end- start;
            std::cout << "prefill speed: " <<(input_tokens.size())/(perfill_elapsed.count()) << " tokens/s" << "\n" <<std::endl;
            start = std::chrono::steady_clock::now();
        }

        std::cout<<tok->Decode(res);
        flush(std::cout);
        input_tokens.clear();
        input_tokens.push_back(model.get_response());

        position_ids.clear();
        position_ids.push_back(position++);
    }

    end = std::chrono::steady_clock::now();
    std::cout<<std::endl;
    std::chrono::duration<double> decode_elapsed = end- start;
    std::cout << "decode speed: " <<(generate_num - 1)/(decode_elapsed.count()) << " tokens/s" << "\n" <<std::endl;

    model.reset(); // Reset the model state
    // free(inputs); // Free the allocated memory for inputs
    return 0;
}