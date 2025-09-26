// --- 상단 includes (원본에 추가) ---
#include <thread>
#include <mutex>
#include <atomic>

// ... 기존 includes ...
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <cstdint>

namespace fs = std::filesystem;


// ----- 데이터 구조체 정의 -----

struct Tensor {
    std::vector<int32_t> data;
    std::vector<int> shape;

    // 특정 위치의 인덱스를 계산하는 헬퍼 함수
    int get_index(int h, int w, int c) const {
        return (h * shape[1] + w) * shape[2] + c;
    }
};

struct StaticModel {
    std::map<std::string, Tensor> quant_params;
    std::map<std::string, std::pair<float, int32_t>> act_params;
    std::map<std::string, std::pair<int64_t, int32_t>> requant_params;
};


// ----- 파일 로드 함수 -----
StaticModel load_static_quantized_model(const std::string& param_dir) {
    StaticModel model;
    
    // ✨✨✨ 변경된 부분 시작 ✨✨✨
    // 1. quant_info.txt 로드 및 파싱
    std::ifstream meta_file(param_dir + "/quant_info.txt");
    std::string line;
    while (std::getline(meta_file, line)) {
        std::stringstream ss(line);
        std::string type, key;
        ss >> type >> key;

        if (type == "act_param") {
            float scale;
            int32_t zp;
            ss >> scale >> zp;
            model.act_params[key] = {scale, zp};
        } else if (type == "requant_param") {
            int64_t M;
            int32_t n;
            ss >> M >> n;
            model.requant_params[key] = {M, n};
        }
    }
    // ✨✨✨ 변경된 부분 끝 ✨✨✨

    // 2. 각 layer_..._q.txt 파일 로드 (이전과 동일)
    std::map<std::string, int> param_map = {
        {"W0", 0}, {"b0", 1}, {"W2", 2}, {"b2", 3}, {"W4", 4}, {"b4", 5},
        {"W6", 6}, {"b6", 7}, {"W8", 8}, {"b8", 9}
    };
    for (auto const& [name, index] : param_map) {
        std::ifstream layer_file(param_dir + "/layer_" + std::to_string(index) + "_q.txt");
        // ... (이하 로직은 이전 답변과 완전히 동일) ...
        std::string h_line;
        std::getline(layer_file, h_line);
        std::stringstream ss(h_line);
        std::vector<int> shape;
        int dim;
        while(ss >> dim) shape.push_back(dim);
        std::getline(layer_file, h_line); std::getline(layer_file, h_line);
        std::vector<int32_t> data;
        int32_t val;
        while(layer_file >> val) data.push_back(val);
        model.quant_params[name] = {data, shape};
    }
    return model;
}
Tensor load_and_quantize_image(const std::string& image_path, float s_x, int32_t z_x) {
    std::ifstream img_file(image_path);
    std::string line;
    std::getline(img_file, line); // Shape 헤더 무시
    
    std::vector<float> float_data;
    float val;
    while(img_file >> val) float_data.push_back(val);

    std::vector<int32_t> q_data;
    q_data.reserve(float_data.size());
    for(float f_val : float_data) {
        int32_t q_val = static_cast<int32_t>(round(f_val / s_x) + z_x);
        q_data.push_back(std::clamp(q_val, -128, 127));
    }
    return {q_data, {28, 28, 1}};
}


// ----- 핵심 연산 계층 함수들 -----

void relu_integer(Tensor& t, int32_t zero_point) {
    for (auto& val : t.data) {
        val = std::max(val, zero_point);
    }
}

void maxpool2d_integer(Tensor& out_tensor, const Tensor& in_tensor) {
    int H = in_tensor.shape[0], W = in_tensor.shape[1], C = in_tensor.shape[2];
    int out_h = (H - 2) / 2 + 1, out_w = (W - 2) / 2 + 1;
    out_tensor.shape = {out_h, out_w, C};
    out_tensor.data.resize(out_h * out_w * C);

    for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int32_t max_val = -128;
                for (int ph = 0; ph < 2; ++ph) {
                    for (int pw = 0; pw < 2; ++pw) {
                        int h = oh * 2 + ph;
                        int w = ow * 2 + pw;
                        max_val = std::max(max_val, in_tensor.data[in_tensor.get_index(h, w, c)]);
                    }
                }
                out_tensor.data[out_tensor.get_index(oh, ow, c)] = max_val;
            }
        }
    }
}

void conv2d_static_integer(Tensor& q_out, const Tensor& q_x, const Tensor& q_W, const Tensor& q_b, int32_t Zx, int32_t Zy, int64_t M, int32_t n, bool same_padding) {
    int H = q_x.shape[0], W = q_x.shape[1], Cin = q_x.shape[2];
    int fh = q_W.shape[0], fw = q_W.shape[1], Cout = q_W.shape[3];
    int pad_h = same_padding ? (fh - 1) / 2 : 0, pad_w = same_padding ? (fw - 1) / 2 : 0;
    int out_h = (H + 2 * pad_h - fh) + 1, out_w = (W + 2 * pad_w - fw) + 1;
    q_out.shape = {out_h, out_w, Cout};
    q_out.data.assign(out_h * out_w * Cout, 0);

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int cout = 0; cout < Cout; ++cout) {
                int64_t accumulator = 0;
                for (int cin = 0; cin < Cin; ++cin) {
                    for (int kh = 0; kh < fh; ++kh) {
                        for (int kw = 0; kw < fw; ++kw) {
                            int ih = oh + kh - pad_h, iw = ow + kw - pad_w;
                            int32_t x_val = Zx;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                x_val = q_x.data[q_x.get_index(ih, iw, cin)];
                            }
                            int w_idx = ((kh * fw + kw) * Cin + cin) * Cout + cout;
                            accumulator += static_cast<int64_t>(x_val - Zx) * (q_W.data[w_idx] - 0); // Zw is 0
                        }
                    }
                }
                accumulator += (q_b.data[cout] - 0); // Zb is 0
                int64_t scaled_acc = (accumulator * M) >> n;
                q_out.data[q_out.get_index(oh, ow, cout)] = std::clamp(static_cast<int32_t>(scaled_acc + Zy), -128, 127);
            }
        }
    }
}

void dense_static_integer(Tensor& q_out, const Tensor& q_x, const Tensor& q_W, const Tensor& q_b, int32_t Zx, int32_t Zy, int64_t M, int32_t n) {
    int n_input = q_x.data.size(), n_output = q_W.shape[1];
    q_out.shape = {n_output};
    q_out.data.resize(n_output);

    for (int j = 0; j < n_output; ++j) {
        int64_t accumulator = 0;
        for (int i = 0; i < n_input; ++i) {
            accumulator += static_cast<int64_t>(q_x.data[i] - Zx) * (q_W.data[i * n_output + j] - 0);
        }
        accumulator += (q_b.data[j] - 0);
        int64_t scaled_acc = (accumulator * M) >> n;
        q_out.data[j] = std::clamp(static_cast<int32_t>(scaled_acc + Zy), -128, 127);
    }
}


// ----- 전체 추론 파이프라인 -----

std::pair<Tensor, std::pair<float, int32_t>> lenet5_forward(const Tensor& q_x_in, const StaticModel& model) {
    Tensor q_x = q_x_in;
    Tensor temp_out;

    // Layer C1
    auto [s_in, z_in] = model.act_params.at("input");
    auto [s_c1, z_c1] = model.act_params.at("c1_out");
    auto [m1, n1] = model.requant_params.at("c1");
    conv2d_static_integer(temp_out, q_x, model.quant_params.at("W0"), model.quant_params.at("b0"), z_in, z_c1, m1, n1, true);
    q_x = temp_out; relu_integer(q_x, z_c1);

    // Layer S2
    maxpool2d_integer(temp_out, q_x);
    q_x = temp_out;

    // Layer C3
    auto [s_c3, z_c3] = model.act_params.at("c3_out");
    auto [m3, n3] = model.requant_params.at("c3");
    conv2d_static_integer(temp_out, q_x, model.quant_params.at("W2"), model.quant_params.at("b2"), z_c1, z_c3, m3, n3, false);
    q_x = temp_out; relu_integer(q_x, z_c3);

    // Layer S4
    maxpool2d_integer(temp_out, q_x);
    q_x = temp_out;

    // Layer C5
    auto [s_c5, z_c5] = model.act_params.at("c5_out");
    auto [m5, n5] = model.requant_params.at("c5");
    conv2d_static_integer(temp_out, q_x, model.quant_params.at("W4"), model.quant_params.at("b4"), z_c3, z_c5, m5, n5, false);
    q_x = temp_out; relu_integer(q_x, z_c5);

    // Flatten
    q_x.shape = {(int)q_x.data.size()};

    // Layer F6
    auto [s_f6, z_f6] = model.act_params.at("f6_out");
    auto [m6, n6] = model.requant_params.at("f6");
    dense_static_integer(temp_out, q_x, model.quant_params.at("W6"), model.quant_params.at("b6"), z_c5, z_f6, m6, n6);
    q_x = temp_out; relu_integer(q_x, z_f6);

    // Output Layer
    auto [s_out, z_out] = model.act_params.at("output_out");
    auto [m_out, n_out] = model.requant_params.at("output");
    dense_static_integer(temp_out, q_x, model.quant_params.at("W8"), model.quant_params.at("b8"), z_f6, z_out, m_out, n_out);
    q_x = temp_out;
    
    return {q_x, {s_out, z_out}};
}

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

// ----- Main 함수 -----
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "사용법: " << argv[0] << " <파라미터_디렉토리> <이미지_txt_디렉토리>" << std::endl;
        return 1;
    }

    std::string param_dir = argv[1];
    std::string image_dir = argv[2];

    std::cout << "===== C++ INT8 Evaluation Engine (single-thread baseline) =====" << std::endl;
    StaticModel model = load_static_quantized_model(param_dir);

    auto [s_x, z_x] = model.act_params.at("input");

    // -- 파일 목록 수집
    std::vector<std::string> file_list;
    for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) continue;
        file_list.push_back(entry.path().string());
    }
    if (file_list.empty()) {
        std::cerr << "이미지 파일이 없습니다: " << image_dir << std::endl;
        return 1;
    }

    int correct_count = 0;
    int total_count = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    // ----- 싱글스레드 루프 -----
    for (size_t i = 0; i < file_list.size(); ++i) {
        const std::string& path = file_list[i];

        // 파일 이름에서 정답 레이블 파싱
        size_t last_ = path.find_last_of('_');
        size_t last_dot = path.find_last_of('.');
        int true_label = -1;
        try {
            true_label = std::stoi(path.substr(last_ + 1, last_dot - last_ - 1));
        } catch (...) {
            std::cerr << "Warning: 파일명에서 레이블 파싱 실패: " << path << std::endl;
            total_count++;
            continue;
        }

        // 이미지 로드 및 양자화
        Tensor q_image = load_and_quantize_image(path, s_x, z_x);

        // 추론
        auto [q_out, out_params] = lenet5_forward(q_image, model);

        // 결과 해석 (Dequantization & ArgMax)
        float scale_out = out_params.first;
        int32_t zp_out = out_params.second;

        float max_score = -1e9f;
        int predicted_label = -1;
        for (int k = 0; k < (int)q_out.data.size(); ++k) {
            float score = static_cast<float>(q_out.data[k] - zp_out) * scale_out;
            if (score > max_score) {
                max_score = score;
                predicted_label = k;
            }
        }

        if (predicted_label == true_label) {
            correct_count++;
        }
        total_count++;

        if ((total_count % 100) == 0) {
            std::cout << "[Single-thread] processed " << total_count << " images." << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    double accuracy = (static_cast<double>(correct_count) / std::max(1, total_count)) * 100.0;

    std::cout << "\n===== Evaluation Result (Single-thread) =====" << std::endl;
    std::cout << "Threads used: 1" << std::endl;
    std::cout << "Total images evaluated: " << total_count << std::endl;
    std::cout << "Correct predictions: " << correct_count << std::endl;
    std::cout << "Total inference time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
