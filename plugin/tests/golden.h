#pragma once
// Access to the golden test vectors in tests/golden/ (plugin/tools/README.md).
// CMake passes the folder as DENOISER_GOLDEN_DIR; a test also accepts it as
// its first command-line argument.

#include "denoiser/json.h"
#include "denoiser/model_file.h"
#include "denoiser/network.h"
#include "denoiser/tensor_file.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace golden {

inline std::string& dir() {
#ifdef DENOISER_GOLDEN_DIR
    static std::string d = DENOISER_GOLDEN_DIR;
#else
    static std::string d = "golden";
#endif
    return d;
}

inline void init(int argc, char** argv) {
    if (argc > 1)
        dir() = argv[1];
}

inline std::string readText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

inline std::vector<char> readBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Case names from index.json, in its order
inline std::vector<std::string> caseNames() {
    const denoiser::Json index = denoiser::Json::parse(readText(dir() + "/index.json"));
    std::vector<std::string> names;
    for (const denoiser::Json& c : index.at("cases").array())
        names.push_back(c.at("name").string());
    return names;
}

struct Case {
    std::string name;
    denoiser::ModelFile model;
    denoiser::TensorFile vectors;

    const denoiser::ModelInfo& info() const { return model.info(); }
    const denoiser::Tensor* find(const char* tensor) const { return vectors.find(tensor); }
    const denoiser::Tensor& get(const char* tensor) const { return vectors.get(tensor); }
};

inline Case load(const std::string& name) {
    return Case{name, denoiser::ModelFile::load(dir() + "/" + name + ".model.bin"),
                denoiser::TensorFile::load(dir() + "/" + name + ".vectors.bin")};
}

// max |a - b| / max(1, max |b|): the error measure of tools/README.md
inline double relErr(const float* a, const float* b, std::size_t n) {
    double diff = 0.0, peak = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        diff = std::max(diff, std::fabs(double(a[i]) - double(b[i])));
        peak = std::max(peak, std::fabs(double(b[i])));
    }
    return diff / peak;
}

// The worst error over all rows compared so far of one tensor
struct ErrorTracker {
    double worst = 0.0;
    double peak = 1.0;
    double diff = 0.0;
    void add(const float* a, const float* b, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            diff = std::max(diff, std::fabs(double(a[i]) - double(b[i])));
            peak = std::max(peak, std::fabs(double(b[i])));
        }
        worst = diff / peak;
    }
};

// A network that replays the golden pred, row by row, and remembers the x it
// was given
class ReplayNetwork : public denoiser::FrameNetwork {
public:
    explicit ReplayNetwork(const denoiser::Tensor& pred) : pred_(pred) {}

    void reset() noexcept override {
        row_ = 0;
        xs_.clear();
    }

    void process(const float* x, float* y) noexcept override {
        const std::size_t rows = pred_.shape[0], cols = pred_.shape[1];
        xs_.emplace_back(x, x + nIn);
        if (row_ < rows)
            std::copy(pred_.data + row_ * cols, pred_.data + (row_ + 1) * cols, y);
        else
            std::fill(y, y + cols, 0.0f);
        ++row_;
    }

    std::size_t nIn = 0;
    std::size_t rows() const { return row_; }
    const std::vector<std::vector<float>>& inputs() const { return xs_; }

private:
    const denoiser::Tensor& pred_;
    std::size_t row_ = 0;
    std::vector<std::vector<float>> xs_;
};

}  // namespace golden
