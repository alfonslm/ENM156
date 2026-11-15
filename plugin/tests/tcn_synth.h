#pragma once
// Synthetic TCN model files with random weights, built in memory (no Python):
// for the benchmark and for the brute-force check of the incremental
// evaluation. The STFT settings are the repo's defaults (n_fft 128, hop 80,
// long window 1280, magnitude output: n_in 130, n_out 65, relu).

#include "denoiser/model_file.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace tcn_synth {

struct Spec {
    int channels = 256;
    int kernel = 3;
    std::vector<int> dilations{1, 2, 4, 8, 16, 32};
    std::uint32_t seed = 1;
    // Added to every LayerNorm bias: large values make every ReLU after a
    // norm pass everything (no zeros to skip: the worst case for speed)
    float normBiasShift = 0.0f;
    // Added to every conv bias (with normBiasShift, makes everything dense)
    float convBiasShift = 0.0f;
};

constexpr int kNIn = 130;
constexpr int kNOut = 65;

class Random {
public:
    explicit Random(std::uint32_t seed) : state_(seed * 2654435761u + 12345u) {}
    // Uniform in [-1, 1)
    float uniform() {
        state_ = state_ * 1664525u + 1013904223u;
        return float(state_ >> 8) / float(1u << 23) - 1.0f;
    }

private:
    std::uint32_t state_;
};

struct Named {
    std::string name;
    std::vector<std::size_t> shape;
    std::vector<float> data;
};

inline std::string container(const std::string& headerStart, const std::vector<Named>& tensors) {
    std::string list = "[";
    std::string data;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const Named& t = tensors[i];
        while (data.size() % 64 != 0)
            data += '\0';
        std::string shape = "[";
        for (std::size_t k = 0; k < t.shape.size(); ++k)
            shape += (k ? ", " : "") + std::to_string(t.shape[k]);
        shape += "]";
        list += std::string(i ? ", " : "") + "{\"name\": \"" + t.name + "\", \"shape\": " + shape +
                ", \"dtype\": \"f32\", \"offset\": " + std::to_string(data.size()) + "}";
        const std::size_t at = data.size();
        data.resize(at + 4 * t.data.size());
        for (std::size_t k = 0; k < t.data.size(); ++k) {
            std::uint32_t bits;
            std::memcpy(&bits, &t.data[k], 4);
            for (int b = 0; b < 4; ++b)
                data[at + 4 * k + std::size_t(b)] = char((bits >> (8 * b)) & 0xFF);
        }
    }
    list += "]";
    std::string header = headerStart + ", \"tensors\": " + list + "}";
    while ((16 + header.size()) % 64 != 0)
        header += ' ';
    std::string out = "ENMTENSR";
    for (std::uint32_t v : {1u, static_cast<std::uint32_t>(header.size())})
        for (int b = 0; b < 4; ++b)
            out += char((v >> (8 * b)) & 0xFF);
    return out + header + data;
}

// The bytes of a model file
inline std::string modelBytes(const Spec& s) {
    const std::size_t C = std::size_t(s.channels), K = std::size_t(s.kernel);
    int context = 1;
    std::string blocks = "[";
    for (std::size_t b = 0; b < s.dilations.size(); ++b) {
        const int trim = (s.kernel - 1) * s.dilations[b];
        context += trim;
        blocks += std::string(b ? ", " : "") + "{\"dilation\": " + std::to_string(s.dilations[b]) +
                  ", \"trim\": " + std::to_string(trim) + "}";
    }
    blocks += "]";
    const std::string header =
        "{\"kind\": \"denoiser-model\", \"kind_version\": 1, \"realtime\": true, \"derived\": {\"sample_rate\": 16000, "
        "\"n_fft\": 128, \"hop\": 80, \"long_factor\": 10, \"long_n_fft\": 1280, \"n_freq\": 65, \"n_in\": 130, "
        "\"n_out\": 65, \"input_layout\": [[\"log1p_mag\", 65], [\"log1p_long_mag\", 65]], \"output_layout\": "
        "[[\"log1p_mag\", 65]], \"output\": \"magnitude\", \"out_activation\": \"relu\", \"compress\": null, "
        "\"decompress\": null, \"mask_bound\": null, \"context_frames\": " +
        std::to_string(context) +
        ", \"framing\": \"causal\", \"stream_delay\": 48}, \"net\": {\"type\": \"tcn\", \"channels\": " +
        std::to_string(C) + ", \"kernel\": " + std::to_string(K) + ", \"blocks\": " + blocks +
        ", \"layer_norm_eps\": 1e-05, \"context\": " + std::to_string(context) + "}";

    Random rnd(s.seed);
    auto weights = [&](std::size_t n, std::size_t fanIn) {
        std::vector<float> v(n);
        const float scale = 1.0f / std::sqrt(float(fanIn));
        for (float& x : v)
            x = scale * rnd.uniform();
        return v;
    };
    auto around = [&](std::size_t n, float centre, float spread) {
        std::vector<float> v(n);
        for (float& x : v)
            x = centre + spread * rnd.uniform();
        return v;
    };
    std::vector<Named> t;
    t.push_back({"input.weight", {C, std::size_t(kNIn)}, weights(C * kNIn, kNIn)});
    t.push_back({"input.bias", {C}, around(C, 0.0f, 0.1f)});
    for (std::size_t b = 0; b < s.dilations.size(); ++b) {
        const std::string p = "block" + std::to_string(b) + ".";
        t.push_back({p + "norm.weight", {C}, around(C, 1.0f, 0.2f)});
        t.push_back({p + "norm.bias", {C}, around(C, s.normBiasShift, 0.1f)});
        t.push_back({p + "conv.weight", {C, C, K}, weights(C * C * K, C * K)});
        t.push_back({p + "conv.bias", {C}, around(C, s.convBiasShift, 0.1f)});
        t.push_back({p + "mix.weight", {C, C}, weights(C * C, C)});
        t.push_back({p + "mix.bias", {C}, around(C, 0.0f, 0.1f)});
    }
    t.push_back({"final_norm.weight", {C}, around(C, 1.0f, 0.2f)});
    t.push_back({"final_norm.bias", {C}, around(C, s.normBiasShift, 0.1f)});
    t.push_back({"output.weight", {std::size_t(kNOut), C}, weights(kNOut * C, C)});
    t.push_back({"output.bias", {std::size_t(kNOut)}, around(kNOut, 0.0f, 0.1f)});
    return container(header, t);
}

inline denoiser::ModelFile model(const Spec& s) {
    const std::string bytes = modelBytes(s);
    return denoiser::ModelFile::fromMemory(bytes.data(), bytes.size());
}

}  // namespace tcn_synth
