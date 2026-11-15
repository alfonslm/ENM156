#pragma once
// The container of model and vectors files (format: plugin/tools/README.md,
// "Container"): magic "ENMTENSR", version 1, a JSON header and float32
// tensors. Loading allocates and may throw; the tensors are then plain
// read-only memory, safe to read from any thread (also the audio thread).
//
//   auto file = denoiser::TensorFile::load("model.bin");     // throws FormatError
//   const denoiser::Tensor& w = file.get("linear0.weight", {24, 260});
//   float w_ij = w.data[i * w.shape[1] + j];                   // row-major
//
// Every malformed or truncated file gives a FormatError (never undefined
// behaviour): magic, version, header length, JSON, the tensor list, dtypes,
// shapes and offsets are all checked against the file size.

#include "denoiser/format_error.h"
#include "denoiser/json.h"

#include <cstddef>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace denoiser {

struct Tensor {
    std::string name;
    std::vector<std::size_t> shape;  // row-major, last index fastest; {} = scalar
    const float* data = nullptr;     // size() floats, owned by the TensorFile
    std::size_t size() const noexcept;  // product of shape (1 for a scalar)
};

class TensorFile {
public:
    // Reads a whole file. Throws FormatError (also if it can't be opened).
    static TensorFile load(const std::string& path);
    // Parses bytes already in memory (they are copied)
    static TensorFile fromMemory(const void* bytes, std::size_t size);

    TensorFile(TensorFile&&) noexcept = default;
    TensorFile& operator=(TensorFile&&) noexcept = default;
    TensorFile(const TensorFile&) = delete;
    TensorFile& operator=(const TensorFile&) = delete;

    // The whole JSON header (an object), with "kind", "tensors", ...
    const Json& header() const noexcept { return header_; }
    // The header's "kind", e.g. "denoiser-model"
    const std::string& kind() const { return header_.at("kind").string(); }

    // All tensors, in storage order
    const std::vector<Tensor>& tensors() const noexcept { return tensors_; }
    // A tensor by name, or nullptr
    const Tensor* find(std::string_view name) const noexcept;
    // A tensor by name; throws FormatError if absent
    const Tensor& get(std::string_view name) const;
    // ... and checks its shape; throws FormatError if it differs
    const Tensor& get(std::string_view name, std::initializer_list<std::size_t> shape) const;

private:
    TensorFile() = default;

    std::vector<float> storage_;  // the whole file, float-aligned; Tensor::data points in here
    Json header_;
    std::vector<Tensor> tensors_;
};

// "[24, 260]": for messages
std::string shapeString(const std::vector<std::size_t>& shape);

}  // namespace denoiser
