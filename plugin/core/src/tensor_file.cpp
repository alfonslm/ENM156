#include "denoiser/tensor_file.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>

namespace denoiser {

namespace {

constexpr char kMagic[8] = {'E', 'N', 'M', 'T', 'E', 'N', 'S', 'R'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kPreamble = 16;  // magic, version, header length

std::uint32_t readU32(const unsigned char* p) {
    return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 | std::uint32_t(p[2]) << 16 | std::uint32_t(p[3]) << 24;
}

bool littleEndian() {
    const std::uint32_t one = 1;
    unsigned char first = 0;
    std::memcpy(&first, &one, 1);
    return first == 1;
}

}  // namespace

std::size_t Tensor::size() const noexcept {
    std::size_t n = 1;
    for (std::size_t d : shape)
        n *= d;
    return n;
}

std::string shapeString(const std::vector<std::size_t>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i)
        s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "]";
}

TensorFile TensorFile::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw FormatError("can't open " + path);
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0)
        throw FormatError("can't read " + path);
    in.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    if (size > 0 && !in.read(bytes.data(), size))
        throw FormatError("can't read " + path);
    try {
        return fromMemory(bytes.data(), bytes.size());
    } catch (const FormatError& e) {
        throw FormatError(path + ": " + e.what());
    }
}

TensorFile TensorFile::fromMemory(const void* bytes, std::size_t size) {
    if (!littleEndian())
        throw FormatError("tensor files are little-endian; big-endian machines are not supported");
    const auto* b = static_cast<const unsigned char*>(bytes);
    if (size < kPreamble || std::memcmp(b, kMagic, sizeof kMagic) != 0)
        throw FormatError("not a denoiser tensor file (no ENMTENSR magic)");
    const std::uint32_t version = readU32(b + 8);
    if (version != kVersion)
        throw FormatError("container version " + std::to_string(version) + " is not supported (expected 1)");
    const std::uint32_t headerLen = readU32(b + 12);
    if (headerLen > size - kPreamble)
        throw FormatError("truncated: header of " + std::to_string(headerLen) + " bytes, file has " +
                          std::to_string(size) + " bytes");
    const std::size_t dataStart = kPreamble + headerLen;
    if (dataStart % 64 != 0)
        throw FormatError("data section does not start on a 64-byte boundary");

    TensorFile file;
    // Copy into float storage so that every tensor (offsets are multiples of
    // 64 from a 64-aligned start) is float-aligned
    file.storage_.resize((size + sizeof(float) - 1) / sizeof(float));
    std::memcpy(file.storage_.data(), b, size);
    const auto* base = reinterpret_cast<const unsigned char*>(file.storage_.data());

    try {
        file.header_ = Json::parse(std::string_view(reinterpret_cast<const char*>(base + kPreamble), headerLen));
    } catch (const FormatError& e) {
        throw FormatError(std::string("header: ") + e.what());
    }
    if (!file.header_.isObject())
        throw FormatError("header: not a JSON object");

    const std::size_t dataSize = size - dataStart;
    const Json* list = file.header_.find("tensors");
    if (!list || !list->isArray())
        throw FormatError("header: \"tensors\" missing or not a list");
    for (const Json& entry : list->array()) {
        Tensor t;
        try {
            t.name = entry.at("name").string();
            const std::string& dtype = entry.at("dtype").string();
            if (dtype != "f32")
                throw FormatError("dtype \"" + dtype + "\" is not supported (only f32)");
            // Every element count is bounded by the data size, so the
            // products below cannot overflow
            const std::size_t maxCount = dataSize / sizeof(float);
            std::size_t count = 1;
            for (const Json& dim : entry.at("shape").array()) {
                const auto d = static_cast<std::size_t>(dim.integer(0, std::numeric_limits<std::int32_t>::max()));
                t.shape.push_back(d);
                if (d != 0 && count > maxCount / d)
                    throw FormatError("shape " + shapeString(t.shape) + "... is larger than the file");
                count *= d;
            }
            const auto offset = static_cast<std::size_t>(entry.at("offset").integer(0, std::numeric_limits<std::int64_t>::max() / 2));
            if (offset % 64 != 0)
                throw FormatError("offset " + std::to_string(offset) + " is not a multiple of 64");
            if (offset > dataSize || count * sizeof(float) > dataSize - offset)
                throw FormatError("truncated: data ends at byte " + std::to_string(size) + ", tensor needs up to byte " +
                                  std::to_string(dataStart + offset + count * sizeof(float)));
            t.data = reinterpret_cast<const float*>(base + dataStart + offset);
        } catch (const FormatError& e) {
            throw FormatError("tensor " + std::to_string(file.tensors_.size()) +
                              (t.name.empty() ? "" : " \"" + t.name + "\"") + ": " + e.what());
        }
        if (file.find(t.name))
            throw FormatError("tensor \"" + t.name + "\" appears twice");
        file.tensors_.push_back(std::move(t));
    }
    return file;
}

const Tensor* TensorFile::find(std::string_view name) const noexcept {
    for (const Tensor& t : tensors_)
        if (t.name == name)
            return &t;
    return nullptr;
}

const Tensor& TensorFile::get(std::string_view name) const {
    if (const Tensor* t = find(name))
        return *t;
    throw FormatError("tensor \"" + std::string(name) + "\" is missing");
}

const Tensor& TensorFile::get(std::string_view name, std::initializer_list<std::size_t> shape) const {
    const Tensor& t = get(name);
    if (t.shape != std::vector<std::size_t>(shape))
        throw FormatError("tensor \"" + t.name + "\" has shape " + shapeString(t.shape) + ", expected " +
                          shapeString(std::vector<std::size_t>(shape)));
    return t;
}

}  // namespace denoiser
