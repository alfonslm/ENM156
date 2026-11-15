#include "denoiser/model_file.h"

#include <cmath>
#include <limits>

namespace denoiser {

namespace {

// Sizes far beyond any real model, small enough that products of them can't overflow an int
constexpr long long kMaxSize = 1 << 20;

[[noreturn]] void bad(const std::string& key, const std::string& what) {
    throw FormatError("model header: " + key + ": " + what);
}

// Runs f(), prefixing any FormatError with the key
template <class F>
auto field(const std::string& key, F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const FormatError& e) {
        bad(key, e.what());
    }
}

int intField(const Json& obj, const char* section, const char* key, long long min, long long max) {
    return field(std::string(section) + "." + key, [&] { return static_cast<int>(obj.at(key).integer(min, max)); });
}

std::string stringField(const Json& obj, const char* section, const char* key) {
    return field(std::string(section) + "." + key, [&] { return obj.at(key).string(); });
}

// A number or null (null -> fallback)
double numberOrNull(const Json& obj, const char* key, double fallback) {
    return field(std::string("derived.") + key, [&] {
        const Json& v = obj.at(key);
        return v.isNull() ? fallback : v.number();
    });
}

InputPart inputPart(const std::string& name) {
    if (name == "log1p_mag") return InputPart::Log1pMag;
    if (name == "compressed_real") return InputPart::CompressedReal;
    if (name == "compressed_imag") return InputPart::CompressedImag;
    if (name == "log1p_long_mag") return InputPart::Log1pLongMag;
    bad("derived.input_layout", "unknown part \"" + name + "\"");
}

std::vector<std::pair<std::string, int>> layout(const Json& d, const char* key) {
    return field(std::string("derived.") + key, [&] {
        std::vector<std::pair<std::string, int>> parts;
        for (const Json& p : d.at(key).array()) {
            const auto& a = p.array();
            if (a.size() != 2)
                throw FormatError("each part must be [name, size]");
            parts.emplace_back(a[0].string(), static_cast<int>(a[1].integer(0, kMaxSize)));
        }
        return parts;
    });
}

}  // namespace

ModelInfo parseModelInfo(const Json& header) {
    if (!header.isObject())
        throw FormatError("model header: not a JSON object");
    const std::string kind = stringField(header, "header", "kind");
    if (kind != "denoiser-model")
        bad("kind", "\"" + kind + "\" is not a model file (expected \"denoiser-model\")");
    const int kindVersion = intField(header, "header", "kind_version", 0, kMaxSize);
    if (kindVersion != 1)
        bad("kind_version", std::to_string(kindVersion) + " is not supported (expected 1)");

    ModelInfo m;
    m.realtime = field("realtime", [&] { return header.at("realtime").boolean(); });
    const Json* derived = header.find("derived");
    if (!derived || !derived->isObject())
        bad("derived", "missing or not an object");
    const Json& d = *derived;
    m.netType = field("net.type", [&] { return header.at("net").at("type").string(); });

    m.sampleRate = intField(d, "derived", "sample_rate", 1, 10'000'000);
    m.nFft = intField(d, "derived", "n_fft", 2, kMaxSize);
    if (m.nFft % 2 != 0)
        bad("derived.n_fft", "must be even, got " + std::to_string(m.nFft));
    m.hop = intField(d, "derived", "hop", 1, m.nFft);
    m.longFactor = intField(d, "derived", "long_factor", 0, kMaxSize / m.nFft);
    m.longNFft = intField(d, "derived", "long_n_fft", 0, kMaxSize);
    if (m.longNFft != m.nFft * m.longFactor)
        bad("derived.long_n_fft", "must be n_fft * long_factor = " + std::to_string(m.nFft * m.longFactor));
    m.nFreq = intField(d, "derived", "n_freq", 1, kMaxSize);
    if (m.nFreq != m.nFft / 2 + 1)
        bad("derived.n_freq", "must be n_fft / 2 + 1 = " + std::to_string(m.nFft / 2 + 1));
    m.nIn = intField(d, "derived", "n_in", 1, kMaxSize);
    m.nOut = intField(d, "derived", "n_out", 1, kMaxSize);

    const std::string output = stringField(d, "derived", "output");
    if (output == "magnitude") m.output = OutputKind::Magnitude;
    else if (output == "mask") m.output = OutputKind::Mask;
    else if (output == "complex_mask") m.output = OutputKind::ComplexMask;
    else bad("derived.output", "unknown output \"" + output + "\"");
    const bool complex = m.output == OutputKind::ComplexMask;

    const std::string act = stringField(d, "derived", "out_activation");
    if (act == "none") m.outActivation = Activation::None;
    else if (act == "relu") m.outActivation = Activation::Relu;
    else if (act == "sigmoid") m.outActivation = Activation::Sigmoid;
    else bad("derived.out_activation", "unknown activation \"" + act + "\"");

    // Input layout: known parts, each n_freq values, each at most once, summing to n_in
    int total = 0;
    bool seen[4] = {false, false, false, false};
    for (const auto& [name, size] : layout(d, "input_layout")) {
        const InputPart part = inputPart(name);
        if (size != m.nFreq)
            bad("derived.input_layout", "part \"" + name + "\" must have n_freq values");
        if (seen[static_cast<int>(part)])
            bad("derived.input_layout", "part \"" + name + "\" appears twice");
        seen[static_cast<int>(part)] = true;
        m.inputLayout.emplace_back(part, size);
        total += size;
    }
    if (total != m.nIn)
        bad("derived.input_layout", "sizes add up to " + std::to_string(total) + ", n_in is " + std::to_string(m.nIn));
    const bool hasCompressed = seen[static_cast<int>(InputPart::CompressedReal)] && seen[static_cast<int>(InputPart::CompressedImag)];
    if (complex != hasCompressed)
        bad("derived.input_layout", complex ? "complex_mask needs compressed_real and compressed_imag"
                                            : "compressed parts are only for complex_mask");
    if (!complex && !seen[static_cast<int>(InputPart::Log1pMag)])
        bad("derived.input_layout", "needs log1p_mag");
    if ((m.longFactor > 0) != seen[static_cast<int>(InputPart::Log1pLongMag)])
        bad("derived.input_layout", "log1p_long_mag must be present exactly when long_factor > 0");

    // Output layout
    const auto out = layout(d, "output_layout");
    std::vector<std::pair<std::string, int>> expected;
    if (m.output == OutputKind::Magnitude) expected = {{"log1p_mag", m.nFreq}};
    else if (m.output == OutputKind::Mask) expected = {{"gain", m.nFreq}};
    else expected = {{"mask_real", m.nFreq}, {"mask_imag", m.nFreq}};
    if (out != expected)
        bad("derived.output_layout", "does not match output \"" + output + "\"");
    if (m.nOut != (complex ? 2 : 1) * m.nFreq)
        bad("derived.n_out", "does not match output \"" + output + "\"");

    if (complex) {
        const double c = numberOrNull(d, "compress", 0.0);
        if (!(c > 0.0))
            bad("derived.compress", "must be > 0 for complex_mask");
        const double dc = numberOrNull(d, "decompress", 0.0);
        if (!(dc > 0.0))
            bad("derived.decompress", "must be > 0 for complex_mask");
        const double b = numberOrNull(d, "mask_bound", 0.0);
        if (b < 0.0)
            bad("derived.mask_bound", "must be >= 0");
        m.compress = static_cast<float>(c);
        m.decompress = static_cast<float>(dc);
        m.maskBound = static_cast<float>(b);
    }

    m.contextFrames = intField(d, "derived", "context_frames", 1, kMaxSize);
    const std::string framing = stringField(d, "derived", "framing");
    if (framing == "causal") m.framing = Framing::Causal;
    else if (framing == "centred") m.framing = Framing::Centred;
    else bad("derived.framing", "unknown framing \"" + framing + "\"");

    if (m.framing == Framing::Causal) {
        m.streamDelay = intField(d, "derived", "stream_delay", 0, kMaxSize);
        if (m.streamDelay != m.nFft - m.hop)
            bad("derived.stream_delay", "must be n_fft - hop for causal framing");
    } else {
        m.streamDelay = -1;
        if (m.realtime)
            bad("realtime", "a centred model can't be real-time");
    }
    return m;
}

ModelFile::ModelFile(TensorFile file) : file_(std::move(file)), info_(parseModelInfo(file_.header())) {}

ModelFile ModelFile::fromTensorFile(TensorFile file) { return ModelFile(std::move(file)); }

ModelFile ModelFile::load(const std::string& path) {
    TensorFile file = TensorFile::load(path);
    try {
        return ModelFile(std::move(file));
    } catch (const FormatError& e) {
        throw FormatError(path + ": " + e.what());
    }
}

ModelFile ModelFile::fromMemory(const void* bytes, std::size_t size) {
    return ModelFile(TensorFile::fromMemory(bytes, size));
}

}  // namespace denoiser
