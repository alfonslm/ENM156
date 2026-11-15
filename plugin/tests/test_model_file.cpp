// The JSON parser, the container reader and the model header: every golden
// file loads with the right settings and tensors, and malformed, truncated or
// inconsistent files give a FormatError (run under AddressSanitizer to show
// there is no out-of-bounds access on the way).
#include "check.h"
#include "golden.h"

#include "denoiser/json.h"
#include "denoiser/model_file.h"
#include "denoiser/tensor_file.h"

#include <clocale>
#include <cstdint>
#include <cstring>
#include <string>

using namespace denoiser;

namespace {

template <class F>
bool throwsFormatError(F&& f) {
    try {
        f();
    } catch (const FormatError&) {
        return true;
    } catch (...) {
        return false;  // anything else (std::bad_alloc, std::out_of_range, ...) is a bug
    }
    return false;
}

void checkJson() {
    const Json j = Json::parse(
        R"( {"a": [1, -2.5e3, 0, 1E-5, true, false, null], "s": "x\"\\\/\b\f\n\r\tA\u00e9\ud83d\ude00", "o": {}, "e": []} )");
    CHECK(j.isObject());
    const auto& a = j.at("a").array();
    CHECK(a.size() == 7);
    CHECK(a[0].integer() == 1);
    CHECK(a[1].number() == -2500.0);
    CHECK(a[2].integer() == 0);
    CHECK(a[3].number() == 1e-5);
    CHECK(a[4].boolean() && !a[5].boolean() && a[6].isNull());
    CHECK(j.at("s").string() == "x\"\\/\b\f\n\r\tA\xC3\xA9\xF0\x9F\x98\x80");
    CHECK(j.at("o").members().empty() && j.at("e").array().empty());
    CHECK(j.find("missing") == nullptr);
    CHECK(Json::parse("0.3").number() == 0.3);
    CHECK(Json::parse("3.3333333333333335").number() == 3.3333333333333335);

    // Wrong types and missing keys are FormatErrors too
    CHECK(throwsFormatError([&] { j.at("missing"); }));
    CHECK(throwsFormatError([&] { j.at("a").string(); }));
    CHECK(a[1].integer() == -2500);
    CHECK(throwsFormatError([&] { a[3].integer(); }));      // not whole
    CHECK(throwsFormatError([&] { a[0].integer(2, 5); }));  // out of range
    CHECK(throwsFormatError([&] { a[0].find("x"); }));

    const char* bad[] = {"", " ", "{", "}", "[1,]", "[1 2]", "{\"a\":}", "{\"a\" 1}", "{a:1}", "{\"a\":1,}", "01", "1.",
                         ".5", "-", "+1", "1e", "1e+", "NaN", "Infinity", "1e999", "tru", "nul", "\"abc", "\"\\x\"",
                         "\"\\u12\"", "\"\\ud800\"", "\"\\udc00\"", "\"a\nb\"", "[1] 2", "{\"a\":1}}"};
    for (const char* text : bad)
        CHECK(throwsFormatError([&] { Json::parse(text); }));
    // Nesting is limited, so a hostile file can't overflow the stack
    CHECK(throwsFormatError([&] { Json::parse(std::string(100000, '[')); }));
    CHECK(Json::parse(std::string(60, '[') + std::string(60, ']')).isArray());

    // Numbers don't depend on the C locale (a host may set one with a decimal comma)
    for (const char* loc : {"de_DE.UTF-8", "de_DE.utf8", "de_DE", "fr_FR.UTF-8", "German_Germany"}) {
        if (std::setlocale(LC_ALL, loc)) {
            std::printf("checked numbers under locale %s\n", loc);
            CHECK(Json::parse("0.25").number() == 0.25);
            break;
        }
    }
    std::setlocale(LC_ALL, "C");
}

void checkGoldenFiles() {
    const Json index = Json::parse(golden::readText(golden::dir() + "/index.json"));
    int count = 0;
    for (const Json& c : index.at("cases").array()) {
        const std::string name = c.at("name").string();
        const ModelFile model = ModelFile::load(golden::dir() + "/" + c.at("model_file").string());
        const ModelInfo& m = model.info();
        CHECK(m.nFft == c.at("n_fft").integer());
        CHECK(m.hop == c.at("hop").integer());
        CHECK(m.longFactor == c.at("long_factor").integer());
        CHECK(m.longNFft == m.nFft * m.longFactor);
        CHECK(m.nIn == c.at("n_in").integer());
        CHECK(m.nOut == c.at("n_out").integer());
        CHECK(m.contextFrames == c.at("context_frames").integer());
        CHECK(m.realtime == c.at("realtime").boolean());
        CHECK(m.netType == c.at("net").string());
        CHECK(m.sampleRate == 16000);
        CHECK((m.framing == Framing::Causal) == m.realtime);
        CHECK(m.streamDelay == (m.realtime ? m.nFft - m.hop : -1));
        const std::string& out = c.at("output").string();
        CHECK((out == "magnitude") == (m.output == OutputKind::Magnitude));
        CHECK((out == "mask") == (m.output == OutputKind::Mask));
        CHECK((out == "complex_mask") == (m.output == OutputKind::ComplexMask));
        if (m.output == OutputKind::ComplexMask) {
            const Json& d = model.header().at("derived");
            CHECK(m.compress == static_cast<float>(d.at("compress").number()));
            CHECK(m.decompress == static_cast<float>(d.at("decompress").number()));
            CHECK(m.maskBound == static_cast<float>(d.at("mask_bound").number()));
        }

        // The network's tensors are reachable by name with their shapes
        const Json& net = model.net();
        CHECK(net.at("type").string() == m.netType);
        if (m.netType == "mlp") {
            const auto& layers = net.at("layers").array();
            for (std::size_t i = 0; i < layers.size(); ++i) {
                const auto o = std::size_t(layers[i].at("out").integer()), in = std::size_t(layers[i].at("in").integer());
                const std::string p = "linear" + std::to_string(i);
                const Tensor& w = model.tensors().get(p + ".weight", {o, in});
                CHECK(w.size() == o * in);
                CHECK(std::isfinite(w.data[w.size() - 1]));
                model.tensors().get(p + ".bias", {o});
            }
            CHECK(throwsFormatError([&] { model.tensors().get("linear0.weight", {1, 2}); }));
        } else if (m.netType == "tcn") {
            const auto C = std::size_t(net.at("channels").integer());
            model.tensors().get("input.weight", {C, std::size_t(m.nIn)});
            model.tensors().get("output.weight", {std::size_t(m.nOut), C});
        } else {
            CHECK(m.netType == "snn");
            model.tensors().get("layer0.weight");
            model.tensors().get("output.bias", {std::size_t(m.nOut)});
        }
        CHECK(model.tensors().find("no.such.tensor") == nullptr);
        CHECK(throwsFormatError([&] { model.tensors().get("no.such.tensor"); }));

        // A vectors file is a valid container but not a model file
        const TensorFile vectors = TensorFile::load(golden::dir() + "/" + c.at("vectors_file").string());
        CHECK(vectors.kind() == "denoiser-vectors");
        CHECK(vectors.get("input").shape == std::vector<std::size_t>{1600});
        CHECK(throwsFormatError([&] { ModelFile::load(golden::dir() + "/" + c.at("vectors_file").string()); }));
        ++count;
    }
    CHECK(count == 17);
    CHECK(throwsFormatError([] { ModelFile::load(golden::dir() + "/does-not-exist.bin"); }));
}

// A container from a header text and data bytes
std::string build(std::string header, const std::string& data, std::uint32_t version = 1) {
    while ((16 + header.size()) % 64 != 0)
        header += ' ';
    std::string out = "ENMTENSR";
    for (std::uint32_t v : {version, static_cast<std::uint32_t>(header.size())})
        for (int i = 0; i < 4; ++i)
            out += char((v >> (8 * i)) & 0xFF);
    return out + header + data;
}

bool modelFails(const std::string& bytes) {
    return throwsFormatError([&] { ModelFile::fromMemory(bytes.data(), bytes.size()); });
}

// header with the first `from` after `anchor` replaced by `to`; empty if not found
std::string replaced(const std::string& header, const std::string& anchor, const std::string& from, const std::string& to) {
    const std::size_t a = header.find(anchor);
    if (a == std::string::npos)
        return {};
    const std::size_t p = header.find(from, a);
    if (p == std::string::npos)
        return {};
    std::string h = header;
    h.replace(p, from.size(), to);
    return h;
}

void checkMalformed() {
    const std::vector<char> raw = golden::readBytes(golden::dir() + "/mlp_complex_default.model.bin");
    const std::string file(raw.begin(), raw.end());
    std::uint32_t headerLen = 0;
    for (int i = 0; i < 4; ++i)
        headerLen |= std::uint32_t(static_cast<unsigned char>(file[12 + i])) << (8 * i);
    const std::string header = file.substr(16, headerLen);
    const std::string data = file.substr(16 + headerLen);

    // The builder itself reproduces a loadable file
    CHECK(!modelFails(build(header, data)));
    CHECK(!modelFails(file));

    // Every truncation fails (nothing follows the last tensor). All lengths
    // through the header and the first tensors, then a sample of the rest.
    int truncations = 0;
    for (std::size_t n = 0; n < file.size(); n += (n < 16 + headerLen + 512 ? 1 : 61)) {
        CHECK(modelFails(file.substr(0, n)));
        ++truncations;
    }
    CHECK(modelFails(file.substr(0, file.size() - 1)));
    std::printf("checked %d truncated files\n", truncations);

    // Container level
    std::string f = file;
    f[0] = 'X';
    CHECK(modelFails(f));
    CHECK(modelFails(build(header, data, 2)));  // version
    f = file;
    f[12] = f[13] = f[14] = f[15] = char(0xFF);  // header length past the end
    CHECK(modelFails(f));
    f = file;
    f[12] = char(static_cast<unsigned char>(f[12]) - 1);  // data start not 64-aligned
    CHECK(modelFails(f));

    // Header mutations: each one alone must be refused
    struct Mutation {
        const char *anchor, *from, *to;
    };
    const Mutation mutations[] = {
        {"{", "{", "["},                                          // not an object / broken JSON
        {"\"kind\"", "denoiser-model", "denoiser-vectors"},       // wrong kind
        {"\"kind_version\"", "1", "2"},                            // unknown kind_version
        {"\"realtime\"", "true", "1"},                             // wrong type
        {"\"derived\"", "\"derived\"", "\"derivative\""},          // missing derived
        {"\"derived\"", "\"n_fft\": 128", "\"n_fft\": 127"},       // odd n_fft
        {"\"derived\"", "\"n_fft\": 128", "\"n_fft\": \"128\""},   // string
        {"\"derived\"", "\"hop\": 80", "\"hop\": 0"},
        {"\"derived\"", "\"hop\": 80", "\"hop\": 200"},           // hop > n_fft
        {"\"derived\"", "\"hop\": 80", "\"hop\": 80.5"},
        {"\"derived\"", "\"long_n_fft\": 1280", "\"long_n_fft\": 1281"},
        {"\"derived\"", "\"n_freq\": 65", "\"n_freq\": 64"},
        {"\"derived\"", "\"n_in\": 260", "\"n_in\": 261"},
        {"\"derived\"", "\"n_out\": 130", "\"n_out\": 65"},
        {"\"derived\"", "\"compressed_real\"", "\"compressed_rea1\""},  // unknown part
        {"\"derived\"", "\"log1p_long_mag\"", "\"log1p_mag\""},          // part twice
        {"\"derived\"", "\"mask_real\"", "\"gain\""},
        {"\"derived\"", "\"output\": \"complex_mask\"", "\"output\": \"mask\""},
        {"\"derived\"", "\"output\": \"complex_mask\"", "\"output\": \"phase\""},
        {"\"derived\"", "\"out_activation\": \"none\"", "\"out_activation\": \"tanh\""},
        {"\"derived\"", "\"compress\": 0.3", "\"compress\": null"},
        {"\"derived\"", "\"compress\": 0.3", "\"compress\": -0.3"},
        {"\"derived\"", "\"mask_bound\": 1.0", "\"mask_bound\": -1.0"},
        {"\"derived\"", "\"framing\": \"causal\"", "\"framing\": \"sideways\""},
        {"\"derived\"", "\"framing\": \"causal\"", "\"framing\": \"centred\""},  // centred but realtime
        {"\"derived\"", "\"stream_delay\": 48", "\"stream_delay\": 47"},
        {"\"derived\"", "\"context_frames\": 1", "\"context_frames\": 0"},
        {"\"net\"", "\"type\"", "\"kind\""},
        {"\"tensors\"", "\"tensors\"", "\"tensorz\""},
        {"\"tensors\"", "\"dtype\": \"f32\"", "\"dtype\": \"f16\""},
        {"\"tensors\"", "\"offset\": 0", "\"offset\": 4"},           // not a multiple of 64
        {"\"tensors\"", "\"offset\": 0", "\"offset\": -64"},
        {"\"tensors\"", "\"offset\": 0", "\"offset\": 1e300"},
        {"\"tensors\"", "\"offset\": 35008", "\"offset\": 35072"},   // last tensor past the end
        {"\"tensors\"", "24", "2400000000"},                         // dimension too large
        {"\"tensors\"", "24", "240000"},                             // larger than the file
        {"\"tensors\"", "24", "-24"},
        {"\"tensors\"", "24", "24.5"},
        {"\"tensors\"", "\"linear0.bias\"", "\"linear0.weight\""},   // a name twice
        {"\"tensors\"", "\"name\": \"linear0.weight\"", "\"name\": 7"},
    };
    int n = 0;
    for (const Mutation& m : mutations) {
        const std::string h = replaced(header, m.anchor, m.from, m.to);
        CHECK(!h.empty());
        if (h.empty()) {
            std::printf("  mutation not applied: %s -> %s\n", m.from, m.to);
            continue;
        }
        const bool fails = modelFails(build(h, data));
        if (!fails)
            std::printf("  accepted: %s -> %s\n", m.from, m.to);
        CHECK(fails);
        ++n;
    }
    std::printf("checked %d malformed headers\n", n);

    // Random corruption of the preamble and header: either still a valid
    // model (e.g. a changed digit in a weight-free place) or a FormatError,
    // never another exception or a crash
    std::uint32_t seed = 12345;
    int accepted = 0, refused = 0, other = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        f = file;
        for (int flips = 1 + trial % 3; flips > 0; --flips) {
            seed = seed * 1664525u + 1013904223u;
            const std::size_t pos = (seed >> 8) % (16 + headerLen);
            seed = seed * 1664525u + 1013904223u;
            f[pos] = char(seed >> 24);
        }
        try {
            ModelFile::fromMemory(f.data(), f.size());
            ++accepted;
        } catch (const FormatError&) {
            ++refused;
        } catch (...) {
            ++other;
        }
    }
    std::printf("random header corruption: %d accepted, %d refused, %d other exceptions\n", accepted, refused, other);
    CHECK(other == 0);

    // The error says what is wrong
    try {
        ModelFile::fromMemory(file.data(), file.size() - 100);
        CHECK(false);
    } catch (const FormatError& e) {
        std::printf("example message: %s\n", e.what());
        CHECK(std::strstr(e.what(), "truncated") != nullptr);
    }
    try {
        const std::string h = replaced(header, "\"derived\"", "\"n_freq\": 65", "\"n_freq\": 64");
        ModelFile::fromMemory(build(h, data).data(), build(h, data).size());
        CHECK(false);
    } catch (const FormatError& e) {
        std::printf("example message: %s\n", e.what());
        CHECK(std::strstr(e.what(), "derived.n_freq") != nullptr);
    }
}

}  // namespace

int main(int argc, char** argv) {
    golden::init(argc, argv);
    checkJson();
    checkGoldenFiles();
    checkMalformed();
    return check::finish();
}
