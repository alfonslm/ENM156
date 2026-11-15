// The STFT front and back end (features.h, stream.h, offline.h) against every
// golden case in tests/golden/: model input frames, intermediate spectra,
// streamed and offline output with the golden network output fed in; plus
// the analysis/synthesis round trip without a network and reset().
//
// Tolerances (tools/README.md, "Tolerances"; error = max |c++ - golden| / max(1, max |golden|)):
//   x, spec, long_mag, out_spec, streamed after the pre-roll, offline: 1e-5
//   streamed pre-roll (first n_fft - hop samples): 1e-4
//   windows: 1.2e-7 absolute (PyTorch's vectorised cos vs the C library's)
#include "check.h"
#include "golden.h"

#include "denoiser/features.h"
#include "denoiser/offline.h"
#include "denoiser/stream.h"

#include <cstdint>
#include <cstring>

using namespace denoiser;

namespace {

constexpr double kTol = 1e-5;
constexpr double kPreRollTol = 1e-4;
constexpr double kWindowTol = 1.2e-7;

// The (re..., im...) layout of the vectors file's spec / out_spec rows
void splitComplex(const std::complex<float>* z, int n, float* out) {
    for (int k = 0; k < n; ++k) {
        out[k] = z[k].real();
        out[n + k] = z[k].imag();
    }
}

struct Errors {
    golden::ErrorTracker x, spec, longMag, outSpec, streamed, preRoll, offline;
};

void printErr(const char* label, const golden::ErrorTracker& e, bool present) {
    if (present)
        std::printf(" %s %.1e", label, e.worst);
    else
        std::printf(" %s    -   ", label);
}

// The stream as tools/vectors.py ran it: input, then input_zeros zeros, in `chunks` hops
void checkStream(const golden::Case& c, Errors& err) {
    const ModelInfo& info = c.info();
    const int hop = info.hop, nIn = info.nIn, nOut = info.nOut, F = info.nFreq;
    const Tensor& in = c.get("input");
    const Tensor& x = c.get("x");
    const Tensor& pred = c.get("pred");
    const Tensor& streamed = c.get("streamed");
    const Tensor* spec = c.find("spec");
    const Tensor* longMag = c.find("long_mag");
    const Tensor* outSpec = c.find("out_spec");
    const auto chunks = static_cast<std::size_t>(c.vectors.header().at("stream").at("chunks").integer());
    CHECK(c.vectors.header().at("stream").at("delay").integer() == info.streamDelay);
    CHECK(streamed.size() == chunks * std::size_t(hop));

    StreamProcessor stream(info);
    CHECK(stream.delay() == info.nFft - hop);
    CHECK(stream.latency() == info.nFft);
    std::vector<float> chunk(hop), out(hop), row(2 * F);
    int preRollTotal = 0;
    for (std::size_t k = 0; k < chunks; ++k) {
        for (int i = 0; i < hop; ++i) {
            const std::size_t s = k * hop + i;
            chunk[i] = s < in.size() ? in.data[s] : 0.0f;
        }
        err.x.add(stream.analyse(chunk.data()), x.data + k * nIn, nIn);
        if (spec) {
            splitComplex(stream.codec().spectrum(), F, row.data());
            err.spec.add(row.data(), spec->data + k * 2 * F, 2 * F);
        }
        if (longMag)
            err.longMag.add(stream.codec().longMagnitude(), longMag->data + k * F, F);
        const int preRoll = stream.synthesise(pred.data + k * nOut, out.data());
        if (outSpec) {
            splitComplex(stream.codec().outSpectrum(), F, row.data());
            err.outSpec.add(row.data(), outSpec->data + k * 2 * F, 2 * F);
        }
        // pre-roll: the samples with stream index < delay
        int expectPreRoll = 0;
        for (int i = 0; i < hop; ++i) {
            const std::size_t s = k * hop + i;
            if (s < std::size_t(stream.delay())) {
                ++expectPreRoll;
                err.preRoll.add(&out[i], streamed.data + s, 1);
            } else {
                err.streamed.add(&out[i], streamed.data + s, 1);
            }
        }
        CHECK(preRoll == expectPreRoll);
        preRollTotal += preRoll;
    }
    CHECK(preRollTotal == stream.delay());
    CHECK(stream.samplesOut() == std::int64_t(chunks * hop));
}

std::vector<float> reflectPad(const Tensor& in, std::size_t p) {
    const auto n = static_cast<std::ptrdiff_t>(in.size());
    std::vector<float> out(in.size() + 2 * p);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::ptrdiff_t j = std::ptrdiff_t(i) - std::ptrdiff_t(p);
        j = j < 0 ? -j : j;
        j = j >= n ? 2 * (n - 1) - j : j;
        out[i] = in.data[j];
    }
    return out;
}

// Centred cases: spec, long_mag, out_spec frame by frame (denoiseClip hides them)
void checkCentredFrames(const golden::Case& c, Errors& err) {
    const ModelInfo& info = c.info();
    const int F = info.nFreq, hop = info.hop;
    const Tensor& in = c.get("input");
    const Tensor& pred = c.get("pred");
    const Tensor* spec = c.find("spec");
    const Tensor* longMag = c.find("long_mag");
    const Tensor* outSpec = c.find("out_spec");
    const std::vector<float> shortPadded = reflectPad(in, std::size_t(info.nFft / 2));
    const std::vector<float> longPadded = info.longFactor ? reflectPad(in, std::size_t(info.longNFft / 2)) : std::vector<float>();
    FrameCodec codec(info);
    std::vector<float> row(2 * F);
    const std::size_t frames = clipFrames(info, in.size());
    CHECK(frames == pred.shape[0]);
    for (std::size_t t = 0; t < frames; ++t) {
        codec.analyse(shortPadded.data() + t * hop, info.longFactor ? longPadded.data() + t * hop : nullptr);
        if (spec) {
            splitComplex(codec.spectrum(), F, row.data());
            err.spec.add(row.data(), spec->data + t * 2 * F, 2 * F);
        }
        if (longMag)
            err.longMag.add(codec.longMagnitude(), longMag->data + t * F, F);
        codec.synthesise(pred.data + t * info.nOut);
        if (outSpec) {
            splitComplex(codec.outSpectrum(), F, row.data());
            err.outSpec.add(row.data(), outSpec->data + t * 2 * F, 2 * F);
        }
    }
}

// Whole clip, either framing: x of every frame and the output
void checkOffline(const golden::Case& c, Errors& err) {
    const ModelInfo& info = c.info();
    const Tensor& in = c.get("input");
    const Tensor& x = c.get("x");
    const Tensor& offline = c.get("offline");
    golden::ReplayNetwork net(c.get("pred"));
    net.nIn = std::size_t(info.nIn);
    const std::vector<float> y = denoiseClip(info, net, in.data, in.size());
    CHECK(y.size() == offline.size());
    CHECK(net.rows() == x.shape[0]);
    CHECK(clipFrames(info, in.size()) == x.shape[0]);
    for (std::size_t t = 0; t < net.inputs().size() && t < x.shape[0]; ++t)
        err.x.add(net.inputs()[t].data(), x.data + t * info.nIn, std::size_t(info.nIn));
    err.offline.add(y.data(), offline.data, std::min(y.size(), offline.size()));
}

// A network that makes the output spectrum (nearly) the noisy one; not
// possible for complex_mask with a bound (tanh never reaches 1 for bound <= 1)
class PassThroughNetwork : public FrameNetwork {
public:
    explicit PassThroughNetwork(const ModelInfo& info) : info_(info) {
        int offset = 0;
        for (const auto& part : info.inputLayout) {
            if (part.first == InputPart::Log1pMag)
                magOffset_ = offset;
            offset += part.second;
        }
    }
    void reset() noexcept override {}
    void process(const float* x, float* y) noexcept override {
        const int F = info_.nFreq;
        switch (info_.output) {
            case OutputKind::Magnitude: std::copy(x + magOffset_, x + magOffset_ + F, y); break;  // expm1(log1p(|s|))
            case OutputKind::Mask: std::fill(y, y + F, 1.0f); break;
            case OutputKind::ComplexMask:
                std::fill(y, y + F, 1.0f);
                std::fill(y + F, y + 2 * F, 0.0f);
                break;
        }
    }

private:
    const ModelInfo& info_;
    int magOffset_ = 0;
};

// Deterministic noise in [-0.5, 0.5)
std::vector<float> noise(std::size_t n, std::uint32_t seed) {
    std::vector<float> v(n);
    for (float& s : v) {
        seed = seed * 1664525u + 1013904223u;
        s = float(seed >> 8) / float(1u << 24) - 0.5f;
    }
    return v;
}

// "reconstruction" of tests/test_stream.py: without a network, the stream's
// output is its input delayed by n_fft - hop. Returns the max abs error.
double passThroughError(const ModelInfo& info, const std::vector<float>& signal, FrameNetwork* network) {
    StreamProcessor stream(info);
    const std::size_t hop = std::size_t(info.hop), delay = std::size_t(stream.delay());
    const std::size_t chunks = (signal.size() + delay + hop - 1) / hop;
    std::vector<float> chunk(hop), out(hop), pred(std::size_t(info.nOut));
    double worst = 0.0;
    for (std::size_t k = 0; k < chunks; ++k) {
        for (std::size_t i = 0; i < hop; ++i)
            chunk[i] = k * hop + i < signal.size() ? signal[k * hop + i] : 0.0f;
        if (network) {
            stream.process(chunk.data(), out.data(), *network);
        } else {
            stream.analyse(chunk.data());
            stream.synthesisePassThrough(out.data());
        }
        for (std::size_t i = 0; i < hop; ++i) {
            const std::size_t s = k * hop + i;
            if (s >= delay && s - delay < signal.size())
                worst = std::max(worst, std::fabs(double(out[i]) - double(signal[s - delay])));
        }
    }
    return worst;
}

void checkPassThrough(const golden::Case& c) {
    const ModelInfo& info = c.info();
    const Tensor& in = c.get("input");
    const std::vector<float> clip(in.data, in.data + in.size());
    const std::vector<float> rnd = noise(16000, 7);
    const double specErr = std::max(passThroughError(info, clip, nullptr), passThroughError(info, rnd, nullptr));
    CHECK(specErr <= 1e-6);
    std::printf("  pass-through: spectrum %.1e", specErr);
    if (info.output != OutputKind::ComplexMask || info.maskBound == 0.0f) {
        PassThroughNetwork net(info);
        const double netErr = std::max(passThroughError(info, clip, &net), passThroughError(info, rnd, &net));
        CHECK(netErr <= kTol);
        std::printf(", through output_spec %.1e", netErr);
    }
    std::printf("\n");
}

// reset() restores the just-constructed state: after other audio, a reset
// stream gives bit-identical output to a fresh one
void checkReset(const golden::Case& c) {
    const ModelInfo& info = c.info();
    const Tensor& in = c.get("input");
    const std::size_t hop = std::size_t(info.hop);
    const std::size_t chunks = in.size() / hop;
    golden::ReplayNetwork net(c.get("pred"));
    net.nIn = std::size_t(info.nIn);

    auto run = [&](StreamProcessor& s) {
        std::vector<float> out(chunks * hop);
        net.reset();
        for (std::size_t k = 0; k < chunks; ++k)
            s.process(in.data + k * hop, out.data() + k * hop, net);
        return out;
    };
    StreamProcessor fresh(info);
    const std::vector<float> first = run(fresh);

    StreamProcessor used(info);
    const std::vector<float> junk = noise(chunks * hop, 3);
    std::vector<float> out(hop);
    for (std::size_t k = 0; k < chunks; ++k)
        used.process(junk.data() + k * hop, out.data(), net);
    used.reset();
    CHECK(used.samplesOut() == 0);
    const std::vector<float> second = run(used);
    CHECK(std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) == 0);
    fresh.reset();
    const std::vector<float> third = run(fresh);
    CHECK(std::memcmp(first.data(), third.data(), first.size() * sizeof(float)) == 0);
}

// denoiseClip: short clips, progress reports and cancelling
void checkClipEdges(const golden::Case& c) {
    const ModelInfo& info = c.info();
    const Tensor& in = c.get("input");
    golden::ReplayNetwork net(c.get("pred"));
    net.nIn = std::size_t(info.nIn);
    if (info.framing == Framing::Causal) {
        CHECK(minClipSamples(info) == 0);
        CHECK(denoiseClip(info, net, in.data, 0).empty());
        CHECK(denoiseClip(info, net, in.data, 1).size() == 1);
    } else {
        CHECK(minClipSamples(info) == std::size_t(info.widestWindow() / 2 + 1));
        bool refused = false;
        try {
            denoiseClip(info, net, in.data, minClipSamples(info) - 1);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        CHECK(refused);
        CHECK(denoiseClip(info, net, in.data, minClipSamples(info)).size() == minClipSamples(info));
    }
    std::size_t calls = 0, lastDone = 0, total = 0;
    const std::vector<float> y = denoiseClip(info, net, in.data, in.size(), [&](std::size_t done, std::size_t all) {
        ++calls;
        lastDone = done;
        total = all;
        return true;
    });
    CHECK(y.size() == in.size());
    CHECK(calls == clipFrames(info, in.size()) && lastDone == total && total == calls);
    const std::vector<float> cancelled =
        denoiseClip(info, net, in.data, in.size(), [](std::size_t done, std::size_t) { return done < 3; });
    CHECK(cancelled.empty());
    CHECK(net.rows() == 3);
}

void checkWindows(const golden::Case& c) {
    const ModelInfo& info = c.info();
    FrameCodec codec(info);
    if (const Tensor* w = c.find("window")) {
        CHECK(w->size() == std::size_t(info.nFft));
        for (std::size_t i = 0; i < w->size(); ++i)
            CHECK_NEAR(codec.window()[i], w->data[i], kWindowTol);
    }
    if (const Tensor* w = c.find("long_window")) {
        CHECK(w->size() == std::size_t(info.longNFft));
        for (std::size_t i = 0; i < w->size(); ++i)
            CHECK_NEAR(codec.longWindow()[i], w->data[i], kWindowTol);
    }
}

}  // namespace

int main(int argc, char** argv) {
    golden::init(argc, argv);
    const std::vector<std::string> names = golden::caseNames();
    CHECK(names.size() == 17);
    std::printf("worst error per tensor, max|c++ - golden| / max(1, max|golden|)\n");
    for (const std::string& name : names) {
        const golden::Case c = golden::load(name);
        const ModelInfo& info = c.info();
        Errors err;
        checkWindows(c);
        if (info.framing == Framing::Causal) {
            CHECK(c.find("streamed") != nullptr);
            checkStream(c, err);
        } else {
            CHECK(c.find("streamed") == nullptr);
            bool refused = false;
            try {
                StreamProcessor s(info);
            } catch (const std::invalid_argument&) {
                refused = true;
            }
            CHECK(refused);
            checkCentredFrames(c, err);
        }
        checkOffline(c, err);

        const bool causal = info.framing == Framing::Causal;
        std::printf("%-28s", name.c_str());
        printErr("x", err.x, true);
        printErr("spec", err.spec, c.find("spec"));
        printErr("long_mag", err.longMag, c.find("long_mag"));
        printErr("out_spec", err.outSpec, c.find("out_spec"));
        printErr("streamed", err.streamed, causal);
        printErr("pre-roll", err.preRoll, causal);
        printErr("offline", err.offline, true);
        std::printf("\n");
        CHECK(err.x.worst <= kTol);
        CHECK(err.spec.worst <= kTol);
        CHECK(err.longMag.worst <= kTol);
        CHECK(err.outSpec.worst <= kTol);
        CHECK(err.streamed.worst <= kTol);
        CHECK(err.preRoll.worst <= kPreRollTol);
        CHECK(err.offline.worst <= kTol);

        if (causal) {
            checkPassThrough(c);
            checkReset(c);
        }
        checkClipEdges(c);
    }
    return check::finish();
}
