// The incremental TCN (tcn.h):
// - against every golden TCN case from the golden x, from the first frame on:
//   pred and, where stored, the per-block channels tcn.h*;
// - end to end from the golden input: streamed (StreamProcessor) and offline
//   (denoiseClip, also the centred case);
// - reset(): mid-stream, and after the rings were filled with garbage, gives
//   bit for bit what a fresh instance gives; the snapshot reset() restores
//   equals the straightforward start-up (context - 1 zero input frames);
// - against a brute-force evaluation (whole padded sequence, double
//   precision, written from tools/README.md) for synthetic models of several
//   kernels, dilations and channel counts, for clips shorter and longer than
//   the context;
// - malformed TCN headers give a FormatError.
//
// Tolerances (tools/README.md): pred, tcn.h*, streamed after the pre-roll,
// offline 1e-5; pre-roll 1e-4. Error = max |c++ - golden| / max(1, max |golden|).
#include "check.h"
#include "golden.h"
#include "tcn_synth.h"

#include "denoiser/offline.h"
#include "denoiser/stream.h"
#include "denoiser/tcn.h"

#include <cmath>
#include <cstring>
#include <string>

using namespace denoiser;

namespace {

constexpr double kTol = 1e-5;
constexpr double kPreRollTol = 1e-4;

bool sameBits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Runs the network over rows [0, rows) of x; returns all outputs
std::vector<float> run(FrameNetwork& net, const float* x, std::size_t rows, std::size_t nIn, std::size_t nOut) {
    std::vector<float> y(rows * nOut);
    for (std::size_t t = 0; t < rows; ++t)
        net.process(x + t * nIn, y.data() + t * nOut);
    return y;
}

// pred and tcn.h* from the golden x
void checkNetwork(const golden::Case& c) {
    const ModelInfo& info = c.info();
    const Tensor& x = c.get("x");
    const Tensor& pred = c.get("pred");
    const std::size_t T = x.shape[0], nIn = std::size_t(info.nIn), nOut = std::size_t(info.nOut);
    TcnNetwork net(c.model);
    const TcnModel& m = net.model();
    CHECK(m.inputSize() == info.nIn && m.outputSize() == info.nOut);
    CHECK(m.context() == info.contextFrames);
    const std::size_t C = std::size_t(m.channels());

    std::vector<const Tensor*> hs;
    for (int i = 0; i <= m.blocks(); ++i)
        hs.push_back(c.find(("tcn.h" + std::to_string(i)).c_str()));

    golden::ErrorTracker predErr, firstErr;
    std::vector<golden::ErrorTracker> hErr(hs.size());
    std::vector<float> y(nOut);
    net.reset();
    for (std::size_t t = 0; t < T; ++t) {
        net.process(x.data + t * nIn, y.data());
        predErr.add(y.data(), pred.data + t * nOut, nOut);
        if (t == 0)
            firstErr.add(y.data(), pred.data, nOut);
        for (std::size_t i = 0; i < hs.size(); ++i)
            if (hs[i]) {
                CHECK(hs[i]->shape[1] == C);
                hErr[i].add(net.hidden(int(i)), hs[i]->data + t * C, C);
            }
    }
    std::printf("%-28s context %3d, %zu frames: pred %.1e (frame 0: %.1e)", c.name.c_str(), m.context(), T,
                predErr.worst, firstErr.worst);
    CHECK(predErr.worst <= kTol);
    for (std::size_t i = 0; i < hs.size(); ++i)
        if (hs[i]) {
            std::printf(" h%zu %.1e", i, hErr[i].worst);
            CHECK(hErr[i].worst <= kTol);
        }
    std::printf("\n");

    // the factory gives the same network
    std::unique_ptr<FrameNetwork> made = makeTcnNetwork(c.model);
    made->reset();
    const std::vector<float> a = run(*made, x.data, T, nIn, nOut);
    net.reset();
    CHECK(sameBits(a, run(net, x.data, T, nIn, nOut)));
}

// reset() mid-stream and after garbage; the snapshot vs the straightforward start-up
void checkReset(const golden::Case& c) {
    const ModelInfo& info = c.info();
    const Tensor& x = c.get("x");
    const std::size_t T = x.shape[0], nIn = std::size_t(info.nIn), nOut = std::size_t(info.nOut);

    TcnNetwork fresh(c.model);
    const std::vector<float> first = run(fresh, x.data, T, nIn, nOut);  // no reset(): as constructed

    // mid-stream
    TcnNetwork used(c.model);
    run(used, x.data + (T / 3) * nIn, T - T / 3, nIn, nOut);
    used.reset();
    CHECK(sameBits(first, run(used, x.data, T, nIn, nOut)));
    run(used, x.data, T / 2, nIn, nOut);
    used.reset();
    CHECK(sameBits(first, run(used, x.data, T, nIn, nOut)));

    // garbage in every ring and the write positions anywhere, then reset()
    for (std::uint32_t seed = 1; seed <= 3; ++seed) {
        used.scrambleState(seed);
        used.reset();
        CHECK(sameBits(first, run(used, x.data, T, nIn, nOut)));
    }

    // the straightforward route: garbage, then context - 1 zero input frames
    // pushed through. Same ring contents as the snapshot (in time order), and
    // bit-identical outputs afterwards.
    TcnNetwork reference(c.model);
    const std::vector<float> snapshot = reference.orderedState();
    for (std::uint32_t seed = 7; seed <= 9; ++seed) {
        used.scrambleState(seed);
        CHECK(!sameBits(used.orderedState(), snapshot));
        used.startUp();
        CHECK(sameBits(used.orderedState(), snapshot));
        CHECK(sameBits(first, run(used, x.data, T, nIn, nOut)));
    }

    // After the start-up every frame in a block's ring is the same: the
    // response to constant (zero) input is constant once it has filled the
    // ring
    const TcnModel& m = reference.model();
    const std::size_t C = std::size_t(m.channels());
    bool constant = true;
    std::size_t at = 0;
    for (int b = 0; b < m.blocks(); ++b) {
        for (int f = 1; f < m.ringFrames(b); ++f)
            constant = constant && std::memcmp(&snapshot[at], &snapshot[at + std::size_t(f) * C], C * sizeof(float)) == 0;
        at += std::size_t(m.ringFrames(b)) * C;
    }
    CHECK(constant);
    CHECK(m.stateFloats() == snapshot.size());
}

// End to end from the golden input
void checkEndToEnd(const golden::Case& c) {
    const ModelInfo& info = c.info();
    const Tensor& in = c.get("input");
    const Tensor& offline = c.get("offline");
    TcnNetwork net(c.model);

    golden::ErrorTracker off, streamedErr, preRollErr, predErr;
    const std::vector<float> y = denoiseClip(info, net, in.data, in.size());
    CHECK(y.size() == offline.size());
    off.add(y.data(), offline.data, std::min(y.size(), offline.size()));
    CHECK(off.worst <= kTol);
    std::printf("  end to end: offline %.1e", off.worst);

    if (info.framing == Framing::Causal) {
        const Tensor& streamed = c.get("streamed");
        const Tensor& pred = c.get("pred");
        const auto chunks = std::size_t(c.vectors.header().at("stream").at("chunks").integer());
        const std::size_t hop = std::size_t(info.hop), nOut = std::size_t(info.nOut);
        StreamProcessor stream(info);
        std::vector<float> chunk(hop), out(hop), p(nOut);
        net.reset();
        for (std::size_t k = 0; k < chunks; ++k) {
            for (std::size_t i = 0; i < hop; ++i)
                chunk[i] = k * hop + i < in.size() ? in.data[k * hop + i] : 0.0f;
            const float* x = stream.analyse(chunk.data());
            net.process(x, p.data());
            predErr.add(p.data(), pred.data + k * nOut, nOut);
            stream.synthesise(p.data(), out.data());
            for (std::size_t i = 0; i < hop; ++i) {
                const std::size_t s = k * hop + i;
                if (s < std::size_t(stream.delay()))
                    preRollErr.add(&out[i], streamed.data + s, 1);
                else
                    streamedErr.add(&out[i], streamed.data + s, 1);
            }
        }
        std::printf(", pred %.1e, streamed %.1e, pre-roll %.1e", predErr.worst, streamedErr.worst, preRollErr.worst);
        CHECK(predErr.worst <= kTol);
        CHECK(streamedErr.worst <= kTol);
        CHECK(preRollErr.worst <= kPreRollTol);
    }
    std::printf("\n");
}

// ---- brute force -----------------------------------------------------------------

// FrameModel.forward on the whole clip, the way tools/README.md writes it:
// context - 1 zero frames in front, every block's conv over the whole padded
// sequence (unpadded, so each block shortens it by trim). Double precision.
std::vector<double> bruteForce(const ModelFile& model, const std::vector<float>& x, std::size_t T) {
    const Json& net = model.net();
    const TensorFile& t = model.tensors();
    const std::size_t C = std::size_t(net.at("channels").integer()), K = std::size_t(net.at("kernel").integer());
    const std::size_t nIn = std::size_t(model.info().nIn), nOut = std::size_t(model.info().nOut);
    const std::size_t pad = std::size_t(model.info().contextFrames) - 1;
    const double eps = net.at("layer_norm_eps").number();
    std::size_t frames = T + pad;

    auto normRelu = [&](const std::vector<double>& h, std::size_t f, const float* w, const float* b, double* out) {
        double mean = 0.0, var = 0.0;
        for (std::size_t i = 0; i < C; ++i)
            mean += h[f * C + i];
        mean /= double(C);
        for (std::size_t i = 0; i < C; ++i)
            var += (h[f * C + i] - mean) * (h[f * C + i] - mean);
        var /= double(C);
        for (std::size_t i = 0; i < C; ++i)
            out[i] = std::max(0.0, (h[f * C + i] - mean) / std::sqrt(var + eps) * w[i] + b[i]);
    };

    std::vector<double> h(frames * C);
    const float* wi = t.get("input.weight").data;
    const float* bi = t.get("input.bias").data;
    for (std::size_t f = 0; f < frames; ++f)
        for (std::size_t o = 0; o < C; ++o) {
            double s = bi[o];
            if (f >= pad)
                for (std::size_t i = 0; i < nIn; ++i)
                    s += double(wi[o * nIn + i]) * x[(f - pad) * nIn + i];
            h[f * C + o] = s;
        }
    const std::vector<Json>& blocks = net.at("blocks").array();
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        const std::string p = "block" + std::to_string(b) + ".";
        const std::size_t d = std::size_t(blocks[b].at("dilation").integer()), trim = (K - 1) * d;
        std::vector<double> a(frames * C);
        for (std::size_t f = 0; f < frames; ++f)
            normRelu(h, f, t.get(p + "norm.weight").data, t.get(p + "norm.bias").data, &a[f * C]);
        const float* wc = t.get(p + "conv.weight").data;
        const float* bc = t.get(p + "conv.bias").data;
        const float* wm = t.get(p + "mix.weight").data;
        const float* bm = t.get(p + "mix.bias").data;
        const std::size_t outFrames = frames - trim;
        std::vector<double> next(outFrames * C), conv(C);
        for (std::size_t f = 0; f < outFrames; ++f) {
            for (std::size_t o = 0; o < C; ++o) {
                double s = bc[o];
                for (std::size_t i = 0; i < C; ++i)
                    for (std::size_t j = 0; j < K; ++j)
                        s += double(wc[(o * C + i) * K + j]) * a[(f + j * d) * C + i];
                conv[o] = std::max(0.0, s);
            }
            for (std::size_t o = 0; o < C; ++o) {
                double s = bm[o] + h[(f + trim) * C + o];
                for (std::size_t i = 0; i < C; ++i)
                    s += double(wm[o * C + i]) * conv[i];
                next[f * C + o] = s;
            }
        }
        h.swap(next);
        frames = outFrames;
    }
    std::vector<double> y(T * nOut), a(C);
    const float* wo = t.get("output.weight").data;
    const float* bo = t.get("output.bias").data;
    for (std::size_t f = 0; f < T; ++f) {
        normRelu(h, f, t.get("final_norm.weight").data, t.get("final_norm.bias").data, a.data());
        for (std::size_t o = 0; o < nOut; ++o) {
            double s = bo[o];
            for (std::size_t i = 0; i < C; ++i)
                s += double(wo[o * C + i]) * a[i];
            y[f * nOut + o] = std::max(0.0, s);  // relu: the synthetic models' out_activation
        }
    }
    return y;
}

void checkBruteForce() {
    struct Case {
        int channels, kernel;
        std::vector<int> dilations;
        float shift;
    };
    const Case cases[] = {
        {13, 1, {1, 1}, 0.0f},             // kernel 1: rings of one frame
        {8, 2, {1, 3}, 0.0f},              // kernel 2
        {16, 3, {1, 2, 4, 8, 16, 32}, 0.0f},  // the default dilations (context 127)
        {9, 4, {5, 1, 7}, 0.3f},           // kernel 4, dilations not increasing
        {12, 3, {64}, -0.3f},              // one block, long ring
    };
    std::uint32_t seed = 100;
    for (const Case& k : cases) {
        tcn_synth::Spec spec;
        spec.channels = k.channels;
        spec.kernel = k.kernel;
        spec.dilations = k.dilations;
        spec.normBiasShift = k.shift;
        spec.seed = ++seed;
        const ModelFile model = tcn_synth::model(spec);
        TcnNetwork net(model);
        for (std::size_t T : {std::size_t(1), std::size_t(5), std::size_t(net.model().context() + 40)}) {
            // input frames like log1p magnitudes: >= 0, some exactly 0
            tcn_synth::Random rnd(seed * 31u + std::uint32_t(T));
            std::vector<float> x(T * tcn_synth::kNIn);
            for (float& v : x)
                v = std::max(0.0f, 2.0f * rnd.uniform() + 0.5f);
            const std::vector<double> ref = bruteForce(model, x, T);
            net.reset();
            const std::vector<float> y = run(net, x.data(), T, tcn_synth::kNIn, tcn_synth::kNOut);
            double diff = 0.0, peak = 1.0;
            for (std::size_t i = 0; i < y.size(); ++i) {
                diff = std::max(diff, std::fabs(double(y[i]) - ref[i]));
                peak = std::max(peak, std::fabs(ref[i]));
            }
            std::printf("  brute force: C %2d K %d %zu blocks, context %3d, %3zu frames: %.1e\n", k.channels, k.kernel,
                        k.dilations.size(), net.model().context(), T, diff / peak);
            CHECK(diff / peak <= kTol);
        }
    }
}

// ---- malformed headers ----------------------------------------------------------

template <class F>
bool throwsFormatError(F&& f) {
    try {
        f();
    } catch (const FormatError&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void checkMalformed() {
    tcn_synth::Spec spec;
    spec.channels = 4;
    spec.dilations = {1, 2};
    const std::string good = tcn_synth::modelBytes(spec);
    CHECK(!throwsFormatError([&] { TcnNetwork n(ModelFile::fromMemory(good.data(), good.size())); }));
    struct Edit {
        const char* from;
        const char* to;
    };
    const Edit edits[] = {
        {"\"type\": \"tcn\"", "\"type\": \"mlp\""},             // not a TCN
        {"\"trim\": 4}", "\"trim\": 5}"},                       // trim != (K - 1) * d
        {"\"context\": 7}", "\"context\": 8}"},                 // net.context wrong
        {"\"channels\": 4", "\"channels\": 5"},                 // tensor shapes don't match
        {"\"kernel\": 3", "\"kernel\": 2"},                     // conv shape doesn't match (and trims)
        {"\"layer_norm_eps\": 1e-05", "\"layer_norm_eps\": 0"}, // eps
        {"\"layer_norm_eps\": 1e-05", "\"layer_norm_eps\": \"x\""},
        {"\"dilation\": 2", "\"dilation\": 0"},
    };
    int n = 0;
    for (const Edit& e : edits) {
        // the header with the edit, re-padded; the data section unchanged
        std::uint32_t len = 0;
        for (int b = 0; b < 4; ++b)
            len |= std::uint32_t(static_cast<unsigned char>(good[12 + std::size_t(b)])) << (8 * b);
        std::string header = good.substr(16, len);
        const std::string data = good.substr(16 + len);
        header.erase(header.find_last_not_of(' ') + 1);
        const std::size_t at = header.find(e.from);
        CHECK(at != std::string::npos);
        if (at == std::string::npos)
            continue;
        header.replace(at, std::strlen(e.from), e.to);
        while ((16 + header.size()) % 64 != 0)
            header += ' ';
        std::string bad = good.substr(0, 12);
        for (int b = 0; b < 4; ++b)
            bad += char((header.size() >> (8 * b)) & 0xFF);
        bad += header + data;
        const bool refused = throwsFormatError([&] {
            const ModelFile m = ModelFile::fromMemory(bad.data(), bad.size());
            TcnNetwork net(m);
        });
        if (!refused)
            std::printf("  not refused: %s -> %s\n", e.from, e.to);
        CHECK(refused);
        ++n;
    }
    std::printf("  %d malformed TCN headers refused\n", n);
}

}  // namespace

int main(int argc, char** argv) {
    golden::init(argc, argv);
    int tcnCases = 0;
    for (const std::string& name : golden::caseNames()) {
        const golden::Case c = golden::load(name);
        if (c.info().netType != "tcn") {
            CHECK(throwsFormatError([&] { TcnNetwork n(c.model); }));
            continue;
        }
        ++tcnCases;
        checkNetwork(c);
        checkEndToEnd(c);
        checkReset(c);
    }
    CHECK(tcnCases == 5);
    checkBruteForce();
    checkMalformed();
    return check::finish();
}
