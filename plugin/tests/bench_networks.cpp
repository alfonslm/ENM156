// Speed on one core, per model file:
//   bench_networks model.bin [model.bin ...]
// - the network alone: frames per second on model input frames taken from a
//   synthetic noisy signal (SNN: both kernels, and the spike rate);
// - ChannelDenoiser (STFT + network + synthesis) per 16 kHz hop;
// - real-time use: HostRateAdapter at 48 kHz stereo with two ChannelDenoisers,
//   blocks of 512, as a share of one core.
// Live use needs hop frames per second of audio per channel (200 for hop 80).
#include "denoiser/denoiser.h"
#include "denoiser/networks.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace denoiser;
using Clock = std::chrono::steady_clock;

namespace {

// Speech-like test signal: harmonics with a moving pitch in bursts, plus noise
std::vector<float> signal(std::size_t n, int rate) {
    std::vector<float> s(n);
    std::uint32_t seed = 1;
    double phase = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double t = double(i) / rate;
        const double f0 = 140.0 + 40.0 * std::sin(2.0 * 3.141592653589793 * 0.7 * t);
        phase += 2.0 * 3.141592653589793 * f0 / rate;
        double v = 0.0;
        for (int h = 1; h <= 12; ++h)
            v += std::sin(h * phase) / h;
        const double env = std::sin(3.141592653589793 * std::fmod(t * 2.5, 1.0)) > 0.3 ? 0.15 : 0.0;
        seed = seed * 1664525u + 1013904223u;
        const double noise = (double(seed >> 8) / double(1u << 24) - 0.5) * 0.1;
        s[i] = float(env * v + noise);
    }
    return s;
}

template <class F>
double secondsFor(F&& f) {
    const auto start = Clock::now();
    f();
    return std::chrono::duration<double>(Clock::now() - start).count();
}

volatile float sink = 0.0f;

double framesPerSecond(FrameNetwork& net, const std::vector<std::vector<float>>& xs, std::size_t nOut) {
    std::vector<float> y(nOut);
    net.reset();
    for (std::size_t t = 0; t < 200 && t < xs.size(); ++t)  // warm up
        net.process(xs[t].data(), y.data());
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        net.reset();
        best = std::min(best, secondsFor([&] {
                            for (const auto& x : xs) {
                                net.process(x.data(), y.data());
                                sink = sink + y[0];
                            }
                        }));
    }
    return double(xs.size()) / best;
}

void bench(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto model = Model::fromMemory(bytes.data(), bytes.size());
    const ModelInfo& info = model->info();
    std::printf("%s\n  %s\n", path.c_str(), model->describe().c_str());
    const double live = double(info.sampleRate) / info.hop;

    // model input frames of 10 s of the test signal
    const std::vector<float> sig = signal(std::size_t(info.sampleRate) * 10, info.sampleRate);
    StreamProcessor stream(info);
    std::vector<std::vector<float>> xs;
    for (std::size_t k = 0; (k + 1) * std::size_t(info.hop) <= sig.size(); ++k) {
        const float* x = stream.analyse(sig.data() + k * std::size_t(info.hop));
        xs.emplace_back(x, x + info.nIn);
    }
    const std::size_t nOut = std::size_t(info.nOut);
    if (info.netType == "snn") {
        auto w = SnnWeights::build(model->file());
        for (auto kernel : {SnnNetwork::Kernel::Spikes, SnnNetwork::Kernel::Dense}) {
            SnnNetwork net(w, kernel);
            const double fps = framesPerSecond(net, xs, nOut);
            std::printf("  network (%s kernel): %.0f frames/s = %.1fx the %.0f frames/s needed live; spike rate %.3f; "
                        "%.2f M dense multiply-adds per frame\n",
                        kernel == SnnNetwork::Kernel::Spikes ? "spikes" : "dense", fps, fps / live, live,
                        double(net.spikeCount()) / double(net.neuronSteps()), double(w->denseMacsPerFrame()) / 1e6);
        }
    } else {
        auto net = model->makeNetwork();
        const double fps = framesPerSecond(*net, xs, nOut);
        const auto* mlp = dynamic_cast<const MlpNetwork*>(net.get());
        std::printf("  network: %.0f frames/s = %.1fx the %.0f frames/s needed live", fps, fps / live, live);
        if (mlp)
            std::printf("; %.2f M multiply-adds per frame = %.2f GFLOP/s", double(mlp->macsPerFrame()) / 1e6,
                        2.0 * double(mlp->macsPerFrame()) * fps / 1e9);
        std::printf("\n");
    }
    if (!info.realtime)
        return;

    // the whole per-channel denoiser
    ChannelDenoiser d(model);
    std::vector<float> out(std::size_t(info.hop));
    const std::size_t hops = sig.size() / std::size_t(info.hop);
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        d.reset();
        best = std::min(best, secondsFor([&] {
                            for (std::size_t k = 0; k < hops; ++k)
                                d.processHop(sig.data() + k * std::size_t(info.hop), out.data(), info.hop);
                        }));
    }
    std::printf("  ChannelDenoiser: %.0f hops/s = %.1fx real time per channel (%.2f%% of a core per channel)\n",
                double(hops) / best, double(hops) / best / live, 100.0 * best / 10.0);

    // 48 kHz stereo behind the adapter
    const std::vector<float> s48 = signal(48000 * 10, 48000);
    ChannelDenoiser l(model), r(model);
    HostRateAdapter adapter;
    adapter.prepare(l.hostRateConfig(48000, 2));
    adapter.setProcessor(0, &l);
    adapter.setProcessor(1, &r);
    std::vector<float> o0(512), o1(512);
    const double t48 = secondsFor([&] {
        for (std::size_t pos = 0; pos + 512 <= s48.size(); pos += 512) {
            const float* ins[2] = {s48.data() + pos, s48.data() + pos};
            float* outs[2] = {o0.data(), o1.data()};
            adapter.process(ins, outs, 512);
        }
    });
    std::printf("  48 kHz stereo, blocks of 512: %.1f ms per second of audio (%.1f%% of one core)\n", t48 * 100.0,
                t48 * 10.0);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: bench_networks model.bin [model.bin ...]\n");
        return 2;
    }
    for (int i = 1; i < argc; ++i)
        bench(argv[i]);
    return 0;
}
