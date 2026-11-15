// Speed of the incremental TCN at config.toml's default size (n_in 130,
// 256 channels, kernel 3, dilations 1, 2, 4, 8, 16, 32, n_out 65), random
// weights made here. One thread. Not part of the default ctest run except as
// a short smoke run (--frames 50 --repeats 2).
//
//   bench_tcn [--frames N] [--repeats R] [--channels C]
//
// Prints frames per second, the real-time factor at 200 frames/s (16 kHz,
// hop 80), and the cost of reset() and of building the model (which runs the
// start-up once).
#include "tcn_synth.h"

#include "denoiser/tcn.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

std::string cpuName() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line))
        if (line.rfind("model name", 0) == 0)
            return line.substr(line.find(':') + 2);
    return "unknown CPU";
}

volatile float sink = 0.0f;  // keeps the outputs alive

struct Timing {
    double best = 0.0, median = 0.0;  // seconds per frame
};

// `repeats` runs of `frames` frames: best and median seconds per frame
Timing timeFrames(denoiser::TcnNetwork& net, const std::vector<float>& x, std::size_t frames, int repeats) {
    const std::size_t nIn = std::size_t(tcn_synth::kNIn), rows = x.size() / nIn;
    std::vector<float> y(static_cast<std::size_t>(tcn_synth::kNOut));
    std::vector<double> runs;
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = Clock::now();
        for (std::size_t f = 0; f < frames; ++f) {
            net.process(x.data() + (f % rows) * nIn, y.data());
            sink = sink + y[f % y.size()];
        }
        runs.push_back(seconds(t0, Clock::now()) / double(frames));
    }
    std::sort(runs.begin(), runs.end());
    return {runs.front(), runs[runs.size() / 2]};
}

void report(const char* label, Timing t, std::size_t macs) {
    const double fps = 1.0 / t.best, medianFps = 1.0 / t.median;
    std::printf("  %-38s best %6.1f us/frame = %6.0f frames/s = %5.1fx real time (%.1f GMAC/s dense-equivalent); "
                "median %6.1f us = %5.1fx\n",
                label, t.best * 1e6, fps, fps / 200.0, double(macs) * fps * 1e-9, t.median * 1e6, medianFps / 200.0);
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t frames = 1000;
    int channels = 256, repeats = 15;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--frames") == 0)
            frames = std::size_t(std::atol(argv[i + 1]));
        else if (std::strcmp(argv[i], "--repeats") == 0)
            repeats = std::max(1, std::atoi(argv[i + 1]));
        else if (std::strcmp(argv[i], "--channels") == 0)
            channels = std::atoi(argv[i + 1]);
    }
    std::printf("bench_tcn: %s, one thread\n", cpuName().c_str());

    tcn_synth::Spec spec;
    spec.channels = channels;
    const std::string bytes = tcn_synth::modelBytes(spec);
    const denoiser::ModelFile model = denoiser::ModelFile::fromMemory(bytes.data(), bytes.size());

    auto t0 = Clock::now();
    auto shared = std::make_shared<const denoiser::TcnModel>(model);
    const double build = seconds(t0, Clock::now());
    denoiser::TcnNetwork net(shared);
    const denoiser::TcnModel& m = net.model();
    std::printf("TCN n_in %d, %d channels, kernel %d, %d blocks, context %d, n_out %d: %.2f M multiply-adds per frame "
                "(dense)\n",
                m.inputSize(), m.channels(), m.kernel(), m.blocks(), m.context(), m.outputSize(),
                double(m.macsPerFrame()) * 1e-6);
    std::printf("  weights %.1f MB per model (shared), state %.1f kB per instance (+ the same again for the snapshot, "
                "per model)\n",
                double(m.macsPerFrame()) * 4e-6, double(m.stateFloats()) * 4e-3);

    // input frames like log1p magnitudes: >= 0, a few exact zeros
    tcn_synth::Random rnd(5);
    std::vector<float> x(std::size_t(tcn_synth::kNIn) * 64);
    for (float& v : x)
        v = std::max(0.0f, 1.5f * rnd.uniform() + 1.0f);

    timeFrames(net, x, std::min<std::size_t>(frames, 200), 1);  // warm up
    report("random weights:", timeFrames(net, x, frames, repeats), m.macsPerFrame());

    // the worst case: nothing after a ReLU is zero, so no row is skipped
    tcn_synth::Spec dense = spec;
    dense.normBiasShift = 10.0f;
    dense.convBiasShift = 10.0f;
    const denoiser::ModelFile denseModel = tcn_synth::model(dense);
    denoiser::TcnNetwork denseNet(denseModel);
    timeFrames(denseNet, x, std::min<std::size_t>(frames, 200), 1);
    report("dense (no zeros to skip, worst case):", timeFrames(denseNet, x, frames, repeats), m.macsPerFrame());

    // reset(): a copy of the snapshot
    const int resets = 2000;
    double bestReset = 1e30;
    for (int r = 0; r < 5; ++r) {
        t0 = Clock::now();
        for (int i = 0; i < resets; ++i) {
            net.reset();
            sink = sink + net.state().rings[std::size_t(i) % net.state().rings.size()];
        }
        bestReset = std::min(bestReset, seconds(t0, Clock::now()) / resets);
    }
    // the straightforward start-up, for comparison: context - 1 frames
    t0 = Clock::now();
    net.startUp();
    const double startUp = seconds(t0, Clock::now());
    std::printf("  reset() %.2f us (copies %.0f kB); the start-up it replaces (%d frames) %.1f ms; building the model "
                "(incl. one start-up) %.1f ms\n",
                bestReset * 1e6, double(m.stateFloats()) * 4e-3, m.context() - 1, startUp * 1e3, build * 1e3);
    return 0;
}
