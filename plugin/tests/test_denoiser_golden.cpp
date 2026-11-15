// End to end, from the golden input waveform, for every golden case: MLP,
// SNN and TCN (networks.h + stream.h + offline.h + denoiser.h together):
//   - offline: denoiseClip vs golden offline (causal and centred)            1e-5
//   - streamed: ChannelDenoiser hop by hop (pre-roll not muted) vs golden
//     streamed: after the pre-roll 1e-5, the pre-roll itself 1e-4;
//     with muting on, the pre-roll is exactly 0 and the rest bit-identical
//   - the plugin's path at a 16 kHz host rate: ChannelDenoiser behind
//     HostRateAdapter. processClipAligned equals denoiseClip bit for bit over
//     the whole clip; in real-time use (random block sizes) the output is
//     exactly 0 for the first latencySamples() samples, then denoiseClip's
//     output bit for bit, shifted by latencySamples().
// Errors: max|c++ - golden| / max(1, max|golden|) (tools/README.md).
#include "check.h"
#include "golden.h"

#include "denoiser/denoiser.h"
#include "denoiser/offline.h"

#include <cstdint>
#include <cstring>

using namespace denoiser;

namespace {

constexpr double kTol = 1e-5;
constexpr double kPreRollTol = 1e-4;

std::shared_ptr<const Model> loadModel(const std::string& name) {
    const std::vector<char> bytes = golden::readBytes(golden::dir() + "/" + name + ".model.bin");
    return Model::fromMemory(bytes.data(), bytes.size());
}

bool sameBits(const float* a, const float* b, std::size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

std::uint32_t nextRandom(std::uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return seed >> 8;
}

// Stream through ChannelDenoiser, `chunks` hops of input then zeros
std::vector<float> streamHops(ChannelDenoiser& d, const Tensor& in, std::size_t chunks) {
    const std::size_t hop = std::size_t(d.hop());
    std::vector<float> chunk(hop), out(chunks * hop);
    d.reset();
    for (std::size_t k = 0; k < chunks; ++k) {
        for (std::size_t i = 0; i < hop; ++i)
            chunk[i] = k * hop + i < in.size() ? in.data[k * hop + i] : 0.0f;
        d.processHop(chunk.data(), out.data() + k * hop, int(hop));
    }
    return out;
}

// The 16 kHz host path vs the offline result of the same model, on `clip`
void checkHostPath(const std::shared_ptr<const Model>& model, const std::vector<float>& clip, const char* label) {
    std::unique_ptr<FrameNetwork> net = model->makeNetwork();
    const std::vector<float> offline = denoiseClip(model->info(), *net, clip.data(), clip.size());

    ChannelDenoiser d(model);
    HostRateAdapter adapter;
    adapter.prepare(d.hostRateConfig(16000, 1));
    adapter.setProcessor(0, &d);
    const int latency = adapter.latencySamples();
    CHECK(latency == hostLatencySamples(16000, d.hop(), d.processorDelay()));
    CHECK(latency == model->info().nFft - 1);

    // offline use of the adapter (ARA): aligned and the same length
    std::vector<float> aligned(clip.size());
    processClipAligned(adapter, 0, clip.data(), aligned.data(), std::int64_t(clip.size()));
    const bool alignedExact = sameBits(aligned.data(), offline.data(), clip.size());
    CHECK(alignedExact);

    // real-time use: the clip, then latency zeros, in random blocks of 0..700 samples
    const std::size_t total = clip.size() + std::size_t(latency);
    std::vector<float> in(total, 0.0f), out(total, 1.0f);
    std::copy(clip.begin(), clip.end(), in.begin());
    adapter.reset();
    std::uint32_t seed = 12345;
    for (std::size_t pos = 0; pos < total;) {
        const std::size_t n = std::min<std::size_t>(nextRandom(seed) % 701, total - pos);
        adapter.processChannel(0, in.data() + pos, out.data() + pos, int(n));
        pos += n;
    }
    bool leadingZeros = true;
    for (int i = 0; i < latency; ++i)
        leadingZeros = leadingZeros && out[std::size_t(i)] == 0.0f;
    CHECK(leadingZeros);
    const bool shiftedExact = sameBits(out.data() + latency, offline.data(), clip.size());
    CHECK(shiftedExact);
    CHECK(adapter.underruns() == 0);
    std::printf("    16 kHz host path (%s, %zu samples, latency %d): aligned %s, real-time blocks %s\n", label, clip.size(),
                latency, alignedExact ? "bit-exact" : "DIFFERS", leadingZeros && shiftedExact ? "bit-exact" : "DIFFERS");
}

void checkCase(const std::string& name) {
    const golden::Case c = golden::load(name);
    const ModelInfo& info = c.info();
    const std::shared_ptr<const Model> model = loadModel(name);
    const Tensor& in = c.get("input");

    // offline (either framing)
    std::unique_ptr<FrameNetwork> net = model->makeNetwork();
    const std::vector<float> y = denoiseClip(info, *net, in.data, in.size());
    const Tensor& offline = c.get("offline");
    CHECK(y.size() == offline.size());
    const double offErr = golden::relErr(y.data(), offline.data, y.size());
    CHECK(offErr <= kTol);
    std::printf("%-28s offline %.1e", name.c_str(), offErr);

    if (info.framing != Framing::Causal) {
        std::printf("   (centred: no stream)\n");
        return;
    }
    // streamed, raw
    const Tensor& streamed = c.get("streamed");
    const auto chunks = std::size_t(c.vectors.header().at("stream").at("chunks").integer());
    ChannelDenoiser d(model);
    CHECK(d.mutePreRoll());
    d.setMutePreRoll(false);
    const std::vector<float> raw = streamHops(d, in, chunks);
    CHECK(raw.size() == streamed.size());
    const std::size_t delay = std::size_t(d.processorDelay());
    const double preErr = golden::relErr(raw.data(), streamed.data, delay);
    const double strErr = golden::relErr(raw.data() + delay, streamed.data + delay, raw.size() - delay);
    CHECK(preErr <= kPreRollTol);
    CHECK(strErr <= kTol);
    // muted pre-roll
    d.setMutePreRoll(true);
    const std::vector<float> muted = streamHops(d, in, chunks);
    bool zeros = true;
    for (std::size_t i = 0; i < delay; ++i)
        zeros = zeros && muted[i] == 0.0f;
    CHECK(zeros);
    CHECK(sameBits(muted.data() + delay, raw.data() + delay, raw.size() - delay));
    double peak = 0.0;
    for (std::size_t i = 0; i < delay; ++i)
        peak = std::max(peak, std::fabs(double(raw[i])));
    std::printf("  streamed %.1e  pre-roll %.1e (max |pre-roll| %.0f)\n", strErr, preErr, peak);

    // the host path at 16 kHz, on the golden clip and on a longer one (the clip 8 times, 0.8 s)
    checkHostPath(model, std::vector<float>(in.data, in.data + in.size()), "golden clip");
    std::vector<float> longer;
    for (int r = 0; r < 8; ++r)
        longer.insert(longer.end(), in.data, in.data + in.size());
    checkHostPath(model, longer, "clip x 8");
}

}  // namespace

int main(int argc, char** argv) {
    golden::init(argc, argv);
    int cases = 0;
    for (const std::string& name : golden::caseNames()) {
        checkCase(name);
        ++cases;
    }
    CHECK(cases == 17);
    return check::finish();
}
