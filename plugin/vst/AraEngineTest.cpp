// The ARA engine below the ARA interface (AraEngine.h), no JUCE, no host:
//
//   denoiser_ara_engine_test <golden dir> [stress seconds]
//
//   - SourceAnalysis: the stored audio equals the core's denoiseClipAtRate
//     (16, 44.1, 48 kHz; causal and centred models; both high-band choices);
//     other playback rates through 16 kHz; no model, too short, empty and
//     unsupported rates keep the original with a message; cancellation
//   - renderPlan: timeline -> source mapping for moved, trimmed, split and
//     overlapping regions, two regions of one source, mono/stereo in and
//     out, the mix, regions without audio or without denoised audio yet
//   - no heap allocation in renderPlan (counted; off under sanitizers)
//   - AudioSlot under load: one thread publishes new audio while two others
//     render from it; every block must come from one complete, live
//     SourceAudio (run under ThreadSanitizer / AddressSanitizer, see README)
#include "AraEngine.h"

#include "denoiser/denoiser.h"
#include "denoiser/offline.h"
#include "denoiser/offline_rate.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#if DENOISER_COUNT_ALLOCATIONS
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#endif
#include "../tests/alloc_hooks.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

using namespace enm156::ara;

namespace {

int failures = 0;
std::string goldenDir;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::shared_ptr<const denoiser::Model> loadModel(const std::string& name) {
    std::ifstream in(goldenDir + "/" + name + ".model.bin", std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return denoiser::Model::fromMemory(bytes.data(), bytes.size());
}

std::vector<float> signal(std::size_t n, int seed) {
    std::vector<float> x(n);
    std::uint32_t s = std::uint32_t(seed) * 2654435761u + 1u;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        x[i] = 0.3f * float(std::sin(0.013 * double(i) * (1 + seed % 3))) + 0.1f * (float(s >> 8) / 8388608.0f - 1.0f);
    }
    return x;
}

double relErr(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        return 1e9;
    double d = 0.0, peak = 1.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::fabs(double(a[i]) - double(b[i])));
        peak = std::max(peak, std::fabs(double(b[i])));
    }
    return d / peak;
}

// ---- Analysis ----

void testAnalysis() {
    std::printf("analysis vs denoiseClipAtRate\n");
    struct Case {
        const char* model;
        int rate;
    };
    for (const Case& k : {Case{"mlp_magnitude_default", 16000}, Case{"snn_complex_default", 48000},
                          Case{"tcn_magnitude_default", 44100}, Case{"offline_mlp_mask_centred", 48000},
                          Case{"offline_tcn_complex_centred", 16000}}) {
        auto model = loadModel(k.model);
        const std::size_t n = std::size_t(k.rate) / 3 + 11;
        Planar in{signal(n, 1), signal(n, 2)};
        SourceAnalysis a;
        a.setInput(k.rate, in);
        int calls = 0;
        const Outcome o = a.denoise(model, [&](float) {
            ++calls;
            return true;
        });
        check(o == Outcome::Denoised, std::string(k.model) + ": not denoised: " + a.message());
        check(calls > 2, "no progress reported");
        for (auto hb : {denoiser::HighBand::Drop, denoiser::HighBand::PassThrough}) {
            auto audio = a.atSourceRate(hb);
            check(audio->channels == 2 && audio->length == std::int64_t(n) && audio->sampleRate == k.rate,
                  "shape");
            check(audio->dry && *audio->dry == in, "dry != input");
            double worst = 0.0;
            for (std::size_t c = 0; c < 2; ++c) {
                auto net = model->makeNetwork();
                const auto ref = denoiser::denoiseClipAtRate(model->info(), *net, k.rate, in[c].data(), n, hb);
                worst = std::max(worst, relErr(audio->wet[c], ref));
            }
            std::printf("  %s at %d Hz, high band %s: %.2e\n", k.model, k.rate,
                        hb == denoiser::HighBand::Drop ? "drop" : "pass", worst);
            check(worst <= 1e-6, std::string(k.model) + ": stored audio != denoiseClipAtRate");
        }
        // Another playback rate: the 16 kHz results upsampled there
        const int other = k.rate == 48000 ? 44100 : 48000;
        auto audio = a.atRate(other);
        check(audio != nullptr && audio->sampleRate == other && audio->wet.size() == 2, "rendition");
        if (audio) {
            const std::size_t m = std::size_t(audio->length);
            check(std::llabs(std::int64_t(m) - std::llround(double(n) * other / k.rate)) == 0, "rendition length");
            auto filterIn = denoiser::ResampleFilter::create(k.rate);
            auto filterOut = denoiser::ResampleFilter::create(other);
            auto net = model->makeNetwork();
            const auto x16 = denoiser::clipToModelRate(*filterIn, in[0].data(), n);
            const auto y16 = denoiser::denoiseClip(model->info(), *net, x16.data(), x16.size());
            const auto ref = denoiser::clipFromModelRate(*filterOut, y16.data(), y16.size(), m);
            check(relErr(audio->wet[0], ref) <= 1e-6, "rendition wet");
        }
    }
}

void testAnalysisEdges() {
    std::printf("analysis: no model, too short, empty, unsupported rate, >2 channels, cancel\n");
    auto causal = loadModel("mlp_magnitude_default");
    auto centred = loadModel("offline_mlp_mask_centred");
    SourceAnalysis a;
    a.setInput(48000, Planar{signal(4800, 3)});
    check(a.denoise(nullptr, nullptr) == Outcome::NoModel && !a.message().empty(), "no model");
    auto audio = a.atSourceRate(denoiser::HighBand::Drop);
    check(audio->wet.empty() && audio->dry->size() == 1, "no model: original only");
    check(audio->dry == a.atSourceRate(denoiser::HighBand::PassThrough)->dry, "the original is shared, not copied");

    const std::size_t min16 = denoiser::minClipSamples(centred->info());
    a.setInput(48000, Planar{signal(3 * (min16 - 1), 4)});
    check(a.denoise(centred, nullptr) == Outcome::TooShort, "too short for a centred model");
    check(a.message().find("Too short") != std::string::npos, "too-short message: " + a.message());
    a.setInput(48000, Planar{signal(3 * (min16 - 1), 4)});
    check(a.denoise(causal, nullptr) == Outcome::Denoised, "a causal model takes any length");

    a.setInput(48000, Planar{std::vector<float>{}, std::vector<float>{}});
    check(a.denoise(causal, nullptr) == Outcome::Empty, "empty");
    check(a.atSourceRate(denoiser::HighBand::Drop)->length == 0, "empty audio");

    a.setInput(8000, Planar{signal(800, 5)});
    check(a.denoise(causal, nullptr) == Outcome::UnsupportedRate, "8 kHz");
    check(a.atRate(48000) == nullptr, "no rendition from an unsupported rate");
    a.setInput(47999, Planar{signal(800, 5)});
    check(a.denoise(causal, nullptr) == Outcome::UnsupportedRate, "47999 Hz");

    a.setInput(16000, Planar{signal(500, 1), signal(500, 2), signal(500, 3)});
    check(a.channels() == 2, "at most 2 channels are kept");

    a.setInput(48000, Planar{signal(48000, 6)});
    int calls = 0;
    check(a.denoise(causal, [&](float) { return ++calls < 5; }) == Outcome::Cancelled, "cancel");
    check(!a.denoised(), "cancelled: nothing denoised");
}

// A source of any length without its samples in memory: each sample a
// deterministic function of (channel, position), exactly representable
float synth(int channel, std::int64_t k, std::uint64_t seed = 0) {
    std::uint64_t x = std::uint64_t(k) * std::uint64_t{0x9E3779B97F4A7C15u} +
                      std::uint64_t(channel + 1) * std::uint64_t{0x632BE59BD9B4E019u} +
                      seed * std::uint64_t{0xD6E8FEB86659FD93u};
    x ^= x >> 29;
    x *= std::uint64_t{0xBF58476D1CE4E5B9u};
    x ^= x >> 32;
    return float((x >> 40) & 0xFFFFFFu) / 16777216.0f - 0.5f;
}

struct SynthSource {
    int channels = 2;
    std::int64_t length = 0;
    std::uint64_t seed = 0;
    std::atomic<bool> fail{false};
    std::atomic<long> reads{0};
    SourceReader reader() {
        return [this](std::int64_t pos, std::int64_t n, float* const* out) {
            if (pos < 0 || n < 0 || pos + n > length)
                return ReadStatus::Failed;  // outside the source: the caller's bug
            ++reads;
            if (fail.load())
                return ReadStatus::Failed;
            for (int c = 0; c < channels; ++c)
                for (std::int64_t i = 0; i < n; ++i)
                    out[c][i] = synth(c, pos + i, seed);
            return ReadStatus::Ok;
        };
    }
};

// ---- Rendering ----

std::shared_ptr<AudioSlot> slotWith(int channels, std::int64_t n, float dryBase, float wetBase, bool wet) {
    auto audio = std::make_shared<SourceAudio>();
    audio->sampleRate = 48000;
    audio->channels = channels;
    audio->length = n;
    Planar dry;
    for (int c = 0; c < channels; ++c) {
        std::vector<float> d(static_cast<size_t>(n)), w(static_cast<size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) {
            d[size_t(i)] = dryBase + float(c) * 1000.0f + float(i);  // sample index visible in the value
            w[size_t(i)] = wetBase + float(c) * 1000.0f + float(i);
        }
        dry.push_back(std::move(d));
        if (wet)
            audio->wet.push_back(std::move(w));
    }
    audio->dry = std::make_shared<const Planar>(std::move(dry));
    auto slot = std::make_shared<AudioSlot>();
    slot->publish(audio);
    return slot;
}

std::vector<std::vector<float>> render(const RenderPlan& plan, std::int64_t pos, int n, int channels, float g,
                                       RenderResult* result = nullptr) {
    std::vector<std::vector<float>> out(size_t(channels), std::vector<float>(size_t(n), -7.0f));
    std::vector<float> gain(size_t(n), g);
    float* ptrs[2] = {out[0].data(), channels > 1 ? out[1].data() : nullptr};
    const RenderResult r = renderPlan(plan, pos, n, ptrs, channels, gain.data());
    if (result)
        *result = r;
    return out;
}

void testRendering() {
    std::printf("render plan: regions, mapping, channels, mix\n");
    // Source A: 1000 samples, dry = 10000 + i, wet = 50000 + i (ch 1: +1000)
    auto a = slotWith(2, 1000, 10000.0f, 50000.0f, true);
    RenderPlan plan;
    plan.keepAlive = {a};
    // Region placed at timeline 500, playing source samples 100..399 (moved + trimmed)
    plan.regions.push_back({500, 800, 100 - 500, a.get()});
    auto out = render(plan, 400, 512, 2, 1.0f);
    bool ok = true;
    for (int i = 0; i < 512; ++i) {
        const std::int64_t t = 400 + i;
        const float expect0 = (t >= 500 && t < 800) ? 50000.0f + float(t - 400) : 0.0f;
        const float expect1 = (t >= 500 && t < 800) ? 51000.0f + float(t - 400) : 0.0f;
        ok = ok && out[0][size_t(i)] == expect0 && out[1][size_t(i)] == expect1;
    }
    check(ok, "moved + trimmed region");
    // Mix 0: exactly the original; mix 0.25: the crossfade
    out = render(plan, 500, 10, 2, 0.0f);
    check(out[0][0] == 10100.0f && out[0][9] == 10109.0f, "mix 0 = original");
    out = render(plan, 500, 10, 2, 0.25f);
    check(std::fabs(out[0][3] - (0.75f * 10103.0f + 0.25f * 50103.0f)) < 1e-2f, "mix 0.25");

    // Split: two regions of the same source, back to back, the second also moved
    RenderPlan split;
    split.keepAlive = {a};
    split.regions.push_back({0, 200, 0, a.get()});         // source 0..199 at 0..199
    split.regions.push_back({300, 500, 200 - 300, a.get()});  // source 200..399 at 300..499
    out = render(split, 0, 600, 1, 1.0f);  // stereo source on a mono output: the mean
    ok = true;
    for (int t = 0; t < 600; ++t) {
        float e = 0.0f;
        if (t < 200)
            e = 50500.0f + float(t);
        else if (t >= 300 && t < 500)
            e = 50500.0f + float(t - 100);
        ok = ok && out[0][size_t(t)] == e;
    }
    check(ok, "split regions, stereo source on mono output");

    // Overlapping regions sum; a region running past the end of its source is silent there
    RenderPlan overlap;
    auto b = slotWith(1, 100, 0.0f, 2.0f, true);  // wet = 2 + i
    overlap.keepAlive = {a, b};
    overlap.regions.push_back({0, 300, 0, b.get()});  // only 100 samples of source
    overlap.regions.push_back({50, 60, -50, b.get()});  // source 0..9 again at 50..59
    out = render(overlap, 0, 300, 2, 1.0f);           // mono source on stereo: both channels
    check(out[0][10] == 12.0f && out[1][10] == 12.0f, "mono source on both channels");
    check(out[0][55] == (2.0f + 55.0f) + (2.0f + 5.0f), "overlapping regions sum");
    check(out[0][150] == 0.0f && out[1][299] == 0.0f, "past the end of the source: silence");

    // No wet yet: the original whatever the mix; no audio yet: silence
    auto c = slotWith(1, 100, 3.0f, 0.0f, false);
    auto empty = std::make_shared<AudioSlot>();
    RenderPlan pending;
    pending.keepAlive = {c, empty};
    pending.regions.push_back({0, 100, 0, c.get()});
    pending.regions.push_back({100, 200, -100, empty.get()});
    RenderResult r;
    out = render(pending, 0, 200, 1, 1.0f, &r);
    check(out[0][7] == 10.0f && out[0][150] == 0.0f, "no wet: original; nothing: silence");
    check(r.regionsOriginal == 1 && r.regionsSilent == 1 && r.regionsRendered == 0, "render result counts");

    // Before the start of the source (negative source positions): silence
    RenderPlan early;
    early.keepAlive = {b};
    early.regions.push_back({0, 100, -50, b.get()});
    out = render(early, 0, 100, 1, 1.0f);
    check(out[0][49] == 0.0f && out[0][50] == 2.0f, "region starting before its source");

    check(planIsWorking(plan, 0, 1000) == false, "not working");
    a->working = true;
    check(planIsWorking(plan, 0, 1000) && !planIsWorking(plan, 0, 500) && !planIsWorking(plan, 800, 10),
          "working flag only for overlapping regions");
    a->working = false;
}

#if DENOISER_COUNT_ALLOCATIONS
void testNoAllocation() {
    std::printf("no allocation in renderPlan\n");
    auto a = slotWith(2, 48000, 0.0f, 1.0f, true);
    auto b = slotWith(1, 4800, 0.0f, 1.0f, false);
    auto empty = std::make_shared<AudioSlot>();
    // A streamed source, half of what is played read
    SynthSource src;
    src.length = 1'000'000;
    auto stream = std::make_shared<SourceStream>(48000, 2, src.length, 48000);
    {
        const std::vector<CursorState> cursors{CursorState{0, 0}};
        const auto read = src.reader();
        stream->fillOne(cursors, read);
        stream->fillOne(cursors, read);
    }
    auto streamedAudio = std::make_shared<SourceAudio>();
    streamedAudio->channels = 2;
    streamedAudio->length = stream->length();
    streamedAudio->stream = stream;
    auto s = std::make_shared<AudioSlot>();
    s->publish(streamedAudio);
    auto cursor = s->addCursor();
    RenderPlan plan;
    plan.keepAlive = {a, b, empty, s};
    plan.cursors = {cursor};
    plan.regions = {{0, 48000, 0, a.get()},
                    {1000, 5800, -1000, b.get()},
                    {0, 100000, 0, empty.get()},
                    {0, 64000, 0, s.get(), cursor.get()}};
    std::vector<float> l(256), r(256), g(256, 0.5f);
    float* out[2] = {l.data(), r.data()};
    const long n = alloc::count([&] {
        for (std::int64_t pos = 0; pos < 60000; pos += 256) {
            notePlayPosition(plan, pos);
            renderPlan(plan, pos, 256, out, 2, g.data());
        }
        planIsWorking(plan, 0, 256);
        planStreamState(plan, 0, 256);
    });
    std::printf("  allocations: %ld\n", n);
    check(n == 0, "renderPlan allocated");
}
#endif

// ---- AudioSlot under load ----

void testSlotStress(double seconds) {
    std::printf("AudioSlot: publish while rendering, %.1f s\n", seconds);
    auto slot = std::make_shared<AudioSlot>();
    RenderPlan plan;
    plan.keepAlive = {slot};
    plan.regions.push_back({0, 1 << 20, 0, slot.get()});
    std::atomic<bool> stop{false};
    std::atomic<long> published{0}, blocks{0}, bad{0};

    std::thread writer([&] {
        int version = 1;
        while (!stop.load()) {
            auto audio = std::make_shared<SourceAudio>();
            audio->sampleRate = 48000;
            audio->channels = 2;
            audio->length = 512 + (version % 7) * 300;
            Planar dry;
            for (int c = 0; c < 2; ++c) {
                dry.emplace_back(size_t(audio->length), float(version));
                if (version % 3 != 0)
                    audio->wet.emplace_back(size_t(audio->length), float(version));
            }
            audio->dry = std::make_shared<const Planar>(std::move(dry));
            slot->publish(std::move(audio));
            ++version;
            ++published;
        }
        slot->publish(nullptr);
    });
    std::vector<std::thread> readers;
    for (int k = 0; k < 2; ++k)
        readers.emplace_back([&] {
            std::vector<float> l(256), r(256), g(256, 0.5f);
            float* out[2] = {l.data(), r.data()};
            while (!stop.load()) {
                renderPlan(plan, 0, 256, out, 2, g.data());
                // One SourceAudio per block: all samples equal (the version), or silence
                for (int i = 1; i < 256; ++i)
                    if (l[size_t(i)] != l[0] || r[size_t(i)] != l[0])
                        ++bad;
                ++blocks;
            }
        });
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop = true;
    writer.join();
    for (auto& t : readers)
        t.join();
    std::printf("  %ld publications during %ld blocks, %ld torn blocks\n", published.load(), blocks.load(),
                bad.load());
    check(bad.load() == 0, "a block mixed two publications");
    check(published.load() > 10 && blocks.load() > 10, "the stress test did not run");
}

// ---- Streaming the original ----

// Loads everything the cursor wants (and the pool can hold)
void fillAll(SourceStream& stream, const SourceReader& read, std::int64_t position) {
    const std::vector<CursorState> cursors{CursorState{position, 0}};
    for (int i = 0; i < 1000 && stream.fillOne(cursors, read) == SourceStream::Fill::Loaded; ++i) {
    }
}

std::vector<float> streamed(const SourceStream& stream, std::int64_t pos, int n, int channel, int outChannels = 2) {
    std::vector<std::vector<float>> out(size_t(outChannels), std::vector<float>(size_t(n), 0.0f));
    float* ptrs[2] = {out[0].data(), outChannels > 1 ? out[1].data() : nullptr};
    check(stream.mixInto(pos, n, ptrs, outChannels) == 0, "streamed range not all there");
    return out[size_t(channel)];
}

void testStream() {
    std::printf("SourceStream: the original at its own rate and at others, eviction, failures\n");
    // Own rate: exactly the source's samples, across block edges, mono and stereo
    for (int channels : {1, 2}) {
        SynthSource src;
        src.channels = channels;
        src.length = 5 * SourceStream::kBlockFrames + 123;
        SourceStream stream(48000, channels, src.length, 48000);
        check(stream.length() == src.length && stream.channels() == channels, "stream shape");
        const auto read = src.reader();
        fillAll(stream, read, 0);
        check(stream.ready(0, src.length), "not all read (short source)");
        bool ok = true;
        for (int c = 0; c < 2; ++c) {
            const auto y = streamed(stream, SourceStream::kBlockFrames - 100, 20000, c);
            for (size_t i = 0; i < y.size(); ++i)
                ok = ok && y[i] == synth(std::min(c, channels - 1), SourceStream::kBlockFrames - 100 + std::int64_t(i));
        }
        check(ok, "own rate: not the original bit for bit");
        // Stereo on a mono output: the mean
        if (channels == 2) {
            const auto m = streamed(stream, 7, 10, 0, 1);
            check(m[3] == 0.5f * (synth(0, 10) + synth(1, 10)), "stereo stream on a mono output");
        }
        // The last block is partial: nothing is played past the end
        std::vector<float> l(64, 0.0f), r(64, 0.0f);
        float* ptrs[2] = {l.data(), r.data()};
        check(stream.mixInto(src.length - 10, 64, ptrs, 2) == 0 && l[9] == synth(0, src.length - 1) &&
                  l[10] == 0.0f && l[63] == 0.0f,
              "end of the source");
    }
    // Another rate: what an analysed source plays there (atRate's dry), from any block
    struct Case {
        int from, to;
    };
    for (const Case k : {Case{44100, 48000}, Case{48000, 44100}, Case{16000, 48000}, Case{48000, 16000}}) {
        SynthSource src;
        src.length = std::int64_t(k.from) * 3 + 17;
        Planar whole(2, std::vector<float>(size_t(src.length)));
        for (int c = 0; c < 2; ++c)
            for (std::int64_t i = 0; i < src.length; ++i)
                whole[size_t(c)][size_t(i)] = synth(c, i);
        SourceAnalysis analysis;
        analysis.setInput(k.from, whole);
        const auto ref = analysis.atRate(k.to);
        SourceStream stream(k.from, 2, src.length, k.to);
        check(ref && stream.length() == ref->length, "length at another rate");
        if (!ref)
            continue;
        const auto read = src.reader();
        double worst = 0.0;
        // Cursors at several places, so that blocks are computed from windows in the middle too
        for (std::int64_t at = 0; at < stream.length(); at += 7 * SourceStream::kBlockFrames)
            fillAll(stream, read, at);
        for (std::int64_t at = 0; at < stream.length(); at += 7 * SourceStream::kBlockFrames) {
            const int n = int(std::min<std::int64_t>(4 * SourceStream::kBlockFrames, stream.length() - at));
            for (int c = 0; c < 2; ++c) {
                const auto y = streamed(stream, at, n, c);
                const std::vector<float> r((*ref->dry)[size_t(c)].begin() + at, (*ref->dry)[size_t(c)].begin() + at + n);
                worst = std::max(worst, relErr(y, r));
            }
        }
        std::printf("  %d -> %d Hz vs the whole source through 16 kHz: %.2e\n", k.from, k.to, worst);
        check(worst <= 1e-6, "streamed at another rate != atRate's dry");
    }
    check(!SourceAnalysis().atRate(48000), "sanity");
    bool threw = false;
    try {
        SourceStream bad(8000, 1, 1000, 48000);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "an unsupported rate pair must throw (silence there, as for analysed sources)");

    // Read-ahead and eviction: a long source; the pool keeps what the cursors want soonest
    SynthSource longSrc;
    longSrc.length = 100'000'000;
    SourceStream stream(48000, 2, longSrc.length, 48000);
    const auto read = longSrc.reader();
    const std::int64_t B = SourceStream::kBlockFrames;
    fillAll(stream, read, 10 * B + 5);
    check(stream.ready(10 * B + 5, SourceStream::kReadAheadFrames - 5), "read ahead of the cursor");
    check(!stream.ready(9 * B, 1) && !stream.ready(10 * B + SourceStream::kReadAheadFrames + B, 1),
          "nothing read before the cursor or far past the read-ahead");
    // A cursor whose region starts later than the read-ahead: nothing yet; closer: its start
    const std::vector<CursorState> later{CursorState{50'000'000, SourceStream::kReadAheadFrames + 1}};
    check(stream.fillOne(later, read) == SourceStream::Fill::Idle, "a region far ahead is not read yet");
    const std::vector<CursorState> soon{CursorState{50'000'000, B}};
    check(stream.fillOne(soon, read) == SourceStream::Fill::Loaded && stream.ready(50'000'000, 1),
          "a region starting soon: its start is read");
    // Many cursors: the pool (64 blocks) is shared; the nearest blocks come first
    std::vector<CursorState> many;
    for (int i = 0; i < 20; ++i)
        many.push_back({std::int64_t(i) * 1000 * B, 0});
    for (int i = 0; i < 2000 && stream.fillOne(many, read) == SourceStream::Fill::Loaded; ++i) {
    }
    bool nearest = true;
    for (const auto& c : many)
        nearest = nearest && stream.ready(c.position, B);
    check(nearest, "with 20 cursors, each cursor's current block is read");
    // A failed read: nothing installed, failing() until a read succeeds again (after kRetryDelay)
    longSrc.fail = true;
    const std::vector<CursorState> elsewhere{CursorState{80'000'000, 0}};
    check(stream.fillOne(elsewhere, read) == SourceStream::Fill::Failed && stream.failing(), "a failed read");
    check(stream.fillOne(elsewhere, read) == SourceStream::Fill::Idle, "no retry before the delay");
    longSrc.fail = false;
    std::this_thread::sleep_for(SourceStream::kRetryDelay + std::chrono::milliseconds(5));
    check(stream.fillOne(elsewhere, read) == SourceStream::Fill::Loaded && !stream.failing() &&
              stream.ready(80'000'000, 1),
          "read again after the delay");
}

void testReadWholeSource() {
    std::printf("readWholeSource: ok, too long, host failure, no access, cancel\n");
    SynthSource src;
    src.length = 3 * kReadChunk + 5;
    Planar out;
    check(readWholeSource(src.reader(), 2, src.length, out, nullptr) == LoadResult::Ok && out.size() == 2 &&
              out[1][size_t(src.length - 1)] == synth(1, src.length - 1),
          "read whole");
    // Three channels: two kept (the reader writes the kept ones)
    check(readWholeSource(src.reader(), 3, src.length, out, nullptr) == LoadResult::Ok && out.size() == 2, "kept");
    check(readWholeSource(src.reader(), 2, kMaxSourceSamples / 2 + 1, out, nullptr) == LoadResult::TooLong &&
              out.empty(),
          "too long: not allocated");
    src.fail = true;
    check(readWholeSource(src.reader(), 2, src.length, out, nullptr) == LoadResult::Failed, "host failure");
    src.fail = false;
    const SourceReader none = [](std::int64_t, std::int64_t, float* const*) { return ReadStatus::NoAccess; };
    check(readWholeSource(none, 2, src.length, out, nullptr) == LoadResult::NoAccess, "no access");
    int chunks = 0;
    check(readWholeSource(src.reader(), 2, src.length, out, [&](float) { return ++chunks < 2; }) ==
              LoadResult::Cancelled,
          "cancel");
}

#if DENOISER_COUNT_ALLOCATIONS
// An allocation failure, as on a machine short of memory: operator new fails
// for requests above 16 MB. The whole source can't be held; its stream can.
void testOutOfMemory() {
    std::printf("out of memory: the whole source can't be read into memory, its stream plays the original\n");
    SynthSource src;
    src.length = 10'000'000;  // 2 x 40 MB
    Planar out;
    alloc::failAbove = 16u << 20;
    LoadResult result = LoadResult::Ok;
    bool streamMade = false, original = false;
    try {
        result = readWholeSource(src.reader(), 2, src.length, out, nullptr);
        SourceStream stream(48000, 2, src.length, 48000);  // 8 MB
        streamMade = true;
        const auto read = src.reader();
        fillAll(stream, read, 6'000'000);
        std::vector<float> l(4096, 0.0f), r(4096, 0.0f);
        float* ptrs[2] = {l.data(), r.data()};
        original = stream.mixInto(6'000'000, 4096, ptrs, 2) == 0;
        for (int i = 0; i < 4096; ++i)
            original = original && l[size_t(i)] == synth(0, 6'000'000 + i) && r[size_t(i)] == synth(1, 6'000'000 + i);
    } catch (const std::bad_alloc&) {
    }
    alloc::failAbove = static_cast<std::size_t>(-1);
    check(result == LoadResult::OutOfMemory && out.empty(), "the failed allocation is reported, nothing kept");
    check(streamMade && original, "the stream plays the original within the same memory");
}
#endif

// The streaming thread fills while audio threads render and seek: every
// rendered sample is the original or silence (not yet read), never anything
// else; once a position has been waited for, it plays (run under TSan / ASan)
void testStreamStress(double seconds) {
    std::printf("SourceStream: a reader thread fills while two audio threads render and seek, %.1f s\n", seconds);
    SynthSource src;
    src.length = 50'000'000;
    auto stream = std::make_shared<SourceStream>(48000, 2, src.length, 48000);
    auto audio = std::make_shared<SourceAudio>();
    audio->sampleRate = 48000;
    audio->channels = 2;
    audio->length = stream->length();
    audio->stream = stream;
    auto slot = std::make_shared<AudioSlot>();
    slot->publish(audio);
    std::atomic<bool> stop{false};
    std::atomic<long> loads{0}, blocks{0}, bad{0}, complete{0};
    std::thread filler([&] {
        std::vector<CursorState> cursors;
        const auto read = src.reader();
        while (!stop.load()) {
            slot->cursorStates(cursors);
            if (stream->fillOne(cursors, read) == SourceStream::Fill::Loaded)
                ++loads;
            else
                std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    });
    std::vector<std::thread> players;
    for (int k = 0; k < 2; ++k)
        players.emplace_back([&, k] {
            // Two regions per plan: the source from 0 at timeline 0, and a
            // second region of it (source from 20 M) at timeline 1 M
            RenderPlan plan;
            plan.keepAlive = {slot};
            auto c0 = slot->addCursor(), c1 = slot->addCursor();
            plan.cursors = {c0, c1};
            plan.regions.push_back({0, src.length, 0, slot.get(), c0.get()});
            plan.regions.push_back({1'000'000, 1'000'000 + 10'000'000, 19'000'000, slot.get(), c1.get()});
            std::uint32_t rnd = 12345u + std::uint32_t(k);
            std::int64_t pos = 0;
            std::vector<float> l(256), r(256), g(256, 0.7f);
            float* out[2] = {l.data(), r.data()};
            while (!stop.load()) {
                rnd = rnd * 1664525u + 1013904223u;
                if (rnd % 200 == 0)
                    pos = std::int64_t(rnd >> 4) % (src.length - 1000);  // seek
                notePlayPosition(plan, pos);
                const RenderResult res = renderPlan(plan, pos, 256, out, 2, g.data());
                for (int i = 0; i < 256; ++i) {
                    const std::int64_t t = pos + i;
                    // Each region's contribution: the original or 0 (whatever the mix)
                    const float a0 = t < src.length ? synth(0, t) : 0.0f, a1 = t < src.length ? synth(1, t) : 0.0f;
                    const bool in1 = t >= 1'000'000 && t < 11'000'000;
                    const float b0 = in1 ? synth(0, t + 19'000'000) : 0.0f, b1 = in1 ? synth(1, t + 19'000'000) : 0.0f;
                    const auto okOne = [](float y, float a, float b) {
                        return y == 0.0f || y == a || y == 0.0f + b || y == a + b;
                    };
                    if (!okOne(l[size_t(i)], a0, b0) || !okOne(r[size_t(i)], a1, b1))
                        ++bad;
                }
                if (res.framesMissing == 0)
                    ++complete;
                ++blocks;
                pos += 256;
                if (pos + 256 > src.length)
                    pos = 0;
                std::this_thread::sleep_for(std::chrono::microseconds(50));  // ~5x real time
            }
        });
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop = true;
    for (auto& t : players)
        t.join();
    filler.join();
    std::printf("  %ld blocks loaded during %ld rendered blocks (%ld complete), %ld wrong samples\n", loads.load(),
                blocks.load(), complete.load(), bad.load());
    check(bad.load() == 0, "a rendered sample was neither the original nor silence");
    check(loads.load() > 10 && blocks.load() > 100 && complete.load() > blocks.load() / 2,
          "the stress test did not stream (most blocks should be complete)");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <golden dir> [stress seconds]\n", argv[0]);
        return 2;
    }
    goldenDir = argv[1];
    const double seconds = argc > 2 ? std::atof(argv[2]) : 1.0;
    try {
        testAnalysis();
        testAnalysisEdges();
        testRendering();
#if DENOISER_COUNT_ALLOCATIONS
        testNoAllocation();
#endif
        testSlotStress(seconds);
        testStream();
        testReadWholeSource();
#if DENOISER_COUNT_ALLOCATIONS
        testOutOfMemory();
#endif
        testStreamStress(seconds);
    } catch (const std::exception& e) {
        std::printf("FAIL: exception %s\n", e.what());
        ++failures;
    }
    std::printf(failures == 0 ? "all checks passed\n" : "%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
