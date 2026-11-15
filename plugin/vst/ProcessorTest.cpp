// Tests on the processor class itself, without a plugin host (the processor
// sources are compiled into this program; ARA is off here):
//
//   denoiser_processor_test bench <model.bin|none>...   CPU use at 48 kHz stereo (not a test)
//   denoiser_processor_test alloc   <golden dir>   no heap allocation in processBlock, across model swaps
//   denoiser_processor_test core    <golden dir>   high band vs the core's HighBand options; what a swap does
//   denoiser_processor_test stress  <golden dir> [seconds]
//                                                  models swapped from other threads while audio runs
//                                                  (meant to be run under ThreadSanitizer, see README.md)
#include "PluginProcessor.h"

#include "denoiser/denoiser.h"

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_events/juce_events.h>

#if DENOISER_COUNT_ALLOCATIONS
#include "../tests/alloc_hooks.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

namespace {

int failures = 0;
juce::File goldenDir;

void fail(const juce::String& message) {
    std::printf("FAIL: %s\n", message.toRawUTF8());
    ++failures;
}

void check(bool ok, const juce::String& message) {
    if (!ok)
        fail(message);
}

juce::File golden(const char* name) {
    return goldenDir.getChildFile(juce::String(name) + ".model.bin");
}

std::unique_ptr<DenoiserProcessor> makeProcessor(int channels) {
    auto p = std::make_unique<DenoiserProcessor>();
    const auto layout = channels == 1 ? juce::AudioChannelSet::mono() : juce::AudioChannelSet::stereo();
    juce::AudioProcessor::BusesLayout buses;
    buses.inputBuses.add(layout);
    buses.outputBuses.add(layout);
    if (!p->setBusesLayout(buses))
        fail("layout not accepted");
    return p;
}

void setParam(DenoiserProcessor& p, const char* id, float value) {
    auto* param = p.parameters().getParameter(id);
    param->setValueNotifyingHost(param->convertTo0to1(value));
}

void fillNoise(juce::AudioBuffer<float>& b, juce::Random& random) {
    for (int c = 0; c < b.getNumChannels(); ++c)
        for (int i = 0; i < b.getNumSamples(); ++i)
            b.setSample(c, i, 0.3f * std::sin(0.02f * float(i + c)) + 0.2f * (random.nextFloat() * 2.0f - 1.0f));
}

bool allFinite(const juce::AudioBuffer<float>& b) {
    for (int c = 0; c < b.getNumChannels(); ++c)
        for (int i = 0; i < b.getNumSamples(); ++i)
            if (!std::isfinite(b.getSample(c, i)))
                return false;
    return true;
}

// Mono signal rendered in blocks of `block`; `onBlock(k)` runs before block k
template <class F>
std::vector<float> render(DenoiserProcessor& p, const std::vector<float>& in, int block, F&& onBlock) {
    std::vector<float> out(in.size());
    juce::MidiBuffer midi;
    juce::AudioBuffer<float> buffer(1, block);
    int k = 0;
    for (size_t pos = 0; pos < in.size(); pos += size_t(block), ++k) {
        onBlock(k);
        const int n = int(std::min(in.size() - pos, size_t(block)));
        buffer.setSize(1, n, false, false, true);
        std::copy(in.begin() + long(pos), in.begin() + long(pos) + n, buffer.getWritePointer(0));
        p.processBlock(buffer, midi);
        std::copy(buffer.getReadPointer(0), buffer.getReadPointer(0) + n, out.begin() + long(pos));
    }
    return out;
}

std::vector<float> noise(size_t n, int seed) {
    juce::Random random(seed);
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i)
        x[i] = 0.3f * std::sin(0.031f * float(i)) + 0.2f * (random.nextFloat() * 2.0f - 1.0f) +
               0.1f * std::sin(2.6f * float(i));  // also something above 8 kHz at 48 kHz
    return x;
}

double relErr(const float* a, const float* b, size_t n) {
    double diff = 0.0, peak = 1.0;
    for (size_t i = 0; i < n; ++i) {
        diff = std::max(diff, std::fabs(double(a[i]) - double(b[i])));
        peak = std::max(peak, std::fabs(double(b[i])));
    }
    return diff / peak;
}

// ---------------------------------------------------------------------------

#if DENOISER_COUNT_ALLOCATIONS
void testAllocations() {
    int* probe = nullptr;
    check(alloc::countOnThisThread([&] { probe = new int(1); }) >= 1, "the allocation counter doesn't work");
    delete probe;
    for (int rate : {48000, 44100, 16000}) {
        std::printf("allocations in processBlock at %d Hz, stereo\n", rate);
        auto p = makeProcessor(2);
        p->loadModelFile(golden("mlp_magnitude_default"));
        p->prepareToPlay(rate, 512);

        juce::Random random(1);
        // Buffers that refer to memory allocated here, so the test itself
        // doesn't allocate while counting
        std::vector<float> left(512), right(512);
        float* channels[] = {left.data(), right.data()};
        juce::MidiBuffer midi;
        auto* mix = p->parameters().getParameter(DenoiserProcessor::kMixId);
        auto* bypass = p->parameters().getParameter(DenoiserProcessor::kBypassId);
        auto* high = p->parameters().getParameter(DenoiserProcessor::kHighBandId);
        bool finite = true;
        // Blocks of random sizes 0..512 with parameter changes the way a VST3
        // wrapper delivers them on the audio thread (setValue), and a host reset
        const auto run = [&](int blocks) {
            // Only this (the audio) thread: JUCE's timer thread allocates
            // while posting timer messages nobody dispatches here
            return alloc::countOnThisThread([&] {
                for (int b = 0; b < blocks; ++b) {
                    const int n = random.nextInt(513);
                    juce::AudioBuffer<float> buffer(channels, 2, n);
                    fillNoise(buffer, random);
                    if (b % 17 == 0)
                        mix->setValue(random.nextFloat());
                    if (b % 23 == 0)
                        high->setValue(high->getValue() < 0.5f ? 1.0f : 0.0f);
                    if (b % 31 == 0)
                        bypass->setValue(bypass->getValue() < 0.5f ? 1.0f : 0.0f);
                    if (b == blocks / 2)
                        p->reset();
                    p->processBlock(buffer, midi);
                    finite = finite && allFinite(buffer);
                }
            });
        };

        long count = run(300);
        check(count == 0, "processBlock allocated " + juce::String(count) + " times");
        // Swaps prepared beforehand: TCN (same shape), other shape, no model,
        // offline-only model, SNN. Each is taken, crossfaded in and retired on
        // the audio thread; the old one is freed here, outside the count.
        for (const char* next : {"tcn_magnitude_default", "mlp_magnitude_short", "", "offline_mlp_mask_centred",
                                 "snn_complex_default"}) {
            if (*next)
                p->loadModelFile(golden(next));
            else
                p->clearModel();
            count = run(200);  // > 1 s at 48 kHz: the crossfade is long finished
            check(count == 0, juce::String("processBlock allocated ") + juce::String(count) + " times swapping to " +
                                  (*next ? next : "no model"));
            p->collectGarbage();
        }
        check(finite, "non-finite output");
        p->releaseResources();
    }
}
#endif

// The high-band switch against the core's own HostRateAdapter options, and
// mix 1 / bypass against the core directly
void testAgainstCore() {
    const auto model = [] {
        juce::MemoryBlock bytes;
        golden("mlp_complex_default").loadFileAsData(bytes);
        return denoiser::Model::fromMemory(bytes.getData(), bytes.getSize());
    }();
    for (int rate : {48000, 44100}) {
        for (auto band : {denoiser::HighBand::Drop, denoiser::HighBand::PassThrough}) {
            const bool pass = band == denoiser::HighBand::PassThrough;
            std::printf("plugin vs core HostRateAdapter at %d Hz, high band %s\n", rate, pass ? "pass" : "drop");
            const std::vector<float> in = noise(size_t(rate), 11);

            denoiser::ChannelDenoiser ch(model);
            denoiser::HostRateAdapter adapter;
            adapter.prepare(ch.hostRateConfig(rate, 1, band));
            adapter.setProcessor(0, &ch);
            std::vector<float> ref(in.size());
            for (size_t pos = 0; pos < in.size(); pos += 300) {
                const int n = int(std::min<size_t>(300, in.size() - pos));
                const float* i0 = in.data() + pos;
                float* o0 = ref.data() + pos;
                adapter.process(&i0, &o0, n);
            }

            auto p = makeProcessor(1);
            p->loadModelFile(golden("mlp_complex_default"));
            setParam(*p, DenoiserProcessor::kHighBandId, pass ? 1.0f : 0.0f);
            p->prepareToPlay(rate, 1024);
            check(p->getLatencySamples() == adapter.latencySamples(), "latency differs from the core's");
            const std::vector<float> out = render(*p, in, 1024, [](int) {});
            const double err = relErr(out.data(), ref.data(), in.size());
            std::printf("  error %.2e\n", err);
            check(err <= 1e-5, "plugin differs from the core: " + juce::String(err));
        }
    }

    std::printf("messages\n");
    auto p = makeProcessor(2);
    p->prepareToPlay(48000, 512);
    check(!p->loadModelFile(goldenDir.getChildFile("missing.model.bin")), "missing file loaded?");
    auto s = p->getStatus();
    check(!s.denoising && s.message.contains("not found") && s.latencySamples == 669, "missing file status: " + s.message);
    check(!p->loadModelFile(golden("offline_tcn_complex_centred")), "centred model runs live?");
    s = p->getStatus();
    check(!s.denoising && s.message.contains("centred") && p->getModel() != nullptr &&
              s.description.isNotEmpty(),
          "centred model status: " + s.message);
    check(p->loadModelFile(golden("snn_complex_default")), "snn model doesn't run");
    s = p->getStatus();
    check(s.denoising && s.message.isEmpty() && s.description.startsWith("snn"), "snn status: " + s.description);
    p->prepareToPlay(8000, 512);
    s = p->getStatus();
    check(!s.denoising && s.message.contains("not supported") && s.latencySamples == 0, "8 kHz status: " + s.message);
}

// What a swap does: the old model stays audible until the new one is warm,
// then a 10 ms crossfade; from then on the output equals a fresh instance
// of the new model started at the swap
void testSwap() {
    std::printf("model swap while running\n");
    const int rate = 48000, block = 256;
    const std::vector<float> in = noise(size_t(rate), 12);
    const int swapBlock = 40;  // the swap happens at the start of this block

    auto a = makeProcessor(1);
    a->loadModelFile(golden("mlp_magnitude_default"));
    a->prepareToPlay(rate, block);
    const std::vector<float> out = render(*a, in, block, [&](int k) {
        if (k == swapBlock)
            a->loadModelFile(golden("tcn_magnitude_default"));
    });

    const size_t swapAt = size_t(swapBlock * block);
    const std::vector<float> tail(in.begin() + long(swapAt), in.end());
    auto b = makeProcessor(1);
    b->loadModelFile(golden("tcn_magnitude_default"));
    b->prepareToPlay(rate, block);
    const std::vector<float> fresh = render(*b, tail, block, [](int) {});

    auto oldOnly = makeProcessor(1);
    oldOnly->loadModelFile(golden("mlp_magnitude_default"));
    oldOnly->prepareToPlay(rate, block);
    const std::vector<float> old = render(*oldOnly, in, block, [](int) {});

    // Until the new engine is warm: the old model's output, exactly
    const int latency = b->getLatencySamples();
    const int widest = b->getModel()->info().widestWindow();
    const int warm = latency + int(std::ceil(double(widest) * rate / 16000));  // + the analysis window
    const int fade = rate / 100;
    check(relErr(out.data(), old.data(), swapAt + size_t(warm)) <= 0.0, "before the crossfade: not the old model");
    const size_t settled = swapAt + size_t(warm + fade);
    const double err = relErr(out.data() + settled, fresh.data() + (settled - swapAt), in.size() - settled);
    std::printf("  after the crossfade vs a fresh instance: %.2e\n", err);
    check(err <= 1e-6, "after the crossfade: not the new model");
    // No jump during the crossfade bigger than the signals themselves
    double jump = 0.0;
    for (size_t i = swapAt + size_t(warm); i < settled; ++i)
        jump = std::max(jump, double(std::fabs(out[i] - out[i - 1])));
    double typical = 0.0;
    for (size_t i = 1; i < swapAt; ++i)
        typical = std::max(typical, double(std::fabs(old[i] - old[i - 1])));
    for (size_t i = settled + 1; i < out.size(); ++i)
        typical = std::max(typical, double(std::fabs(out[i] - out[i - 1])));
    std::printf("  largest step during the crossfade %.3f (outside it %.3f)\n", jump, typical);
    check(jump <= typical * 1.5, "a click during the crossfade");
}

// Audio runs on one thread while other threads load models, change
// parameters, reset, save / restore state and read the status
void testStress(double seconds) {
    std::printf("stress: %.1f s of audio with swaps from other threads\n", seconds);
    auto p = makeProcessor(2);
    p->prepareToPlay(48000, 1024);

    std::atomic<bool> stop{false};
    std::atomic<long> blocks{0}, loads{0}, statusReads{0};
    std::atomic<bool> finite{true};

    std::thread audio([&] {
        juce::Random random(3);
        juce::AudioBuffer<float> buffer(2, 1024);
        juce::MidiBuffer midi;
        while (!stop.load()) {
            const int n = 1 + random.nextInt(1024);
            buffer.setSize(2, n, false, false, true);
            fillNoise(buffer, random);
            p->processBlock(buffer, midi);
            if (!allFinite(buffer))
                finite = false;
            ++blocks;
        }
    });
    std::thread status([&] {
        while (!stop.load()) {
            const auto s = p->getStatus();
            const auto m = p->getModel();
            juce::ignoreUnused(s, m);
            ++statusReads;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    });

    const char* models[] = {"mlp_magnitude_default", "tcn_magnitude_default", "snn_complex_default",
                            "mlp_magnitude_short", "tcn_mask_short", "offline_mlp_mask_centred", "missing"};
    juce::Random random(4);
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < end) {
        const int action = random.nextInt(10);
        if (action < 6) {
            p->loadModelFile(golden(models[random.nextInt(int(std::size(models)))]));
            ++loads;
        } else if (action == 6) {
            p->clearModel();
        } else if (action == 7) {
            juce::MemoryBlock state;
            p->getStateInformation(state);
            p->setStateInformation(state.getData(), int(state.getSize()));
        } else if (action == 8) {
            setParam(*p, DenoiserProcessor::kMixId, random.nextFloat());
            setParam(*p, DenoiserProcessor::kHighBandId, float(random.nextInt(2)));
            setParam(*p, DenoiserProcessor::kBypassId, float(random.nextInt(2)));
        } else {
            p->reset();
        }
        p->collectGarbage();
        std::this_thread::sleep_for(std::chrono::milliseconds(random.nextInt(8)));
    }
    stop = true;
    audio.join();
    status.join();
    std::printf("  %ld audio blocks, %ld model loads, %ld status reads\n", blocks.load(), loads.load(),
                statusReads.load());
    check(finite.load(), "non-finite output");
    check(blocks.load() > 100 && loads.load() > 10, "too little happened");
    p->releaseResources();
}

// CPU use of processBlock at 48 kHz stereo, blocks of 512, for model files
// given on the command line (not a test; see README.md for the numbers)
void bench(const std::vector<juce::String>& files, double seconds) {
    const int rate = 48000, block = 512, channels = 2;
    const int blocks = int(seconds * rate / block);
    std::vector<float> left(block), right(block);
    float* data[] = {left.data(), right.data()};
    juce::Random random(5);
    juce::MidiBuffer midi;
    for (const juce::String& file : files) {
        for (bool highPass : {false, true}) {
            auto p = makeProcessor(channels);
            if (file != "none")
                p->loadModelFile(juce::File::getCurrentWorkingDirectory().getChildFile(file));
            setParam(*p, DenoiserProcessor::kHighBandId, highPass ? 1.0f : 0.0f);
            p->prepareToPlay(rate, block);
            double best = 1e30, worstBlock = 0.0;
            for (int run = 0; run < 3; ++run) {
                p->reset();
                double total = 0.0, slowest = 0.0;
                for (int b = 0; b < blocks; ++b) {
                    juce::AudioBuffer<float> buffer(data, channels, block);
                    fillNoise(buffer, random);
                    const auto t0 = std::chrono::steady_clock::now();
                    p->processBlock(buffer, midi);
                    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    total += t;
                    slowest = std::max(slowest, t);
                }
                if (total < best) {
                    best = total;
                    worstBlock = slowest;
                }
            }
            const auto s = p->getStatus();
            std::printf("%s, high band %s: %.2f %% of one core (%.1f ms per second of audio), slowest block "
                        "%.0f %% of its duration; %s\n",
                        file.toRawUTF8(), highPass ? "pass" : "drop", 100.0 * best / seconds, 1000.0 * best / seconds,
                        100.0 * worstBlock * rate / block, s.denoising ? s.description.toRawUTF8() : "pass-through");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: %s alloc|core|stress <golden dir> [seconds]\n"
                    "       %s bench <model.bin|none>...\n",
                    argv[0], argv[0]);
        return 2;
    }
    const juce::String mode = argv[1];
    if (mode == "bench") {
        juce::ScopedJuceInitialiser_GUI juce;
        std::vector<juce::String> files;
        for (int i = 2; i < argc; ++i)
            files.push_back(juce::String::fromUTF8(argv[i]));
        bench(files, 10.0);
        return 0;
    }
    goldenDir = juce::File::getCurrentWorkingDirectory().getChildFile(juce::String::fromUTF8(argv[2]));
    juce::ScopedJuceInitialiser_GUI juce;

    if (mode == "alloc") {
#if DENOISER_COUNT_ALLOCATIONS
        testAllocations();
#else
        std::printf("allocation counting is not available in this build (sanitizer)\n");
#endif
    } else if (mode == "core") {
        testAgainstCore();
        testSwap();
    } else if (mode == "stress") {
        testStress(argc > 3 ? std::atof(argv[3]) : 2.0);
    } else {
        std::printf("unknown mode %s\n", argv[1]);
        return 2;
    }
    std::printf(failures == 0 ? "all checks passed\n" : "%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
