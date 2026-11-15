// Loads the built VST3 the way a host does (JUCE's VST3 hosting, no GUI) and
// checks the real-time denoiser through the plugin interface only: latency,
// output against the core library, block sizes, bypass, mix, state, and what
// happens without a usable model. Run by ctest as `plugin_vst3`.
//
//   denoiser_plugin_host_test "<path to>/ENM156 Denoiser.vst3" <golden dir>
//
// Models are given to the plugin the way a host restores a session: through
// its state (the model file path is part of it). The output is compared
// with the core (denoiseClip, built into this test from the same sources)
// with a tolerance, because the plugin is compiled separately (link-time
// optimisation, maybe fused multiply-adds); exact equality is only required
// for what the plugin copies (pass-through, bypass, mix 0) and for latencies.
#include "denoiser/denoiser.h"
#include "denoiser/offline.h"
#include "denoiser/tensor_file.h"

#include <juce_audio_processors_headless/juce_audio_processors_headless.h>
#include <juce_events/juce_events.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

namespace {

using Audio = std::vector<std::vector<float>>;  // planar

int failures = 0;
juce::String goldenDir;

void fail(const juce::String& message) {
    std::printf("FAIL: %s\n", message.toRawUTF8());
    ++failures;
}

void check(bool ok, const juce::String& message) {
    if (!ok)
        fail(message);
}

juce::String golden(const juce::String& name) {
    return juce::File(goldenDir).getChildFile(name).getFullPathName();
}

// ---- Plugin instances ----

struct Host {
    juce::VST3PluginFormatHeadless format;
    juce::PluginDescription description;

    std::unique_ptr<juce::AudioPluginInstance> create(int channels) {
        juce::String error;
        auto plugin = format.createInstanceFromDescription(description, 48000.0, 4096, error);
        if (!plugin) {
            fail("could not create plugin: " + error);
            return nullptr;
        }
        const auto layout = channels == 1 ? juce::AudioChannelSet::mono() : juce::AudioChannelSet::stereo();
        juce::AudioProcessor::BusesLayout buses;
        buses.inputBuses.add(layout);
        buses.outputBuses.add(layout);
        if (!plugin->setBusesLayout(buses))
            fail("layout not accepted: " + layout.getDescription());
        return plugin;
    }
};

// The plugin's own state (our XML) inside the VST3 host's wrapper state
std::unique_ptr<juce::XmlElement> getPluginXml(juce::AudioPluginInstance& plugin) {
    juce::MemoryBlock block;
    plugin.getStateInformation(block);
    auto wrapper = juce::AudioProcessor::getXmlFromBinary(block.getData(), int(block.getSize()));
    if (wrapper == nullptr || wrapper->getChildByName("IComponent") == nullptr)
        return nullptr;
    juce::MemoryBlock component;
    component.fromBase64Encoding(wrapper->getChildByName("IComponent")->getAllSubText());
    // The JUCE wrapper appends its own data after ours; ours has a length header
    return juce::AudioProcessor::getXmlFromBinary(component.getData(), int(component.getSize()));
}

void setPluginXml(juce::AudioPluginInstance& plugin, const juce::XmlElement& xml) {
    juce::MemoryBlock component;
    juce::AudioProcessor::copyXmlToBinary(xml, component);
    juce::XmlElement wrapper("VST3PluginState");
    wrapper.createNewChildElement("IComponent")->addTextElement(component.toBase64Encoding());
    juce::MemoryBlock block;
    juce::AudioProcessor::copyXmlToBinary(wrapper, block);
    plugin.setStateInformation(block.getData(), int(block.getSize()));
}

struct Settings {
    juce::String modelPath;
    float mix = 1.0f;
    bool bypass = false;
    bool highBandPass = false;
    int hop = 80, nFft = 128;  // shape remembered for a missing model
};

void setParam(juce::XmlElement& params, const juce::String& id, double value) {
    for (auto* p : params.getChildIterator())
        if (p->getStringAttribute("id") == id) {
            p->setAttribute("value", value);
            return;
        }
    auto* p = params.createNewChildElement("PARAM");
    p->setAttribute("id", id);
    p->setAttribute("value", value);
}

void apply(juce::AudioPluginInstance& plugin, const Settings& s) {
    auto xml = getPluginXml(plugin);
    if (xml == nullptr) {
        fail("could not read the plugin's state");
        return;
    }
    xml->setAttribute("modelPath", s.modelPath);
    xml->setAttribute("modelHop", s.hop);
    xml->setAttribute("modelNFft", s.nFft);
    auto* params = xml->getChildByName("PARAMETERS");
    if (params == nullptr)
        params = xml->createNewChildElement("PARAMETERS");
    setParam(*params, "mix", s.mix);
    setParam(*params, "bypass", s.bypass ? 1.0 : 0.0);
    setParam(*params, "highBand", s.highBandPass ? 1.0 : 0.0);
    setPluginXml(plugin, *xml);
}

juce::String statusMessage(juce::AudioPluginInstance& plugin) {
    auto xml = getPluginXml(plugin);
    auto* status = xml ? xml->getChildByName("Status") : nullptr;
    return status ? status->getStringAttribute("message") : juce::String("<no status>");
}

// ---- Rendering ----

// Prepares the plugin at `rate` (a full reset) and renders `in` followed by
// `tail` zeros, in blocks of the sizes nextBlock() returns
Audio render(juce::AudioPluginInstance& plugin, double rate, const Audio& in, int tail,
             const std::function<int()>& nextBlock) {
    plugin.releaseResources();
    plugin.setRateAndBufferSizeDetails(rate, 4096);
    plugin.prepareToPlay(rate, 4096);
    const int channels = int(in.size());
    const int total = int(in[0].size()) + tail;
    Audio out(size_t(channels), std::vector<float>(size_t(total), 0.0f));
    juce::MidiBuffer midi;
    for (int pos = 0; pos < total;) {
        const int n = std::min(std::clamp(nextBlock(), 1, 4096), total - pos);
        juce::AudioBuffer<float> buffer(channels, n);
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < n; ++i) {
                const size_t k = size_t(pos + i);
                buffer.setSample(c, i, k < in[size_t(c)].size() ? in[size_t(c)][k] : 0.0f);
            }
        plugin.processBlock(buffer, midi);
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < n; ++i)
                out[size_t(c)][size_t(pos + i)] = buffer.getSample(c, i);
        pos += n;
    }
    return out;
}

std::function<int()> fixedBlocks(int n) {
    return [n] { return n; };
}

std::function<int()> varyingBlocks(int seed) {
    auto random = std::make_shared<juce::Random>(seed);
    return [random] { return 1 + random->nextInt(4096); };
}

std::function<int()> pattern(std::vector<int> sizes) {
    auto i = std::make_shared<size_t>(0);
    return [sizes, i] { return sizes[(*i)++ % sizes.size()]; };
}

Audio noise(int channels, int n, int seed) {
    juce::Random random(seed);
    Audio a(static_cast<size_t>(channels), std::vector<float>(static_cast<size_t>(n)));
    for (auto& ch : a)
        for (size_t i = 0; i < ch.size(); ++i)
            ch[i] = 0.3f * std::sin(0.031f * float(i)) + 0.2f * (random.nextFloat() * 2.0f - 1.0f);
    return a;
}

// max |a - b| / max(1, max |b|) over the compared range
double relErr(const std::vector<float>& a, size_t aStart, const std::vector<float>& b, size_t bStart, size_t n) {
    double diff = 0.0, peak = 1.0;
    for (size_t i = 0; i < n; ++i) {
        const double x = a[aStart + i], y = b[bStart + i];
        diff = std::max(diff, std::fabs(x - y));
        peak = std::max(peak, std::fabs(y));
    }
    return diff / peak;
}

double relErr(const Audio& a, const Audio& b) {
    double worst = 0.0;
    for (size_t c = 0; c < a.size(); ++c)
        worst = std::max(worst, relErr(a[c], 0, b[c], 0, a[c].size()));
    return worst;
}

// out == in delayed by `latency`, bit for bit
bool isDelayedInput(const Audio& out, const Audio& in, int latency, juce::String& where) {
    for (size_t c = 0; c < out.size(); ++c)
        for (size_t i = 0; i < out[c].size(); ++i) {
            const long k = long(i) - latency;
            const float expected = k >= 0 && size_t(k) < in[c].size() ? in[c][size_t(k)] : 0.0f;
            if (!juce::exactlyEqual(out[c][i], expected)) {
                where = "channel " + juce::String(int(c)) + ", sample " + juce::String(int(i)) + ": " +
                        juce::String(out[c][i]) + " != " + juce::String(expected);
                return false;
            }
        }
    return true;
}

constexpr double kTolerance = 1e-5;

struct GoldenModel {
    std::shared_ptr<const denoiser::Model> model;
    juce::String path;
};

GoldenModel loadGolden(const juce::String& name) {
    GoldenModel g;
    g.path = golden(name + ".model.bin");
    juce::MemoryBlock bytes;
    if (!juce::File(g.path).loadFileAsData(bytes)) {
        fail("cannot read " + g.path);
        return g;
    }
    g.model = denoiser::Model::fromMemory(bytes.getData(), bytes.getSize());
    return g;
}

int expectedLatency(int rate, const denoiser::Model& model) {
    return denoiser::hostLatencySamples(rate, model.info().hop, model.info().nFft - model.info().hop);
}

// ---- Tests ----

void testNoModel(Host& host) {
    std::printf("no model: pass-through with the default model's latency\n");
    for (int channels : {2, 1}) {
        auto plugin = host.create(channels);
        if (!plugin)
            return;
        const Audio in = noise(channels, 3000, 1);
        const Audio out = render(*plugin, 48000.0, in, 800, pattern({512, 1, 80, 333, 512, 7, 256}));
        const int latency = plugin->getLatencySamples();
        // The latency of config.toml's model shape (n_fft 128, hop 80), so that
        // loading such a model doesn't change the host's compensation
        check(latency == 669, "no-model latency at 48 kHz should be 669, got " + juce::String(latency));
        juce::String where;
        check(isDelayedInput(out, in, latency, where), "no model, " + juce::String(channels) + " ch: " + where);
        check(statusMessage(*plugin).contains("No model"), "no-model message: " + statusMessage(*plugin));
    }
}

void testLatency(Host& host) {
    std::printf("latency = hostLatencySamples at 16, 44.1, 48, 96 kHz\n");
    auto plugin = host.create(2);
    if (!plugin)
        return;
    for (const char* name : {"mlp_magnitude_default", "mlp_magnitude_short"}) {
        const GoldenModel g = loadGolden(name);
        if (!g.model)
            continue;
        apply(*plugin, {g.path});
        for (int rate : {16000, 44100, 48000, 96000}) {
            plugin->releaseResources();
            plugin->setRateAndBufferSizeDetails(rate, 512);
            plugin->prepareToPlay(rate, 512);
            const int expected = expectedLatency(rate, *g.model);
            check(plugin->getLatencySamples() == expected, juce::String(name) + " at " + juce::String(rate) +
                                                               ": latency " + juce::String(plugin->getLatencySamples()) +
                                                               ", expected " + juce::String(expected));
        }
    }
    // An unsupported rate: pass-through, no latency, a message
    const Audio in = noise(2, 1000, 2);
    const Audio out = render(*plugin, 8000.0, in, 0, fixedBlocks(100));
    juce::String where;
    check(plugin->getLatencySamples() == 0, "8 kHz latency should be 0");
    check(isDelayedInput(out, in, 0, where), "8 kHz should pass through: " + where);
    check(statusMessage(*plugin).contains("not supported"), "8 kHz message: " + statusMessage(*plugin));
}

// At the model's rate the plugin's output is the core's whole-clip result,
// shifted by the reported latency (the first `latency` samples are zero)
void testNullAgainstCore(Host& host) {
    for (const char* name : {"mlp_magnitude_default", "snn_complex_default", "tcn_magnitude_default"}) {
        std::printf("null test at 16 kHz: %s\n", name);
        const GoldenModel g = loadGolden(name);
        if (!g.model)
            continue;
        const denoiser::TensorFile vectors =
            denoiser::TensorFile::load(golden(juce::String(name) + ".vectors.bin").toStdString());
        const denoiser::Tensor& input = vectors.get("input");
        // The golden clip and a longer synthetic one, in two channels
        Audio in{std::vector<float>(input.data, input.data + input.size()), noise(1, 16000, 3)[0]};
        in[0].resize(in[1].size(), 0.0f);

        auto plugin = host.create(2);
        if (!plugin)
            return;
        apply(*plugin, {g.path});
        const int latency = expectedLatency(16000, *g.model);
        const Audio out = render(*plugin, 16000.0, in, latency, varyingBlocks(4));
        check(plugin->getLatencySamples() == latency, "latency at 16 kHz");
        check(statusMessage(*plugin).isEmpty(), "unexpected message: " + statusMessage(*plugin));
        for (size_t c = 0; c < 2; ++c) {
            auto network = g.model->makeNetwork();
            const std::vector<float> ref =
                denoiser::denoiseClip(g.model->info(), *network, in[c].data(), in[c].size());
            const std::vector<float> zeros(size_t(latency), 0.0f);
            const double head = relErr(out[c], 0, zeros, 0, size_t(latency));
            const double err = relErr(out[c], size_t(latency), ref, 0, ref.size());
            std::printf("  channel %d: error %.2e (before the latency: %.2e)\n", int(c), err, head);
            check(err <= kTolerance && head <= kTolerance,
                  juce::String(name) + ": plugin != core shifted by the latency, error " + juce::String(err));
        }
    }
}

void testBlockSizes(Host& host) {
    std::printf("block sizes 16 ... 4096 and varying, 48 kHz stereo\n");
    const GoldenModel g = loadGolden("mlp_complex_default");
    auto plugin = host.create(2);
    if (!plugin || !g.model)
        return;
    apply(*plugin, {g.path});
    const Audio in = noise(2, 24000, 5);
    const Audio ref = render(*plugin, 48000.0, in, 700, fixedBlocks(16));
    double worst = 0.0;
    for (const auto& blocks : {fixedBlocks(17), fixedBlocks(64), fixedBlocks(256), fixedBlocks(333),
                               fixedBlocks(1024), fixedBlocks(4096), varyingBlocks(6), varyingBlocks(7),
                               pattern({1, 4096, 2, 100, 3000})}) {
        const Audio out = render(*plugin, 48000.0, in, 700, blocks);
        worst = std::max(worst, relErr(out, ref));
    }
    std::printf("  worst difference to blocks of 16: %.2e\n", worst);
    check(worst <= kTolerance, "block sizes change the output: " + juce::String(worst));
}

void testBypassMixState(Host& host) {
    std::printf("bypass, mix, state, 48 kHz stereo\n");
    const GoldenModel g = loadGolden("mlp_mask_default");
    auto plugin = host.create(2);
    if (!plugin || !g.model)
        return;
    const Audio in = noise(2, 12000, 8);
    const int latency = expectedLatency(48000, *g.model);
    const auto blocks = [] { return pattern({512, 77, 1024}); };
    juce::String where;

    apply(*plugin, {g.path, 1.0f});
    const Audio wet = render(*plugin, 48000.0, in, latency, blocks());
    check(relErr(wet, in) > 1e-3, "the model should change the audio");

    apply(*plugin, {g.path, 1.0f, true});
    const Audio bypassed = render(*plugin, 48000.0, in, latency, blocks());
    check(plugin->getLatencySamples() == latency, "bypass keeps the latency");
    check(isDelayedInput(bypassed, in, latency, where), "bypass != input delayed by the latency: " + where);

    apply(*plugin, {g.path, 0.0f});
    const Audio dry = render(*plugin, 48000.0, in, latency, blocks());
    check(isDelayedInput(dry, in, latency, where), "mix 0 != bypass: " + where);

    apply(*plugin, {g.path, 0.5f});
    const Audio half = render(*plugin, 48000.0, in, latency, blocks());
    Audio mean = wet;
    for (size_t c = 0; c < mean.size(); ++c)
        for (size_t i = 0; i < mean[c].size(); ++i)
            mean[c][i] = 0.5f * (wet[c][i] + bypassed[c][i]);
    const double mixErr = relErr(half, mean);
    std::printf("  mix 0.5 vs mean of wet and dry: %.2e\n", mixErr);
    check(mixErr <= kTolerance, "mix 0.5 != mean: " + juce::String(mixErr));

    // The host's own bypass parameter is the plugin's bypass parameter
    apply(*plugin, {g.path, 1.0f});
    if (auto* bypass = plugin->getBypassParameter()) {
        bypass->setValue(1.0f);
        render(*plugin, 48000.0, noise(2, 64, 1), 0, fixedBlocks(64));  // delivers the change
        const Audio hostBypassed = render(*plugin, 48000.0, in, latency, blocks());
        check(isDelayedInput(hostBypassed, in, latency, where), "host bypass != delayed input: " + where);
        bypass->setValue(0.0f);
        render(*plugin, 48000.0, noise(2, 64, 1), 0, fixedBlocks(64));
        const Audio back = render(*plugin, 48000.0, in, latency, blocks());
        check(relErr(back, wet) <= kTolerance, "after host bypass off: not the wet output");
    } else {
        fail("the plugin exposes no bypass parameter to the host");
    }

    // Save -> new instance -> restore: same output
    apply(*plugin, {g.path, 0.7f, false, true});
    const Audio before = render(*plugin, 48000.0, in, latency, blocks());
    juce::MemoryBlock saved;
    plugin->getStateInformation(saved);
    auto restored = host.create(2);
    if (!restored)
        return;
    restored->setStateInformation(saved.getData(), int(saved.getSize()));
    const Audio after = render(*restored, 48000.0, in, latency, blocks());
    const double stateErr = relErr(after, before);
    std::printf("  restored instance vs original: %.2e\n", stateErr);
    check(stateErr <= kTolerance, "restored state gives other output: " + juce::String(stateErr));
    check(relErr(before, wet) > 1e-4, "mix 0.7 + high band should differ from the wet output");
}

void testUnusableModels(Host& host) {
    std::printf("missing and offline-only models: pass-through with a message\n");
    auto plugin = host.create(2);
    if (!plugin)
        return;
    const Audio in = noise(2, 4000, 9);
    juce::String where;

    // Missing file: the shape saved with it keeps the latency it had
    apply(*plugin, {golden("no_such_model.model.bin"), 1.0f, false, false, 64, 128});
    Audio out = render(*plugin, 48000.0, in, 800, fixedBlocks(480));
    const int latency = denoiser::hostLatencySamples(48000, 64, 64);
    check(plugin->getLatencySamples() == latency, "missing model latency " +
                                                      juce::String(plugin->getLatencySamples()) + ", expected " +
                                                      juce::String(latency));
    check(isDelayedInput(out, in, plugin->getLatencySamples(), where), "missing model: " + where);
    check(statusMessage(*plugin).contains("not found"), "missing model message: " + statusMessage(*plugin));

    // A broken file
    apply(*plugin, {golden("index.json")});
    out = render(*plugin, 48000.0, in, 800, fixedBlocks(480));
    check(isDelayedInput(out, in, plugin->getLatencySamples(), where), "broken model: " + where);
    check(statusMessage(*plugin).contains("Could not load"), "broken model message: " + statusMessage(*plugin));

    // Centred (offline-only) model
    apply(*plugin, {golden("offline_mlp_mask_centred.model.bin")});
    out = render(*plugin, 48000.0, in, 800, fixedBlocks(480));
    check(plugin->getLatencySamples() == 669, "centred model latency");
    check(isDelayedInput(out, in, plugin->getLatencySamples(), where), "centred model: " + where);
    check(statusMessage(*plugin).contains("centred"), "centred model message: " + statusMessage(*plugin));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::printf("usage: %s <plugin.vst3> <golden dir>\n", argv[0]);
        return 2;
    }
    goldenDir = juce::File::getCurrentWorkingDirectory().getChildFile(juce::String::fromUTF8(argv[2])).getFullPathName();
    juce::ScopedJuceInitialiser_GUI juce;  // a message thread, as in a host

    Host host;
    juce::OwnedArray<juce::PluginDescription> found;
    host.format.findAllTypesForFile(found, juce::String::fromUTF8(argv[1]));
    if (found.isEmpty()) {
        std::printf("FAIL: no plugin found in %s\n", argv[1]);
        return 1;
    }
    host.description = *found.getFirst();
    std::printf("found: %s by %s, %d in / %d out, ARA: %s\n", host.description.name.toRawUTF8(),
                host.description.manufacturerName.toRawUTF8(), host.description.numInputChannels,
                host.description.numOutputChannels, host.description.hasARAExtension ? "yes" : "no");
#if DENOISER_EXPECT_ARA
    if (!host.description.hasARAExtension)
        fail("plugin was built with ARA but the host sees no ARA factory");
#endif

    try {
        testNoModel(host);
        testLatency(host);
        testNullAgainstCore(host);
        testBlockSizes(host);
        testBypassMixState(host);
        testUnusableModels(host);
    } catch (const std::exception& e) {
        fail(juce::String("exception: ") + e.what());
    }

    std::printf(failures == 0 ? "all checks passed\n" : "%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
