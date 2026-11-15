// Loads the built VST3 the way a host does (JUCE's VST3 hosting, no GUI) and
// checks that it passes audio through unchanged, in mono and stereo and with
// odd block sizes. Run by ctest as `plugin_passthrough`.
//
//   denoiser_plugin_host_test "<path to>/ENM156 Denoiser.vst3"
#include <juce_audio_processors_headless/juce_audio_processors_headless.h>
#include <juce_events/juce_events.h>

#include <cstdio>
#include <memory>

namespace {

int failures = 0;

void fail(const juce::String& message) {
    std::printf("FAIL: %s\n", message.toRawUTF8());
    ++failures;
}

void checkPassThrough(juce::AudioPluginInstance& plugin, const juce::AudioChannelSet& layout) {
    const int channels = layout.size();
    std::printf("layout: %s\n", layout.getDescription().toRawUTF8());

    juce::AudioProcessor::BusesLayout buses;
    buses.inputBuses.add(layout);
    buses.outputBuses.add(layout);
    if (!plugin.setBusesLayout(buses)) {
        fail("layout not accepted: " + layout.getDescription());
        return;
    }

    const double sampleRate = 48000.0;
    const int maxBlock = 512;
    plugin.setRateAndBufferSizeDetails(sampleRate, maxBlock);
    plugin.prepareToPlay(sampleRate, maxBlock);
    if (plugin.getLatencySamples() != 0)
        fail("latency should be 0 for pass-through, got " + juce::String(plugin.getLatencySamples()));

    juce::Random random(42);
    juce::MidiBuffer midi;
    const int blockSizes[] = {512, 1, 80, 333, 512, 7, 256};
    int block = 0;
    for (int size : blockSizes) {
        juce::AudioBuffer<float> buffer(channels, size);
        for (int ch = 0; ch < channels; ++ch)
            for (int i = 0; i < size; ++i)
                buffer.setSample(ch, i, random.nextFloat() * 2.0f - 1.0f);
        juce::AudioBuffer<float> expected(buffer);

        plugin.processBlock(buffer, midi);

        for (int ch = 0; ch < channels; ++ch)
            for (int i = 0; i < size; ++i)
                if (!juce::exactlyEqual(buffer.getSample(ch, i), expected.getSample(ch, i))) {
                    fail("block " + juce::String(block) + " (size " + juce::String(size) + "), channel " +
                         juce::String(ch) + ", sample " + juce::String(i) + ": " +
                         juce::String(buffer.getSample(ch, i)) + " != " +
                         juce::String(expected.getSample(ch, i)));
                    ch = channels;
                    break;
                }
        ++block;
    }
    plugin.releaseResources();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::printf("usage: %s <plugin.vst3>\n", argv[0]);
        return 2;
    }
    juce::ScopedJuceInitialiser_GUI juce;  // a message thread, as in a host

    juce::VST3PluginFormatHeadless format;
    juce::OwnedArray<juce::PluginDescription> found;
    format.findAllTypesForFile(found, juce::String::fromUTF8(argv[1]));
    if (found.isEmpty()) {
        std::printf("FAIL: no plugin found in %s\n", argv[1]);
        return 1;
    }
    const auto& description = *found.getFirst();
    std::printf("found: %s by %s, %d in / %d out, ARA: %s\n", description.name.toRawUTF8(),
                description.manufacturerName.toRawUTF8(), description.numInputChannels,
                description.numOutputChannels, description.hasARAExtension ? "yes" : "no");
#if DENOISER_EXPECT_ARA
    if (!description.hasARAExtension)
        fail("plugin was built with ARA but the host sees no ARA factory");
#endif

    juce::String error;
    std::unique_ptr<juce::AudioPluginInstance> plugin =
        format.createInstanceFromDescription(description, 48000.0, 512, error);
    if (!plugin) {
        std::printf("FAIL: could not create plugin: %s\n", error.toRawUTF8());
        return 1;
    }

    checkPassThrough(*plugin, juce::AudioChannelSet::stereo());
    checkPassThrough(*plugin, juce::AudioChannelSet::mono());
    plugin.reset();

    std::printf(failures == 0 ? "all checks passed\n" : "%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
