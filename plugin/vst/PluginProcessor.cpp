#include "PluginProcessor.h"

#include "denoiser/core.h"

DenoiserProcessor::DenoiserProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)) {}

void DenoiserProcessor::prepareToPlay(double sampleRate, int maximumExpectedSamplesPerBlock) {
#if JucePlugin_Enable_ARA
    prepareToPlayForARA(sampleRate, maximumExpectedSamplesPerBlock,
                        getMainBusNumOutputChannels(), getProcessingPrecision());
#else
    juce::ignoreUnused(sampleRate, maximumExpectedSamplesPerBlock);
#endif
    // The real-time denoiser will report its latency here with setLatencySamples()
    setLatencySamples(0);
}

void DenoiserProcessor::releaseResources() {
#if JucePlugin_Enable_ARA
    releaseResourcesForARA();
#endif
}

void DenoiserProcessor::reset() {}

bool DenoiserProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto out = layouts.getMainOutputChannelSet();
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo())
        return false;
    return layouts.getMainInputChannelSet() == out;
}

void DenoiserProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) {
    juce::ignoreUnused(midi);
    juce::ScopedNoDenormals noDenormals;

#if JucePlugin_Enable_ARA
    // Bound to an ARA document: the ARA playback renderer fills the buffer.
    // Its default renderer (no document controller logic yet) outputs silence.
    if (processBlockForARA(buffer, isRealtime(), getPlayHead()))
        return;
#endif

    // Pass-through: input and output share the buffer, so there is nothing to
    // copy. The call into the core stands in for the denoiser.
    const int numSamples = buffer.getNumSamples();
    for (int ch = 0; ch < getTotalNumInputChannels(); ++ch)
        denoiser::passThrough(buffer.getReadPointer(ch), buffer.getWritePointer(ch),
                              static_cast<std::size_t>(numSamples));
    for (int ch = getTotalNumInputChannels(); ch < getTotalNumOutputChannels(); ++ch)
        buffer.clear(ch, 0, numSamples);
}

void DenoiserProcessor::getStateInformation(juce::MemoryBlock& destData) {
    destData.reset();  // no state yet (later: the model file path)
}

void DenoiserProcessor::setStateInformation(const void* data, int sizeInBytes) {
    juce::ignoreUnused(data, sizeInBytes);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new DenoiserProcessor();
}
