#include "PluginProcessor.h"

#include "PluginEditor.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace {

constexpr int kStateVersion = 1;
constexpr const char* kStateTag = "ENM156Denoiser";
constexpr double kSmoothingSeconds = 0.02;  // mix, bypass and high-band ramps
constexpr double kCrossfadeSeconds = 0.01;  // model swaps

juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout() {
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{DenoiserProcessor::kMixId, 1}, "Mix", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f),
        1.0f,
        juce::AudioParameterFloatAttributes()
            .withLabel("%")
            .withStringFromValueFunction([](float v, int) { return juce::String(juce::roundToInt(v * 100.0f)); })
            .withValueFromStringFunction([](const juce::String& s) {
                return juce::jlimit(0.0f, 1.0f, s.getFloatValue() / 100.0f);
            })));
    layout.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{DenoiserProcessor::kBypassId, 1},
                                                          "Bypass", false));
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{DenoiserProcessor::kHighBandId, 1}, "High band",
        juce::StringArray{"Drop", "Pass through"}, 0));
    return layout;
}

juce::String noModelMessage() {
    return "No model loaded: audio passes through, delayed by the latency the host already compensates.";
}

}  // namespace

DenoiserProcessor::DenoiserProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters_(*this, nullptr, "PARAMETERS", createParameterLayout()) {
    mixParam_ = parameters_.getRawParameterValue(kMixId);
    bypassParam_ = parameters_.getRawParameterValue(kBypassId);
    highBandParam_ = parameters_.getRawParameterValue(kHighBandId);
    {
        std::lock_guard<std::mutex> lock(controlMutex_);
        loadMessage_ = noModelMessage();
        reportLocked(nullptr);
    }
    startTimerHz(10);
}

DenoiserProcessor::~DenoiserProcessor() {
    stopTimer();
    delete pending_.exchange(nullptr);
    collectGarbage();
    delete active_;
    delete fading_;
}

// ============================================================================
// Control side

juce::AudioProcessorParameter* DenoiserProcessor::getBypassParameter() const {
    return parameters_.getParameter(kBypassId);
}

juce::AudioProcessorEditor* DenoiserProcessor::createEditor() {
    return new DenoiserEditor(*this);
}

void DenoiserProcessor::collectGarbage() {
    for (auto& slot : retired_)
        delete slot.exchange(nullptr, std::memory_order_acq_rel);
}

void DenoiserProcessor::prepareToPlay(double sampleRate, int maximumExpectedSamplesPerBlock) {
#if JucePlugin_Enable_ARA
    prepareToPlayForARA(sampleRate, maximumExpectedSamplesPerBlock, getMainBusNumOutputChannels(),
                        getProcessingPrecision());
#else
    juce::ignoreUnused(maximumExpectedSamplesPerBlock);
#endif
    std::lock_guard<std::mutex> lock(controlMutex_);
    collectGarbage();
    prepared_ = true;
    preparedRate_ = juce::roundToInt(sampleRate);
    preparedChannels_ = juce::jlimit(1, enm156::kMaxChannels, getMainBusNumOutputChannels());
    std::unique_ptr<Engine> engine = Engine::create(model_, preparedRate_, preparedChannels_, shape_);
    const Engine* report = engine.get();

    // The audio thread is stopped (hosts don't process while preparing); the
    // callback lock makes sure of it with the JUCE VST3 wrapper.
    Engine *oldActive = nullptr, *oldFading = nullptr, *oldPending = nullptr;
    {
        const juce::ScopedLock heldCallbackLock(getCallbackLock());
        oldActive = std::exchange(active_, engine.release());
        oldFading = std::exchange(fading_, nullptr);
        oldPending = pending_.exchange(nullptr);
        fadingDone_ = false;
        holdRemaining_ = fadePos_ = 0;
        sampleRate_ = sampleRate > 0.0 ? sampleRate : 48000.0;
        mixSmooth_.reset(sampleRate_, kSmoothingSeconds);
        highSmooth_.reset(sampleRate_, kSmoothingSeconds);
        snapSmoothers();
        resetRequested_.store(false);
    }
    delete oldActive;
    delete oldFading;
    delete oldPending;
    reportLocked(report);
}

void DenoiserProcessor::releaseResources() {
#if JucePlugin_Enable_ARA
    releaseResourcesForARA();
#endif
    std::lock_guard<std::mutex> lock(controlMutex_);
    prepared_ = false;
    Engine *oldActive = nullptr, *oldFading = nullptr, *oldPending = nullptr;
    {
        const juce::ScopedLock heldCallbackLock(getCallbackLock());
        oldActive = std::exchange(active_, nullptr);
        oldFading = std::exchange(fading_, nullptr);
        oldPending = pending_.exchange(nullptr);
        fadingDone_ = false;
    }
    delete oldActive;
    delete oldFading;
    delete oldPending;
    collectGarbage();
}

void DenoiserProcessor::reset() {
    // May come from any thread: the audio thread resets at its next block
    resetRequested_.store(true);
}

bool DenoiserProcessor::loadModelFile(const juce::File& file) {
    std::lock_guard<std::mutex> lock(controlMutex_);
    loadPathLocked(file.getFullPathName());
    return model_ != nullptr && model_->info().realtime;
}

void DenoiserProcessor::clearModel() {
    std::lock_guard<std::mutex> lock(controlMutex_);
    loadPathLocked({});
}

void DenoiserProcessor::loadPathLocked(const juce::String& path) {
    collectGarbage();
    std::shared_ptr<const denoiser::Model> model;
    juce::String message;
    if (path.isEmpty()) {
        message = noModelMessage();
    } else if (!juce::File::isAbsolutePath(path)) {
        message = "The model path is not absolute: " + path + ". Audio passes through.";
    } else {
        const juce::File file(path);
        juce::MemoryBlock bytes;
        if (!file.existsAsFile()) {
            message = "Model file not found: " + path + ". Audio passes through; choose the file again.";
        } else if (!file.loadFileAsData(bytes)) {
            message = "Could not read the model file " + path + ". Audio passes through.";
        } else {
            try {
                model = denoiser::Model::fromMemory(bytes.getData(), bytes.getSize());
            } catch (const std::exception& error) {
                message = "Could not load " + file.getFileName() + ": " + juce::String::fromUTF8(error.what()) +
                          ". Audio passes through.";
            }
        }
    }

    modelPath_ = path;
    model_ = std::move(model);
    loadMessage_ = message;
    if (model_ && model_->info().realtime)
        shape_ = enm156::ModelShape{model_->info().hop, model_->info().nFft};
    rebuildLocked();
}

void DenoiserProcessor::rebuildLocked() {
    if (!prepared_) {
        reportLocked(nullptr);
        return;
    }
    std::unique_ptr<Engine> engine = Engine::create(model_, preparedRate_, preparedChannels_, shape_);
    reportLocked(engine.get());
    collectGarbage();
    // Hand over; an earlier engine the audio thread hasn't taken yet was never used
    delete pending_.exchange(engine.release(), std::memory_order_acq_rel);
}

void DenoiserProcessor::reportLocked(const Engine* engine) {
    Status s;
    s.modelPath = modelPath_;
    s.description = model_ ? juce::String::fromUTF8(model_->describe().c_str()) : juce::String();
    s.shape = shape_;
    s.prepared = prepared_ && engine != nullptr;
    s.sampleRate = prepared_ ? double(preparedRate_) : 0.0;
    s.denoising = engine != nullptr && engine->denoising();
    s.message = loadMessage_;
    if (s.message.isEmpty() && engine != nullptr)
        s.message = juce::String::fromUTF8(engine->message().c_str());
    if (s.message.isEmpty() && engine == nullptr && model_ && !model_->info().realtime)
        s.message = "This model uses centred (non-causal) frames, so it can't run in real time; audio passes "
                    "through. It can be used in ARA mode.";

    int latency = engine != nullptr ? engine->latencySamples() : 0;
#if JucePlugin_Enable_ARA
    if (isBoundToARA())
        latency = 0;  // ARA playback is aligned; JUCE's VST3 wrapper requires 0
#endif
    s.latencySamples = latency;
    if (engine != nullptr)
        setLatencySamples(latency);

    {
        std::lock_guard<std::mutex> lock(statusMutex_);
        status_ = s;
        statusModel_ = model_;
    }
    sendChangeMessage();
}

DenoiserProcessor::Status DenoiserProcessor::getStatus() const {
    std::lock_guard<std::mutex> lock(statusMutex_);
    return status_;
}

std::shared_ptr<const denoiser::Model> DenoiserProcessor::getModel() const {
    std::lock_guard<std::mutex> lock(statusMutex_);
    return statusModel_;
}

// ---- State ----

void DenoiserProcessor::getStateInformation(juce::MemoryBlock& destData) {
    juce::XmlElement xml(kStateTag);
    xml.setAttribute("version", kStateVersion);
    const Status s = getStatus();
    xml.setAttribute("modelPath", s.modelPath);
    xml.setAttribute("modelHop", s.shape.hop);
    xml.setAttribute("modelNFft", s.shape.nFft);
    if (auto params = parameters_.copyState().createXml())
        xml.addChildElement(params.release());
    // For information only (ignored when restoring): what the instance was doing
    auto* info = xml.createNewChildElement("Status");
    info->setAttribute("denoising", s.denoising);
    info->setAttribute("description", s.description);
    info->setAttribute("message", s.message);
    info->setAttribute("latencySamples", s.latencySamples);
    copyXmlToBinary(xml, destData);
}

void DenoiserProcessor::setStateInformation(const void* data, int sizeInBytes) {
    const std::unique_ptr<juce::XmlElement> xml = getXmlFromBinary(data, sizeInBytes);
    if (xml == nullptr || !xml->hasTagName(kStateTag))
        return;  // not ours (or empty): keep the current state
    if (auto* params = xml->getChildByName(parameters_.state.getType()))
        parameters_.replaceState(juce::ValueTree::fromXml(*params));

    enm156::ModelShape shape{xml->getIntAttribute("modelHop", 80), xml->getIntAttribute("modelNFft", 128)};
    if (!shape.valid())
        shape = enm156::ModelShape{};
    std::lock_guard<std::mutex> lock(controlMutex_);
    shape_ = shape;
    loadPathLocked(xml->getStringAttribute("modelPath"));
    resetRequested_.store(true);  // parameters jumped: no ramps from the old values
}

// ============================================================================
// Audio thread

bool DenoiserProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto out = layouts.getMainOutputChannelSet();
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo())
        return false;
    return layouts.getMainInputChannelSet() == out;
}

void DenoiserProcessor::snapSmoothers() noexcept {
    const bool bypass = bypassParam_->load() >= 0.5f;
    mixSmooth_.setCurrentAndTargetValue(bypass ? 0.0f : juce::jlimit(0.0f, 1.0f, mixParam_->load()));
    highSmooth_.setCurrentAndTargetValue(highBandParam_->load() >= 0.5f ? 1.0f : 0.0f);
}

bool DenoiserProcessor::retire(Engine* engine) noexcept {
    for (auto& slot : retired_) {
        Engine* expected = nullptr;
        if (slot.compare_exchange_strong(expected, engine, std::memory_order_acq_rel))
            return true;
    }
    return false;  // all slots full: try again next block
}

void DenoiserProcessor::takePending() noexcept {
    if (fading_ != nullptr) {
        if (!fadingDone_ || !retire(fading_))
            return;  // still crossfading, or no free slot yet
        fading_ = nullptr;
        fadingDone_ = false;
    }
    Engine* next = pending_.exchange(nullptr, std::memory_order_acq_rel);
    if (next == nullptr)
        return;
    if (active_ == nullptr) {
        active_ = next;
        return;
    }
    fading_ = active_;
    active_ = next;
    fadingDone_ = false;
    holdRemaining_ = next->warmUpSamples();
    fadePos_ = 0;
    fadeLength_ = std::max(1, juce::roundToInt(sampleRate_ * kCrossfadeSeconds));
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
    process(buffer, false);
}

void DenoiserProcessor::processBlockBypassed(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) {
    // Only called by wrappers that don't use getBypassParameter(): the same
    // latency-compensated bypass as the parameter
    juce::ignoreUnused(midi);
    juce::ScopedNoDenormals noDenormals;
#if JucePlugin_Enable_ARA
    if (isBoundToARA())
        return;  // ARA playback has no latency: the input as it is
#endif
    process(buffer, true);
}

void DenoiserProcessor::process(juce::AudioBuffer<float>& buffer, bool forceBypass) noexcept {
    takePending();
    if (resetRequested_.exchange(false)) {
        if (active_ != nullptr)
            active_->reset();
        if (fading_ != nullptr && retire(fading_)) {
            fading_ = nullptr;
            fadingDone_ = false;
        } else if (fading_ != nullptr) {
            fadingDone_ = true;
        }
        snapSmoothers();
    }

    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();
    Engine* engine = active_;
    if (engine == nullptr || engine->channels() > numChannels)
        return;  // not prepared (or a layout it wasn't prepared for): leave the audio as it is
    const int channels = engine->channels();
    for (int c = channels; c < numChannels; ++c)
        buffer.clear(c, 0, numSamples);

    const bool bypass = forceBypass || bypassParam_->load() >= 0.5f;
    mixSmooth_.setTargetValue(bypass ? 0.0f : juce::jlimit(0.0f, 1.0f, mixParam_->load()));
    highSmooth_.setTargetValue(highBandParam_->load() >= 0.5f ? 1.0f : 0.0f);

    float* const* data = buffer.getArrayOfWritePointers();
    for (int offset = 0; offset < numSamples; offset += enm156::kChunk) {
        const int n = std::min(enm156::kChunk, numSamples - offset);
        for (int i = 0; i < n; ++i) {
            wetGain_[size_t(i)] = mixSmooth_.getNextValue();
            highGain_[size_t(i)] = highSmooth_.getNextValue();
        }
        float* io[enm156::kMaxChannels] = {};
        for (int c = 0; c < channels; ++c)
            io[c] = data[c] + offset;

        if (fading_ == nullptr || fadingDone_) {
            engine->process(io, io, n, wetGain_.data(), highGain_.data());
            continue;
        }

        // Crossfade from the old engine to the new one: both run on the same input
        float* old[enm156::kMaxChannels] = {};
        for (int c = 0; c < channels; ++c)
            old[c] = fadeOld_[size_t(c)].data();
        const bool sameChannels = fading_->channels() == channels;
        if (sameChannels)
            fading_->process(io, old, n, wetGain_.data(), highGain_.data());
        engine->process(io, io, n, wetGain_.data(), highGain_.data());
        for (int i = 0; i < n; ++i) {
            if (holdRemaining_ > 0) {
                --holdRemaining_;
                fadeGain_[size_t(i)] = 0.0f;
            } else {
                fadePos_ = std::min(fadePos_ + 1, fadeLength_);
                fadeGain_[size_t(i)] = float(fadePos_) / float(fadeLength_);
            }
        }
        if (sameChannels)
            for (int c = 0; c < channels; ++c)
                for (int i = 0; i < n; ++i) {
                    const float w = fadeGain_[size_t(i)];
                    io[c][i] = old[c][i] + w * (io[c][i] - old[c][i]);
                }
        if (fadePos_ >= fadeLength_) {
            fadingDone_ = true;
            if (retire(fading_)) {
                fading_ = nullptr;
                fadingDone_ = false;
            }
        }
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new DenoiserProcessor();
}
