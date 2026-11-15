#pragma once
// The plugin's AudioProcessor: the real-time denoiser (VST3 effect, mono or
// stereo, input layout == output layout), and the ARA entry points (the ARA
// playback renderer is a later issue). See ../README.md, "The VST3 plugin".
//
// Threads
// -------
// - Control threads (the message thread, a host's state thread, the editor):
//   loadModelFile / clearModel / setStateInformation / prepareToPlay /
//   releaseResources. They are serialised by controlMutex_, load and build
//   everything (file reading, parsing, ChannelDenoisers, resamplers) and hand
//   a complete Engine (RealtimeEngine.h) to the audio thread through one
//   atomic pointer (pending_). They also delete engines the audio thread has
//   finished with (collectGarbage(), also from a 10 Hz timer).
// - The audio thread (processBlock): takes a pending engine at the start of a
//   block, crossfades from the old one (see "Swapping"), puts the old one in
//   a retired_ slot. It never allocates, locks, frees, reads files or logs.
//   The only lock on that path is the one the VST3 wrapper itself takes
//   around processBlock (getCallbackLock()), which prepareToPlay /
//   releaseResources also take to replace the engine directly.
//
// Swapping (model load or clear, while audio runs)
// -------------------------------------------------
// The old engine keeps running and stays audible until the new one has been
// fed for its latency plus its widest analysis window (so its output depends
// only on real input), then a 10 ms linear crossfade switches to it. No gap
// and no click; CPU doubles for those ~30 ms. The host is told the new
// latency (setLatencySamples) by the control thread when the engine is
// handed over; it only changes if the new model's hop / n_fft differ.
// prepareToPlay and reset() start from silence instead (a full reset).

#include "RealtimeEngine.h"

#include <juce_audio_processors/juce_audio_processors.h>

#include <array>
#include <atomic>
#include <memory>
#include <mutex>

class DenoiserProcessor final : public juce::AudioProcessor,
                                public juce::ChangeBroadcaster,
                                private juce::Timer
#if JucePlugin_Enable_ARA
    , public juce::AudioProcessorARAExtension
#endif
{
public:
    // Parameter IDs (also the names in the saved state)
    static constexpr const char* kMixId = "mix";
    static constexpr const char* kBypassId = "bypass";
    static constexpr const char* kHighBandId = "highBand";  // 0 = drop, 1 = pass through

    DenoiserProcessor();
    ~DenoiserProcessor() override;

    // ---- AudioProcessor ----
    void prepareToPlay(double sampleRate, int maximumExpectedSamplesPerBlock) override;
    void releaseResources() override;
    void reset() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override;
    void processBlockBypassed(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override;
    using juce::AudioProcessor::processBlock;
    using juce::AudioProcessor::processBlockBypassed;
    juce::AudioProcessorParameter* getBypassParameter() const override;

    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override;

    const juce::String getName() const override { return "ENM156 Denoiser"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Default"; }  // the VST3 validator wants a name
    void changeProgramName(int, const juce::String&) override {}

    // State: the model file path (referenced, not embedded), the shape of the
    // last model that ran (so a missing file restores the same latency), the
    // parameters. See README.md "State".
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // ---- Control: any thread except the audio thread ----

    // Loads a model file and, if prepared, hands a new engine to the audio
    // thread. Blocks while the file is read and the model built (tens of ms
    // for config.toml-sized models). The path is remembered (and saved) even
    // if loading fails; the reason is in getStatus().message. Returns true if
    // the model now runs in real time.
    bool loadModelFile(const juce::File& file);
    // No model: pass-through with the same latency
    void clearModel();

    struct Status {
        juce::String modelPath;     // as chosen / restored ("" = none)
        juce::String description;   // Model::describe() of the loaded model, "" if none
        juce::String message;       // why audio passes through ("" while denoising)
        bool denoising = false;     // a model runs in real time
        bool prepared = false;      // prepareToPlay has been called (else latency is a forecast)
        double sampleRate = 0.0;    // of the last prepareToPlay
        int latencySamples = 0;     // what the host was told (0 when bound to ARA)
        enm156::ModelShape shape;   // the hop / n_fft the no-model latency follows
    };
    Status getStatus() const;

    // The loaded model, also an offline-only (centred) one that can't run
    // live, or null. Immutable and shareable across threads: the hook for the
    // ARA analysis. getStatus().modelPath says which file it came from.
    std::shared_ptr<const denoiser::Model> getModel() const;

    juce::AudioProcessorValueTreeState& parameters() noexcept { return parameters_; }
    float getMix() const noexcept { return mixParam_->load(); }
    bool getBypass() const noexcept { return bypassParam_->load() >= 0.5f; }
    denoiser::HighBand getHighBand() const noexcept {
        return highBandParam_->load() >= 0.5f ? denoiser::HighBand::PassThrough : denoiser::HighBand::Drop;
    }

    // Frees engines the audio thread has finished with. Called by every
    // control call and a 10 Hz timer; tests call it directly.
    void collectGarbage();

private:
    using Engine = enm156::Engine;

    void timerCallback() override { collectGarbage(); }

    // controlMutex_ held:
    void loadPathLocked(const juce::String& path);
    void rebuildLocked();
    void reportLocked(const Engine* engine);

    // Audio thread:
    void process(juce::AudioBuffer<float>& buffer, bool forceBypass) noexcept;
    void takePending() noexcept;
    bool retire(Engine* engine) noexcept;
    void snapSmoothers() noexcept;

    juce::AudioProcessorValueTreeState parameters_;
    std::atomic<float>* mixParam_ = nullptr;
    std::atomic<float>* bypassParam_ = nullptr;
    std::atomic<float>* highBandParam_ = nullptr;

    // ---- control side (controlMutex_) ----
    std::mutex controlMutex_;
    std::shared_ptr<const denoiser::Model> model_;  // the loaded model (any kind)
    juce::String modelPath_;
    juce::String loadMessage_;          // why model_ is null ("" if it isn't)
    enm156::ModelShape shape_;
    bool prepared_ = false;
    int preparedRate_ = 0;
    int preparedChannels_ = 0;

    // ---- the status for the UI / getStatus (statusMutex_, never taken by the audio thread) ----
    mutable std::mutex statusMutex_;
    Status status_;
    std::shared_ptr<const denoiser::Model> statusModel_;

    // ---- hand-over between control threads and the audio thread ----
    std::atomic<Engine*> pending_{nullptr};
    static constexpr int kRetiredSlots = 8;
    std::array<std::atomic<Engine*>, kRetiredSlots> retired_{};
    std::atomic<bool> resetRequested_{false};

    // ---- audio thread only (and prepareToPlay / releaseResources under the callback lock) ----
    Engine* active_ = nullptr;
    Engine* fading_ = nullptr;       // the engine being faded out, or null
    bool fadingDone_ = false;        // fading_ is silent but couldn't be retired yet
    int holdRemaining_ = 0;          // samples before the crossfade starts
    int fadePos_ = 0, fadeLength_ = 1;
    double sampleRate_ = 48000.0;
    juce::SmoothedValue<float> mixSmooth_, highSmooth_;
    std::array<float, enm156::kChunk> wetGain_{}, highGain_{}, fadeGain_{};
    std::array<std::array<float, enm156::kChunk>, enm156::kMaxChannels> fadeOld_{};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DenoiserProcessor)
};
