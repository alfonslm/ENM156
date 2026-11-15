#pragma once
// The ARA side of the plugin: the document controller (one per host
// document, shared by every plugin instance bound to it) and the playback
// renderer (one per bound plugin instance). Compiled only with ARA. See
// README.md "ARA (whole clips)" for the design; AraEngine.h has the parts
// that don't depend on JUCE or ARA.
//
// Threads
// -------
// - Host document thread (normally the message thread): every ARA model
//   call (editing, sources, regions, archives, sample access) and the
//   editor's model / high-band choice. Creates and destroys the host audio
//   readers (ARA allows that only inside these calls).
// - One analysis thread per document controller: reads sources through the
//   host's readers (non-realtime, allowed by ARA), denoises them, publishes
//   the stored audio into AudioSlots, reports progress (atomics, delivered
//   to the host by its notifyModelUpdates()).
// - Audio threads (each bound instance's processBlock): the renderer reads
//   its RenderPlan and the AudioSlots, without allocation or locks.
// mutex_ protects the controller's state; it is never taken by an audio
// thread, and the analysis thread doesn't hold it while working.

#include "AraEngine.h"

#include <juce_audio_processors/juce_audio_processors.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if JucePlugin_Enable_ARA

class DenoiserPlaybackRenderer;

class DenoiserDocumentController final : public juce::ARADocumentControllerSpecialisation {
public:
    DenoiserDocumentController(const ARA::PlugIn::PlugInEntry* entry,
                               const ARA::ARADocumentControllerHostInstance* instance);
    ~DenoiserDocumentController() override;

    // The controller an ARA document controller pointer belongs to
    static DenoiserDocumentController* from(ARA::PlugIn::DocumentController* dc) noexcept {
        return dc == nullptr ? nullptr : getSpecialisedDocumentController<DenoiserDocumentController>(dc);
    }

    // ---- Document settings (host document thread / message thread) ----
    // The model is a document setting: one model for every audio source of
    // the document, chosen in any bound instance's window, saved in the
    // document's ARA archive. Loading reads the file on the calling thread;
    // every source is then analysed again.
    void setModelFile(const juce::String& path);
    // What happens above ~7.5 kHz (also a document setting: it changes the
    // stored audio; no new network run is needed)
    void setHighBand(denoiser::HighBand highBand);

    struct ModelStatus {
        juce::String path;         // "" = none
        juce::String description;  // Model::describe(), "" if none loaded
        juce::String message;      // why there is no model ("" if one is loaded)
        denoiser::HighBand highBand = denoiser::HighBand::Drop;
    };
    ModelStatus getModelStatus() const;

    enum class SourceState { Waiting, Reading, Denoising, Done, Original, Failed };
    struct SourceStatus {
        juce::String name;
        double sampleRate = 0.0;
        int channels = 0;
        std::int64_t length = 0;
        SourceState state = SourceState::Waiting;
        float progress = 0.0f;  // 0..1 while denoising
        juce::String message;   // why the original plays, or an error
    };
    std::vector<SourceStatus> getSourceStatuses() const;
    static juce::String describe(SourceState state);

    // ---- For renderers ----
    void addRenderer(DenoiserPlaybackRenderer* renderer);
    void removeRenderer(DenoiserPlaybackRenderer* renderer);
    // The slot holding `source`'s audio at `rate` (made on first use, which
    // asks the analysis for a version at that rate). Null for an unknown source.
    std::shared_ptr<enm156::ara::AudioSlot> slotFor(juce::ARAAudioSource* source, double rate);

protected:
    juce::ARAPlaybackRenderer* doCreatePlaybackRenderer() noexcept override;
    bool doRestoreObjectsFromStream(juce::ARAInputStream& input,
                                    const juce::ARARestoreObjectsFilter* filter) noexcept override;
    bool doStoreObjectsToStream(juce::ARAOutputStream& output,
                                const juce::ARAStoreObjectsFilter* filter) noexcept override;

    void willBeginEditing(juce::ARADocument*) override;
    void didEndEditing(juce::ARADocument*) override;
    void didAddAudioSourceToDocument(juce::ARADocument*, juce::ARAAudioSource* source) override;
    void willDestroyAudioSource(juce::ARAAudioSource* source) override;
    void didUpdateAudioSourceProperties(juce::ARAAudioSource* source) override;
    void doUpdateAudioSourceContent(juce::ARAAudioSource* source, juce::ARAContentUpdateScopes scopes) override;
    void willEnableAudioSourceSamplesAccess(juce::ARAAudioSource* source, bool enable) override;
    void didEnableAudioSourceSamplesAccess(juce::ARAAudioSource* source, bool enable) override;

private:
    // What a source needs done (bits); each implies the ones after it
    enum Need : unsigned { kRead = 1, kDenoise = 2, kPublish = 4, kAll = 7 };

    struct Source;

    void loadModelLocked(const juce::String& path);
    void requestLocked(Source& source, unsigned needs);
    void updateWorkingLocked(Source& source);
    void captureProperties(Source& source);
    void notifyHostContentChanged();
    void rebuildPlans();
    void run();
    void runJob(Source& source, unsigned job);
    enum class ReadResult { Ok, Cancelled, AccessLost, Failed };
    ReadResult readSource(Source& source, enm156::ara::Planar& out);
    void setStatus(Source& source, SourceState state, float progress, const juce::String& message);
    void publishAll(Source& source, denoiser::HighBand highBand);

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::atomic<bool> stop_{false};
    bool editing_ = false;

    // Settings
    juce::String modelPath_;
    std::shared_ptr<const denoiser::Model> model_;
    juce::String modelMessage_;
    denoiser::HighBand highBand_ = denoiser::HighBand::Drop;

    std::map<juce::ARAAudioSource*, std::shared_ptr<Source>> sources_;
    std::mutex renderersMutex_;  // taken before a renderer's planMutex_, which is taken before mutex_
    std::vector<DenoiserPlaybackRenderer*> renderers_;

    std::thread worker_;
};

class DenoiserPlaybackRenderer final : public juce::ARAPlaybackRenderer {
public:
    DenoiserPlaybackRenderer(ARA::PlugIn::DocumentController* dc, DenoiserDocumentController& owner);
    ~DenoiserPlaybackRenderer() override;

    void prepareToPlay(double sampleRate, int maximumSamplesPerBlock, int numChannels,
                       juce::AudioProcessor::ProcessingPrecision precision,
                       AlwaysNonRealtime alwaysNonRealtime) override;
    void releaseResources() override;
    void reset() override;
    bool processBlock(juce::AudioBuffer<float>& buffer, juce::AudioProcessor::Realtime realtime,
                      const juce::AudioPlayHead::PositionInfo& positionInfo) noexcept override;
    using juce::ARAPlaybackRenderer::processBlock;

    // The owning processor's mix and bypass parameters (per instance)
    void setParameters(const std::atomic<float>* mix, const std::atomic<float>* bypass) noexcept;
    // Audio thread: the host bypassed the instance through processBlockBypassed
    void setForceBypass(bool bypass) noexcept { forceBypass_ = bypass; }

    // Host document thread: the regions or their places changed
    void rebuildPlan();

    // Longest a non-realtime render waits per block for analysis that is still running
    static constexpr double kMaxOfflineWaitSeconds = 120.0;

private:
    void collectGarbage();
    void takePending() noexcept;

    DenoiserDocumentController& owner_;
    std::mutex planMutex_;  // control side only
    double sampleRate_ = 0.0;
    bool prepared_ = false;

    std::atomic<enm156::ara::RenderPlan*> pending_{nullptr};
    std::array<std::atomic<enm156::ara::RenderPlan*>, 8> retired_{};
    enm156::ara::RenderPlan* active_ = nullptr;  // audio thread (and prepare/release while stopped)

    const std::atomic<float>* mixParam_ = nullptr;
    const std::atomic<float>* bypassParam_ = nullptr;
    bool forceBypass_ = false;
    juce::SmoothedValue<float> mix_;
    std::array<float, 256> gain_{};
};

#endif
