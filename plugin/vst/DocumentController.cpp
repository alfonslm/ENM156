// ARA 2: document controller (analysis of whole audio sources) and playback
// renderer. Compiled only when the plugin is built with ARA
// (DENOISER_ENABLE_ARA). See DocumentController.h and README.md "ARA".
#include "DocumentController.h"

#if JucePlugin_Enable_ARA

#include <algorithm>
#include <chrono>
#include <cmath>
#include <new>

using enm156::ara::AudioSlot;
using enm156::ara::Outcome;
using enm156::ara::Planar;
using enm156::ara::RenderPlan;

namespace {

// The archive: a small binary record of the document settings
constexpr int kArchiveMagic = 0x454e4d44;  // "ENMD"
constexpr int kArchiveVersion = 1;
constexpr std::int64_t kReadChunk = 1 << 16;

int rateKey(double rate) {
    return int(std::lround(rate));
}

// Test hook (headless ARA host test only): while the file named by this
// environment variable exists, the analysis waits before running the
// network, so that "playback before analysis finishes" can be observed
// deterministically. Unset in normal use.
juce::File testHoldFile() {
    static const juce::String path = juce::SystemStats::getEnvironmentVariable("ENM156_ARA_TEST_HOLD", {});
    return path.isEmpty() ? juce::File() : juce::File(path);
}

}  // namespace

struct DenoiserDocumentController::Source {
    // Host document thread; null once the source is being destroyed. The
    // analysis thread uses it only for progress notifications, under araMutex.
    juce::ARAAudioSource* ara = nullptr;
    std::mutex araMutex;
    // The host's reader: created and destroyed on the host document thread
    // (where ARA allows it), used by the analysis thread, both under readerMutex
    std::mutex readerMutex;
    std::unique_ptr<ARA::PlugIn::HostAudioReader> reader;

    // mutex_
    juce::String name;
    double rate = 0.0;
    int channels = 0;
    std::int64_t length = 0;
    bool eligible = false;  // the host ended the edit that created it
    bool accessEnabled = false;
    bool destroyed = false;
    bool running = false;   // the analysis thread works on it
    unsigned needs = kAll;
    std::map<int, std::shared_ptr<AudioSlot>> slots;  // by playback rate
    SourceState state = SourceState::Waiting;
    float progress = 0.0f;
    juce::String message;

    std::atomic<bool> cancel{false};

    // Analysis thread only
    enm156::ara::SourceAnalysis analysis;
};

// ============================================================================
// Document controller

DenoiserDocumentController::DenoiserDocumentController(const ARA::PlugIn::PlugInEntry* entry,
                                                       const ARA::ARADocumentControllerHostInstance* instance)
    : ARADocumentControllerSpecialisation(entry, instance) {
    modelMessage_ = "No model chosen: the original audio plays. Choose a model in the plugin's window.";
    worker_ = std::thread([this] { run(); });
}

DenoiserDocumentController::~DenoiserDocumentController() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        for (auto& entry : sources_)
            entry.second->cancel = true;
    }
    wake_.notify_all();
    worker_.join();
    for (auto& entry : sources_) {
        std::lock_guard<std::mutex> lock(entry.second->readerMutex);
        entry.second->reader.reset();
    }
}

// ---- Settings ----

void DenoiserDocumentController::loadModelLocked(const juce::String& path) {
    std::shared_ptr<const denoiser::Model> model;
    juce::String message;
    if (path.isEmpty()) {
        message = "No model chosen: the original audio plays. Choose a model in the plugin's window.";
    } else if (!juce::File::isAbsolutePath(path)) {
        message = "The model path is not absolute: " + path + ". The original audio plays.";
    } else {
        const juce::File file(path);
        juce::MemoryBlock bytes;
        if (!file.existsAsFile()) {
            message = "Model file not found: " + path + ". The original audio plays; choose the file again.";
        } else if (!file.loadFileAsData(bytes)) {
            message = "Could not read the model file " + path + ". The original audio plays.";
        } else {
            try {
                model = denoiser::Model::fromMemory(bytes.getData(), bytes.getSize());
            } catch (const std::exception& error) {
                message = "Could not load " + file.getFileName() + ": " + juce::String::fromUTF8(error.what()) +
                          ". The original audio plays.";
            }
        }
    }
    modelPath_ = path;
    model_ = std::move(model);
    modelMessage_ = message;
    for (auto& entry : sources_)
        requestLocked(*entry.second, kDenoise | kPublish);
}

void DenoiserDocumentController::setModelFile(const juce::String& path) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        loadModelLocked(path);
    }
    wake_.notify_all();
    notifyHostContentChanged();
}

void DenoiserDocumentController::setHighBand(denoiser::HighBand highBand) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (highBand_ == highBand)
            return;
        highBand_ = highBand;
        for (auto& entry : sources_)
            requestLocked(*entry.second, kPublish);
    }
    wake_.notify_all();
    notifyHostContentChanged();
}

DenoiserDocumentController::ModelStatus DenoiserDocumentController::getModelStatus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ModelStatus s;
    s.path = modelPath_;
    s.description = model_ ? juce::String::fromUTF8(model_->describe().c_str()) : juce::String();
    s.message = modelMessage_;
    s.highBand = highBand_;
    return s;
}

std::vector<DenoiserDocumentController::SourceStatus> DenoiserDocumentController::getSourceStatuses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<SourceStatus> result;
    for (const auto& entry : sources_) {
        const Source& s = *entry.second;
        SourceStatus status;
        status.name = s.name;
        status.sampleRate = s.rate;
        status.channels = s.channels;
        status.length = s.length;
        status.state = s.state;
        status.progress = s.progress;
        status.message = s.message;
        result.push_back(status);
    }
    return result;
}

juce::String DenoiserDocumentController::describe(SourceState state) {
    switch (state) {
        case SourceState::Waiting: return "waiting";
        case SourceState::Reading: return "reading";
        case SourceState::Denoising: return "denoising";
        case SourceState::Done: return "denoised";
        case SourceState::Original: return "original";
        case SourceState::Failed: return "failed";
    }
    return {};
}

void DenoiserDocumentController::notifyHostContentChanged() {
    // The stored audio of every source will change: tell the host (it may
    // cache renders and must save the document again)
    if (auto* document = getDocument())
        for (auto* source : document->getAudioSources())
            for (auto* modification : source->getAudioModifications()) {
                modification->notifyContentChanged(juce::ARAContentUpdateScopes::samplesAreAffected(), true);
                for (auto* region : modification->getPlaybackRegions())
                    region->notifyContentChanged(juce::ARAContentUpdateScopes::samplesAreAffected(), true);
            }
}

// ---- Sources ----

void DenoiserDocumentController::requestLocked(Source& source, unsigned needs) {
    source.needs |= needs;
    if (source.running && (needs & (kRead | kDenoise)) != 0)
        source.cancel = true;  // the running job is out of date
    updateWorkingLocked(source);
}

void DenoiserDocumentController::updateWorkingLocked(Source& source) {
    // A render waits only for work that will actually run: not for a source
    // the host has not released yet, nor one whose samples it can't read now
    const bool runnable = source.eligible && !source.destroyed &&
                          ((source.needs & kRead) == 0 || source.accessEnabled);
    const bool working = source.running || (source.needs != 0 && runnable);
    for (auto& slot : source.slots)
        slot.second->working = working;
}

void DenoiserDocumentController::captureProperties(Source& source) {
    juce::ARAAudioSource* ara = source.ara;
    if (ara == nullptr)
        return;
    const char* name = ara->getName();
    source.name = name != nullptr ? juce::String::fromUTF8(name) : juce::String("(unnamed source)");
    source.rate = ara->getSampleRate();
    source.channels = ara->getChannelCount();
    source.length = ara->getSampleCount();
}

void DenoiserDocumentController::didAddAudioSourceToDocument(juce::ARADocument*, juce::ARAAudioSource* ara) {
    auto source = std::make_shared<Source>();
    source->ara = ara;
    std::lock_guard<std::mutex> lock(mutex_);
    captureProperties(*source);
    sources_[ara] = source;
}

void DenoiserDocumentController::willDestroyAudioSource(juce::ARAAudioSource* ara) {
    std::shared_ptr<Source> source;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sources_.find(ara);
        if (it == sources_.end())
            return;
        source = it->second;
        sources_.erase(it);
        source->destroyed = true;
        source->cancel = true;
        updateWorkingLocked(*source);
    }
    {
        std::lock_guard<std::mutex> lock(source->araMutex);
        source->ara = nullptr;
    }
    std::lock_guard<std::mutex> lock(source->readerMutex);
    source->reader.reset();
}

void DenoiserDocumentController::didUpdateAudioSourceProperties(juce::ARAAudioSource* ara) {
    // Rate, length or channels may have changed (only while access is disabled)
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = sources_.find(ara);
    if (it == sources_.end())
        return;
    const double rate = it->second->rate;
    const int channels = it->second->channels;
    const std::int64_t length = it->second->length;
    captureProperties(*it->second);
    if (!juce::exactlyEqual(rate, it->second->rate) || channels != it->second->channels ||
        length != it->second->length)
        requestLocked(*it->second, kAll);
}

void DenoiserDocumentController::doUpdateAudioSourceContent(juce::ARAAudioSource* ara,
                                                            juce::ARAContentUpdateScopes scopes) {
    if (!scopes.affectSamples())
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = sources_.find(ara);
    if (it != sources_.end())
        requestLocked(*it->second, kAll);
}

void DenoiserDocumentController::willEnableAudioSourceSamplesAccess(juce::ARAAudioSource* ara, bool enable) {
    if (enable)
        return;
    std::shared_ptr<Source> source;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sources_.find(ara);
        if (it == sources_.end())
            return;
        source = it->second;
        source->accessEnabled = false;
        updateWorkingLocked(*source);
    }
    // Waits for a read in progress (one chunk); a running read then finds no
    // reader, stops and is done again once access is back
    std::lock_guard<std::mutex> lock(source->readerMutex);
    source->reader.reset();
}

void DenoiserDocumentController::didEnableAudioSourceSamplesAccess(juce::ARAAudioSource* ara, bool enable) {
    if (!enable)
        return;
    std::shared_ptr<Source> source;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sources_.find(ara);
        if (it == sources_.end())
            return;
        source = it->second;
    }
    {
        std::lock_guard<std::mutex> lock(source->readerMutex);
        source->reader = std::make_unique<ARA::PlugIn::HostAudioReader>(ara, false);
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        source->accessEnabled = true;
        updateWorkingLocked(*source);
    }
    wake_.notify_all();
}

void DenoiserDocumentController::willBeginEditing(juce::ARADocument*) {
    std::lock_guard<std::mutex> lock(mutex_);
    editing_ = true;
}

void DenoiserDocumentController::didEndEditing(juce::ARADocument*) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        editing_ = false;
        for (auto& entry : sources_) {
            captureProperties(*entry.second);
            entry.second->eligible = true;
            updateWorkingLocked(*entry.second);
        }
    }
    wake_.notify_all();
    rebuildPlans();
}

// ---- Renderers ----

juce::ARAPlaybackRenderer* DenoiserDocumentController::doCreatePlaybackRenderer() noexcept {
    return new DenoiserPlaybackRenderer(getDocumentController(), *this);
}

void DenoiserDocumentController::addRenderer(DenoiserPlaybackRenderer* renderer) {
    std::lock_guard<std::mutex> lock(renderersMutex_);
    renderers_.push_back(renderer);
}

void DenoiserDocumentController::removeRenderer(DenoiserPlaybackRenderer* renderer) {
    std::lock_guard<std::mutex> lock(renderersMutex_);
    renderers_.erase(std::remove(renderers_.begin(), renderers_.end(), renderer), renderers_.end());
}

void DenoiserDocumentController::rebuildPlans() {
    std::lock_guard<std::mutex> lock(renderersMutex_);
    for (auto* renderer : renderers_)
        renderer->rebuildPlan();
}

std::shared_ptr<AudioSlot> DenoiserDocumentController::slotFor(juce::ARAAudioSource* ara, double rate) {
    bool created = false;
    std::shared_ptr<AudioSlot> slot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sources_.find(ara);
        if (it == sources_.end())
            return nullptr;
        Source& source = *it->second;
        auto& entry = source.slots[rateKey(rate)];
        if (!entry) {
            entry = std::make_shared<AudioSlot>();
            created = true;
            // A version at this rate: made from what the analysis has (or will have)
            requestLocked(source, kPublish);
        }
        slot = entry;
    }
    if (created)
        wake_.notify_all();
    return slot;
}

// ---- Archive: the document settings (model path, high band); the audio is not stored ----

bool DenoiserDocumentController::doStoreObjectsToStream(juce::ARAOutputStream& output,
                                                        const juce::ARAStoreObjectsFilter* filter) noexcept {
    if (filter != nullptr && !filter->shouldStoreDocumentData())
        return true;
    const ModelStatus s = getModelStatus();
    return output.writeInt(kArchiveMagic) && output.writeInt(kArchiveVersion) && output.writeString(s.path) &&
           output.writeInt(s.highBand == denoiser::HighBand::PassThrough ? 1 : 0);
}

bool DenoiserDocumentController::doRestoreObjectsFromStream(juce::ARAInputStream& input,
                                                            const juce::ARARestoreObjectsFilter* filter) noexcept {
    if (filter != nullptr && !filter->shouldRestoreDocumentData())
        return true;
    if (input.getTotalLength() == 0)
        return true;  // nothing stored (e.g. an archive of the plugin before it stored anything)
    if (input.readInt() != kArchiveMagic)
        return false;
    const int version = input.readInt();
    if (version < 1 || version > kArchiveVersion)
        return false;  // written by a newer plugin
    const juce::String path = input.readString();
    const int highBand = input.readInt();
    if (input.failed())
        return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        highBand_ = highBand == 1 ? denoiser::HighBand::PassThrough : denoiser::HighBand::Drop;
        loadModelLocked(path);  // analysis starts when the host ends the edit
    }
    wake_.notify_all();
    return true;
}

// ============================================================================
// Analysis thread

void DenoiserDocumentController::setStatus(Source& source, SourceState state, float progress,
                                           const juce::String& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    source.state = state;
    source.progress = progress;
    source.message = message;
}

void DenoiserDocumentController::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
        std::shared_ptr<Source> next;
        for (auto& entry : sources_) {
            Source& s = *entry.second;
            if (s.needs != 0 && s.eligible && !s.destroyed && ((s.needs & kRead) == 0 || s.accessEnabled)) {
                next = entry.second;
                break;
            }
        }
        if (!next) {
            wake_.wait(lock);
            continue;
        }
        const unsigned job = next->needs;
        next->needs = 0;
        next->running = true;
        next->cancel = false;
        updateWorkingLocked(*next);
        lock.unlock();
        runJob(*next, job);
        lock.lock();
        next->running = false;
        updateWorkingLocked(*next);
    }
}

DenoiserDocumentController::ReadResult DenoiserDocumentController::readSource(Source& source, Planar& out) {
    double rate;
    int channels;
    std::int64_t length;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        rate = source.rate;
        channels = source.channels;
        length = source.length;
    }
    juce::ignoreUnused(rate);
    const int kept = std::min(channels, enm156::ara::kMaxSourceChannels);
    if (channels < 1 || length <= 0) {
        out.assign(size_t(std::max(kept, 1)), {});
        return ReadResult::Ok;
    }
    if (std::int64_t(kept) * length > enm156::ara::kMaxSourceSamples) {
        setStatus(source, SourceState::Failed, 0.0f,
                  "Too long to analyse (" + juce::String(double(length) / std::max(rate, 1.0) / 60.0, 1) +
                      " min; the limit is " + juce::String(enm156::ara::kMaxSourceSamples / kept) +
                      " samples per channel): this source plays silence. Split it into shorter clips.");
        return ReadResult::Failed;
    }
    out.assign(size_t(kept), std::vector<float>(size_t(length)));
    std::vector<std::vector<float>> chunk(static_cast<size_t>(channels),
                                          std::vector<float>(static_cast<size_t>(kReadChunk)));
    std::vector<void*> pointers(static_cast<size_t>(channels));
    for (int c = 0; c < channels; ++c)
        pointers[size_t(c)] = chunk[size_t(c)].data();
    for (std::int64_t pos = 0; pos < length; pos += kReadChunk) {
        if (source.cancel.load() || stop_)
            return ReadResult::Cancelled;
        const std::int64_t n = std::min(kReadChunk, length - pos);
        {
            std::lock_guard<std::mutex> lock(source.readerMutex);
            if (!source.reader)
                return ReadResult::AccessLost;  // read again once access is enabled
            if (!source.reader->readAudioSamples(pos, n, pointers.data())) {
                setStatus(source, SourceState::Failed, 0.0f,
                          "The host could not read this source's audio: it plays silence.");
                return ReadResult::Failed;
            }
        }
        for (int c = 0; c < kept; ++c)
            std::copy(chunk[size_t(c)].begin(), chunk[size_t(c)].begin() + n, out[size_t(c)].begin() + pos);
        setStatus(source, SourceState::Reading, float(double(pos + n) / double(length)), {});
    }
    return ReadResult::Ok;
}

void DenoiserDocumentController::publishAll(Source& source, denoiser::HighBand highBand) {
    std::vector<std::pair<int, std::shared_ptr<AudioSlot>>> slots;
    double rate;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        slots.assign(source.slots.begin(), source.slots.end());
        rate = source.rate;
    }
    for (auto& entry : slots) {
        if (entry.first == rateKey(rate))
            entry.second->publish(source.analysis.atSourceRate(highBand));
        else
            entry.second->publish(source.analysis.atRate(entry.first));
    }
}

void DenoiserDocumentController::runJob(Source& source, unsigned job) {
    std::shared_ptr<const denoiser::Model> model;
    denoiser::HighBand highBand;
    double rate;
    int channels;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        model = model_;
        highBand = highBand_;
        rate = source.rate;
        channels = source.channels;
    }
    const auto notify = [&source](auto&& call) {
        std::lock_guard<std::mutex> lock(source.araMutex);
        if (source.ara != nullptr)
            call(*source.ara);
    };
    // Puts back what was not done, if the job stopped early
    const auto requeue = [this, &source](unsigned bits) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!source.destroyed)
            source.needs |= bits;
    };
    notify([](juce::ARAAudioSource& s) { s.notifyAnalysisProgressStarted(); });
    const auto complete = [&] { notify([](juce::ARAAudioSource& s) { s.notifyAnalysisProgressCompleted(); }); };

    try {
        if ((job & kRead) != 0) {
            setStatus(source, SourceState::Reading, 0.0f, {});
            Planar input;
            const ReadResult read = readSource(source, input);
            if (read != ReadResult::Ok) {
                if (read == ReadResult::Failed) {
                    source.analysis = enm156::ara::SourceAnalysis();  // nothing to play
                    publishAll(source, highBand);
                } else {
                    requeue(job);
                    setStatus(source, SourceState::Waiting, 0.0f, {});
                }
                complete();
                return;
            }
            source.analysis.setInput(rate, std::move(input));
            // The original plays until the denoised audio is ready
            publishAll(source, highBand);
            job |= kDenoise | kPublish;
        }
        if (!source.analysis.hasInput()) {
            complete();
            return;  // not read yet: the read comes with the next job
        }
        if ((job & kDenoise) != 0) {
            if (model)
                setStatus(source, SourceState::Denoising, 0.0f, {});
            const juce::File hold = testHoldFile();
            while (hold != juce::File() && hold.exists() && !source.cancel.load() && !stop_)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            const Outcome outcome = source.analysis.denoise(model, [&](float fraction) {
                if (source.cancel.load() || stop_)
                    return false;
                if (fraction >= 0.0f) {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        source.progress = fraction;
                    }
                    notify([fraction](juce::ARAAudioSource& s) { s.notifyAnalysisProgressUpdated(fraction); });
                }
                return true;
            });
            if (outcome == Outcome::Cancelled) {
                requeue(kDenoise | kPublish);
                complete();
                return;
            }
        }
        publishAll(source, highBand);
        juce::String message = juce::String::fromUTF8(source.analysis.message().c_str());
        if (!source.analysis.denoised() && message.isEmpty())
            message = "The original audio plays.";
        if (channels > enm156::ara::kMaxSourceChannels)
            message = (message.isEmpty() ? juce::String() : message + " ") + "Only the first " +
                      juce::String(enm156::ara::kMaxSourceChannels) + " of " + juce::String(channels) +
                      " channels are played.";
        setStatus(source, source.analysis.denoised() ? SourceState::Done : SourceState::Original, 1.0f, message);
    } catch (const std::bad_alloc&) {
        source.analysis = enm156::ara::SourceAnalysis();
        publishAll(source, highBand);
        setStatus(source, SourceState::Failed, 0.0f, "Out of memory: this source plays silence.");
    } catch (const std::exception& e) {
        setStatus(source, SourceState::Failed, 0.0f, "Analysis failed: " + juce::String::fromUTF8(e.what()));
    }
    complete();
}

// ============================================================================
// Playback renderer

DenoiserPlaybackRenderer::DenoiserPlaybackRenderer(ARA::PlugIn::DocumentController* dc,
                                                   DenoiserDocumentController& owner)
    : juce::ARAPlaybackRenderer(dc), owner_(owner) {
    owner_.addRenderer(this);
}

DenoiserPlaybackRenderer::~DenoiserPlaybackRenderer() {
    owner_.removeRenderer(this);
    delete pending_.exchange(nullptr);
    collectGarbage();
    delete active_;
}

void DenoiserPlaybackRenderer::setParameters(const std::atomic<float>* mix,
                                             const std::atomic<float>* bypass) noexcept {
    mixParam_ = mix;
    bypassParam_ = bypass;
}

void DenoiserPlaybackRenderer::prepareToPlay(double sampleRate, int maximumSamplesPerBlock, int numChannels,
                                             juce::AudioProcessor::ProcessingPrecision precision,
                                             AlwaysNonRealtime alwaysNonRealtime) {
    juce::ARAPlaybackRenderer::prepareToPlay(sampleRate, maximumSamplesPerBlock, numChannels, precision,
                                             alwaysNonRealtime);
    {
        std::lock_guard<std::mutex> lock(planMutex_);
        sampleRate_ = sampleRate;
        prepared_ = true;
    }
    mix_.reset(sampleRate > 0.0 ? sampleRate : 48000.0, 0.02);
    reset();
    rebuildPlan();
    takePending();  // the audio thread is not running yet
    collectGarbage();
}

void DenoiserPlaybackRenderer::releaseResources() {
    {
        std::lock_guard<std::mutex> lock(planMutex_);
        prepared_ = false;
    }
    delete pending_.exchange(nullptr);
    delete active_;
    active_ = nullptr;
    collectGarbage();
    juce::ARAPlaybackRenderer::releaseResources();
}

void DenoiserPlaybackRenderer::reset() {
    float target = 1.0f;
    if (mixParam_ != nullptr)
        target = juce::jlimit(0.0f, 1.0f, mixParam_->load());
    if ((bypassParam_ != nullptr && bypassParam_->load() >= 0.5f) || forceBypass_)
        target = 0.0f;
    mix_.setCurrentAndTargetValue(target);
}

void DenoiserPlaybackRenderer::collectGarbage() {
    for (auto& slot : retired_)
        delete slot.exchange(nullptr);
}

void DenoiserPlaybackRenderer::rebuildPlan() {
    std::lock_guard<std::mutex> lock(planMutex_);
    if (!prepared_)
        return;
    auto plan = std::make_unique<RenderPlan>();
    for (auto* region : getPlaybackRegions()) {
        auto* source = region->getAudioModification()->getAudioSource();
        auto slot = owner_.slotFor(source, sampleRate_);
        if (!slot)
            continue;
        // No time-stretching (none is declared): the region plays its
        // modification range at 1:1 from its start on the timeline
        const auto range =
            region->getSampleRange(sampleRate_, juce::ARAPlaybackRegion::IncludeHeadAndTail::no);
        const std::int64_t sourceStart = std::llround(region->getStartInAudioModificationTime() * sampleRate_);
        plan->regions.push_back({range.getStart(), range.getEnd(), sourceStart - range.getStart(), slot.get()});
        plan->keepAlive.push_back(std::move(slot));
    }
    delete pending_.exchange(plan.release());  // one the audio thread hasn't taken was never used
    collectGarbage();
}

void DenoiserPlaybackRenderer::takePending() noexcept {
    if (pending_.load() == nullptr)
        return;
    // A free slot for the plan being replaced (only this thread fills slots)
    std::atomic<RenderPlan*>* free = nullptr;
    if (active_ != nullptr) {
        for (auto& slot : retired_)
            if (slot.load() == nullptr) {
                free = &slot;
                break;
            }
        if (free == nullptr)
            return;  // try again next block
    }
    RenderPlan* next = pending_.exchange(nullptr);
    if (next == nullptr)
        return;
    if (active_ != nullptr)
        free->store(active_);
    active_ = next;
}

bool DenoiserPlaybackRenderer::processBlock(juce::AudioBuffer<float>& buffer, juce::AudioProcessor::Realtime realtime,
                                            const juce::AudioPlayHead::PositionInfo& positionInfo) noexcept {
    takePending();
    const int n = buffer.getNumSamples();
    const int channels = std::min(buffer.getNumChannels(), enm156::ara::kMaxSourceChannels);
    for (int c = channels; c < buffer.getNumChannels(); ++c)
        buffer.clear(c, 0, n);

    float target = mixParam_ != nullptr ? juce::jlimit(0.0f, 1.0f, mixParam_->load()) : 1.0f;
    if (forceBypass_ || (bypassParam_ != nullptr && bypassParam_->load() >= 0.5f))
        target = 0.0f;

    if (active_ == nullptr || !positionInfo.getIsPlaying() || channels < 1) {
        buffer.clear();
        mix_.setCurrentAndTargetValue(target);
        return true;
    }
    mix_.setTargetValue(target);
    const std::int64_t position = positionInfo.getTimeInSamples().orFallback(0);

    if (realtime == juce::AudioProcessor::Realtime::no) {
        // Offline (bounce / render): wait for analysis still running on these regions
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::duration<double>(kMaxOfflineWaitSeconds);
        while (enm156::ara::planIsWorking(*active_, position, n) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    float* const* data = buffer.getArrayOfWritePointers();
    constexpr int chunk = int(std::tuple_size<decltype(gain_)>::value);
    for (int offset = 0; offset < n; offset += chunk) {
        const int count = std::min(chunk, n - offset);
        for (int i = 0; i < count; ++i)
            gain_[size_t(i)] = mix_.getNextValue();
        float* out[enm156::ara::kMaxSourceChannels] = {};
        for (int c = 0; c < channels; ++c)
            out[c] = data[c] + offset;
        enm156::ara::renderPlan(*active_, position + offset, count, out, channels, gain_.data());
    }
    return true;
}

// ============================================================================

const ARA::ARAFactory* JUCE_CALLTYPE createARAFactory() {
    return juce::ARADocumentControllerSpecialisation::createARAFactory<DenoiserDocumentController>();
}

#endif
