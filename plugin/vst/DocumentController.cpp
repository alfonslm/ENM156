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
using enm156::ara::LoadResult;
using enm156::ara::Outcome;
using enm156::ara::Planar;
using enm156::ara::ReadStatus;
using enm156::ara::RenderPlan;
using enm156::ara::SourceAudio;
using enm156::ara::SourceStream;

namespace {

// The archive: a small binary record of the document settings
constexpr int kArchiveMagic = 0x454e4d44;  // "ENMD"
constexpr int kArchiveVersion = 1;

int rateKey(double rate) {
    return int(std::lround(rate));
}

int keptChannels(int channels) {
    return std::clamp(channels, 1, enm156::ara::kMaxSourceChannels);
}

// The streaming thread looks at the play cursors this often when it has
// nothing to read (an audio thread cannot wake it)
constexpr auto kStreamPoll = std::chrono::milliseconds(2);

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
    bool tooLong = false;   // more than kMaxSourceSamples: never read whole
    bool eligible = false;  // the host ended the edit that created it
    bool accessEnabled = false;
    bool destroyed = false;
    bool running = false;   // the analysis thread works on it
    unsigned needs = kAll;
    std::map<int, std::shared_ptr<AudioSlot>> slots;  // by playback rate
    std::shared_ptr<const enm156::ara::SourceAudio> mainAudio;  // the latest at the source's rate
    SourceState state = SourceState::Waiting;
    float progress = 0.0f;
    juce::String message;

    // Streaming the original (mutex_): why, and the SourceAudio holding the
    // stream for each playback rate (null: could not be made there)
    Streaming streaming = Streaming::No;
    std::uint64_t content = 0;         // counts changes of the source's samples
    std::uint64_t streamsContent = 0;  // the content `streams` were made for
    std::map<int, std::shared_ptr<const SourceAudio>> streams;
    bool streamsUpToDate = true;       // every slot has its stream published
    // Taken (before mutex_) by the analysis and the streaming thread while
    // they publish into the slots, so that neither overwrites the other's
    // newer decision (held in memory, or streamed)
    std::mutex publishMutex;

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
    streamer_ = std::thread([this] { streamRun(); });
}

DenoiserDocumentController::~DenoiserDocumentController() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        for (auto& entry : sources_)
            entry.second->cancel = true;
    }
    wake_.notify_all();
    streamWake_.notify_all();
    // Each returns once a host read it has in progress returns
    worker_.join();
    streamer_.join();
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
        case SourceState::Streamed: return "original (streamed)";
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
    if ((needs & kRead) != 0) {
        ++source.content;  // streamed blocks read before are out of date:
        for (auto& stream : source.streams)  // silent from now on, until replaced
            if (stream.second && stream.second->stream)
                stream.second->stream->invalidate();
        if (source.streaming != Streaming::No) {
            source.streamsUpToDate = false;
            streamWake_.notify_all();
        }
    }
    updateWorkingLocked(source);
}

void DenoiserDocumentController::updateWorkingLocked(Source& source) {
    // A render waits only for work that will actually run: not for a source
    // the host has not released yet, nor one whose samples it can't read now.
    // A too-long source is never analysed; setting up its streams counts.
    const bool alive = source.eligible && !source.destroyed;
    const bool runnable = alive && !source.tooLong && ((source.needs & kRead) == 0 || source.accessEnabled);
    const bool streamSetUp = alive && source.streaming != Streaming::No && !source.streamsUpToDate;
    const bool working = source.running || (source.needs != 0 && runnable) || streamSetUp;
    for (auto& slot : source.slots)
        slot.second->working = working;
}

void DenoiserDocumentController::startStreamingLocked(Source& source, Streaming why, const juce::String& message) {
    source.streaming = why;
    source.streamsUpToDate = false;
    source.mainAudio = nullptr;  // a slot made from now on gets the stream
    source.state = SourceState::Streamed;
    source.progress = 0.0f;
    source.message = message;
    updateWorkingLocked(source);
    streamWake_.notify_all();
}

void DenoiserDocumentController::startStreaming(Source& source, Streaming why, const juce::String& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!source.destroyed)
        startStreamingLocked(source, why, message);
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
    // A source too long to hold in memory plays its original, streamed
    const int kept = keptChannels(source.channels);
    if (std::int64_t(kept) * source.length > enm156::ara::kMaxSourceSamples) {
        juce::String message = "Too long to analyse (" +
                               juce::String(double(source.length) / std::max(source.rate, 1.0) / 60.0, 1) +
                               " min; the limit is " + juce::String(enm156::ara::kMaxSourceSamples / kept) +
                               " samples per channel): the original plays, read from the host as it plays, "
                               "not denoised. Split it into shorter clips to denoise it.";
        if (source.channels > enm156::ara::kMaxSourceChannels)
            message << " Only the first " << enm156::ara::kMaxSourceChannels << " of " << source.channels
                    << " channels are played.";
        source.tooLong = true;
        if (source.streaming != Streaming::TooLong || source.message != message)
            startStreamingLocked(source, Streaming::TooLong, message);
    } else if (source.tooLong) {
        // No longer too long (its properties changed, which also asks for a
        // new read): analysed as usual
        source.tooLong = false;
        source.streaming = Streaming::No;
        source.streams.clear();
        source.streamsUpToDate = true;
        source.state = SourceState::Waiting;
        source.message = {};
        updateWorkingLocked(source);
    }
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
    streamWake_.notify_all();
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
            // At the source's rate: what the analysis has now (no reader can
            // hold the new slot yet, so this doesn't wait)
            if (rateKey(rate) == rateKey(source.rate) && source.mainAudio)
                entry->publish(source.mainAudio);
            // A streamed source: the streaming thread makes a stream at this rate
            if (source.streaming != Streaming::No) {
                source.streamsUpToDate = false;
                streamWake_.notify_all();
            }
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
            if (s.needs != 0 && s.eligible && !s.destroyed && !s.tooLong &&
                ((s.needs & kRead) == 0 || s.accessEnabled)) {
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

ReadStatus DenoiserDocumentController::readFromHost(Source& source, int channels, std::int64_t position,
                                                   std::int64_t n, float* const* out, Planar& scratch) {
    // The host fills every channel of the source; those past the kept ones
    // go to scratch
    const int kept = keptChannels(channels);
    const auto total = size_t(std::max(channels, kept));
    if (scratch.size() < total - size_t(kept))
        scratch.resize(total - size_t(kept));
    std::vector<void*> pointers(total);
    for (size_t c = 0; c < total; ++c) {
        if (c < size_t(kept)) {
            pointers[c] = out[c];
        } else {
            auto& extra = scratch[c - size_t(kept)];
            if (std::int64_t(extra.size()) < n)
                extra.resize(size_t(n));
            pointers[c] = extra.data();
        }
    }
    std::lock_guard<std::mutex> lock(source.readerMutex);
    if (!source.reader)
        return ReadStatus::NoAccess;  // read again once access is enabled
    return source.reader->readAudioSamples(position, n, pointers.data()) ? ReadStatus::Ok : ReadStatus::Failed;
}

LoadResult DenoiserDocumentController::readSource(Source& source, Planar& out) {
    int channels;
    std::int64_t length;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        channels = source.channels;
        length = source.length;
    }
    Planar scratch;
    return enm156::ara::readWholeSource(
        [&](std::int64_t position, std::int64_t n, float* const* into) {
            return readFromHost(source, channels, position, n, into, scratch);
        },
        channels, length, out, [&](float fraction) {
            if (source.cancel.load() || stop_)
                return false;
            setStatus(source, SourceState::Reading, fraction, {});
            return true;
        });
}

void DenoiserDocumentController::publishAll(Source& source, denoiser::HighBand highBand) {
    // The version at the source's own rate is kept, so that a slot made later
    // (a renderer prepared while the analysis is busy) starts with it at once
    const auto main = source.analysis.atSourceRate(highBand);
    std::vector<std::pair<int, std::shared_ptr<AudioSlot>>> slots;
    double rate;
    const std::lock_guard<std::mutex> publishing(source.publishMutex);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (source.tooLong || source.destroyed)
            return;
        // Held in memory from now on: no longer streamed
        source.streaming = Streaming::No;
        source.streams.clear();
        source.streamsUpToDate = true;
        updateWorkingLocked(source);
        source.mainAudio = main;
        slots.assign(source.slots.begin(), source.slots.end());
        rate = source.rate;
    }
    for (auto& entry : slots) {
        if (entry.first == rateKey(rate))
            entry.second->publish(main);
        else
            entry.second->publish(source.analysis.atRate(entry.first));
    }
}

// ============================================================================
// Streaming thread

void DenoiserDocumentController::setUpStreams(Source& source) {
    // What is needed: a stream for each slot (playback rate)
    std::vector<std::pair<int, std::shared_ptr<AudioSlot>>> slots;
    std::map<int, std::shared_ptr<const SourceAudio>> have;
    double rate;
    int channels;
    std::int64_t length;
    std::uint64_t content;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (source.streaming == Streaming::No || source.destroyed) {
            source.streamsUpToDate = true;
            updateWorkingLocked(source);
            return;
        }
        if (source.streamsContent != source.content) {
            source.streams.clear();  // read before the samples changed
            source.streamsContent = source.content;
        }
        slots.assign(source.slots.begin(), source.slots.end());
        have = source.streams;
        rate = source.rate;
        channels = keptChannels(source.channels);
        length = source.length;
        content = source.content;
    }
    bool outOfMemory = false;
    for (const auto& slot : slots) {
        if (have.count(slot.first) != 0)
            continue;
        std::shared_ptr<SourceAudio> audio;
        try {
            auto stream = std::make_shared<SourceStream>(rate, channels, length, slot.first);
            audio = std::make_shared<SourceAudio>();
            audio->sampleRate = slot.first;
            audio->channels = stream->channels();
            audio->length = stream->length();
            audio->stream = std::move(stream);
        } catch (const std::bad_alloc&) {
            outOfMemory = true;  // silence at this rate
        } catch (const std::invalid_argument&) {
            // A rate that can't be resampled (as for analysed sources): silence there
        }
        have[slot.first] = std::move(audio);
    }
    const std::lock_guard<std::mutex> publishing(source.publishMutex);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (source.streaming == Streaming::No || source.destroyed || source.content != content)
            return;  // decided otherwise meanwhile (the loop comes back if still needed)
        source.streams = have;
        if (outOfMemory)
            source.message = "Out of memory: this source plays silence.";
    }
    for (const auto& slot : slots) {
        const auto& audio = have[slot.first];
        if (slot.second->current() != audio)
            slot.second->publish(audio);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    bool complete = source.content == content && source.streaming != Streaming::No;
    for (const auto& slot : source.slots)
        complete = complete && have.count(slot.first) != 0;
    if (complete) {
        source.streamsUpToDate = true;
        updateWorkingLocked(source);
    }
}

void DenoiserDocumentController::streamRun() {
    struct Work {
        std::shared_ptr<Source> source;
        std::shared_ptr<SourceStream> stream;
        std::shared_ptr<AudioSlot> slot;
        int channels = 0;
    };
    std::vector<Work> work;
    std::vector<enm156::ara::CursorState> cursors;
    Planar scratch;
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
        // Streams to make or replace first (cheap, no host read)
        std::shared_ptr<Source> setUp;
        for (auto& entry : sources_) {
            const Source& s = *entry.second;
            if (s.streaming != Streaming::No && !s.streamsUpToDate && s.eligible && !s.destroyed) {
                setUp = entry.second;
                break;
            }
        }
        if (setUp) {
            lock.unlock();
            setUpStreams(*setUp);
            setUp.reset();
            lock.lock();
            continue;
        }
        for (auto& entry : sources_) {
            Source& s = *entry.second;
            if (s.streaming == Streaming::No || !s.eligible || s.destroyed)
                continue;
            for (auto& stream : s.streams) {
                const auto slot = s.slots.find(stream.first);
                if (stream.second && stream.second->stream && slot != s.slots.end())
                    work.push_back({entry.second, stream.second->stream, slot->second, s.channels});
            }
        }
        if (work.empty()) {
            streamWake_.wait(lock);
            continue;
        }
        lock.unlock();
        // One block per stream per round, so that every stream keeps up
        bool loaded = false;
        for (const Work& w : work) {
            if (stop_)
                break;
            w.slot->cursorStates(cursors);
            const auto read = [&](std::int64_t position, std::int64_t n, float* const* out) {
                return readFromHost(*w.source, w.channels, position, n, out, scratch);
            };
            loaded = w.stream->fillOne(cursors, read) == SourceStream::Fill::Loaded || loaded;
        }
        work.clear();
        lock.lock();
        if (!loaded && !stop_)
            streamWake_.wait_for(lock, kStreamPoll);
    }
}

void DenoiserDocumentController::runJob(Source& source, unsigned job) {
    std::shared_ptr<const denoiser::Model> model;
    denoiser::HighBand highBand;
    double rate;
    int channels;
    Streaming streaming;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        model = model_;
        highBand = highBand_;
        rate = source.rate;
        channels = source.channels;
        streaming = source.streaming;
    }
    // A source streamed because reading it whole failed (or memory ran out)
    // is tried again when the model is chosen again
    if (streaming != Streaming::No && (job & kDenoise) != 0 && !source.analysis.hasInput())
        job |= kRead;
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
            const LoadResult read = readSource(source, input);
            if (read != LoadResult::Ok) {
                Planar().swap(input);
                switch (read) {
                    case LoadResult::Failed:
                        source.analysis = enm156::ara::SourceAnalysis();
                        startStreaming(source, Streaming::ReadFailed,
                                       "The host could not read this source's audio: the original plays as the "
                                       "host delivers it, not denoised. It is read again when the source changes "
                                       "or a model is chosen.");
                        break;
                    case LoadResult::OutOfMemory:
                    case LoadResult::TooLong:
                        source.analysis = enm156::ara::SourceAnalysis();
                        startStreaming(source, Streaming::OutOfMemory,
                                       "Out of memory: the original plays, read from the host as it plays, "
                                       "not denoised.");
                        break;
                    case LoadResult::Ok:
                    case LoadResult::Cancelled:
                    case LoadResult::NoAccess: {
                        requeue(job);
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (source.streaming == Streaming::No) {
                            source.state = SourceState::Waiting;
                            source.progress = 0.0f;
                            source.message = {};
                        }
                        break;
                    }
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
        // Nothing of it is kept in memory: its original is streamed instead
        source.analysis = enm156::ara::SourceAnalysis();
        startStreaming(source, Streaming::OutOfMemory,
                       "Out of memory: the original plays, read from the host as it plays, not denoised.");
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
        // Where this region wants the source read ahead, if it is streamed
        auto cursor = slot->addCursor();
        plan->regions.push_back(
            {range.getStart(), range.getEnd(), sourceStart - range.getStart(), slot.get(), cursor.get()});
        plan->keepAlive.push_back(std::move(slot));
        plan->cursors.push_back(std::move(cursor));
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

    const std::int64_t position = positionInfo.getTimeInSamples().orFallback(0);
    // Streamed sources are read ahead of this position, also while stopped
    // (so that playback started here finds its audio)
    if (active_ != nullptr)
        enm156::ara::notePlayPosition(*active_, position);

    if (active_ == nullptr || !positionInfo.getIsPlaying() || channels < 1) {
        buffer.clear();
        mix_.setCurrentAndTargetValue(target);
        return true;
    }
    mix_.setTargetValue(target);

    if (realtime == juce::AudioProcessor::Realtime::no) {
        // Offline (bounce / render): wait for analysis still running on
        // these regions, then for streamed audio still being read (not while
        // the host's reads fail)
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::duration<double>(kMaxOfflineWaitSeconds);
        while (enm156::ara::planIsWorking(*active_, position, n) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        while (enm156::ara::planStreamState(*active_, position, n) == enm156::ara::StreamState::Pending &&
               std::chrono::steady_clock::now() < deadline) {
            owner_.wakeStreaming();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
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
