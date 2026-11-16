// A headless ARA host: loads the built VST3, creates an ARA document through
// the real ARA interface (JUCE's ARA hosting classes on top of the ARA SDK's
// host dispatch library), binds plugin instances to it as playback
// renderers, and checks the plugin's ARA side against the core library.
// Run by ctest as `plugin_ara`.
//
//   denoiser_ara_host_test "<path to>/ENM156 Denoiser.vst3" <golden dir>
//
// The host side plays the role of a DAW: audio sources are in-memory clips
// (golden inputs and synthetic ones), read by the plugin through the host's
// audio reader; the document's settings (model path, high band) reach the
// plugin through its ARA archive, which is also how a DAW restores a
// project. Renders are offline (non-realtime) unless the test is about
// realtime behaviour: an offline render waits for analysis still running,
// as a DAW's bounce would.
//
// Comparisons with the core use a tolerance (1e-5 of max(1, |ref|)), as in
// HostTest.cpp, because the plugin is compiled separately; exact equality
// only where the plugin copies samples (the original audio, mix 0, bypass).
#include "denoiser/denoiser.h"
#include "denoiser/offline.h"
#include "denoiser/offline_rate.h"
#include "denoiser/tensor_file.h"

#include <juce_audio_processors_headless/juce_audio_processors_headless.h>
#include <juce_events/juce_events.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>


#define DENOISER_COUNT_WRAPPER_ALLOCATIONS 0
#if defined(__linux__) && defined(__GLIBC__) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
#if defined(__has_feature)
#if !__has_feature(address_sanitizer) && !__has_feature(thread_sanitizer) && !__has_feature(memory_sanitizer)
#undef DENOISER_COUNT_WRAPPER_ALLOCATIONS
#define DENOISER_COUNT_WRAPPER_ALLOCATIONS 1
#endif
#else
#undef DENOISER_COUNT_WRAPPER_ALLOCATIONS
#define DENOISER_COUNT_WRAPPER_ALLOCATIONS 1
#endif
#endif
#if DENOISER_COUNT_WRAPPER_ALLOCATIONS
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#endif
#include "../tests/alloc_hooks.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

namespace {

using Audio = std::vector<std::vector<float>>;  // planar
namespace HM = juce::ARAHostModel;

int failures = 0;
juce::String goldenDir;
const ARA::ARAFactory* araFactory = nullptr;

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

constexpr double kTolerance = 1e-5;

double relErr(const std::vector<float>& a, std::size_t aStart, const std::vector<float>& b, std::size_t bStart,
              std::size_t n) {
    double diff = 0.0, peak = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = a[aStart + i], y = b[bStart + i];
        diff = std::max(diff, std::fabs(x - y));
        peak = std::max(peak, std::fabs(y));
    }
    return diff / peak;
}

bool sameSamples(const std::vector<float>& a, std::size_t aStart, const std::vector<float>& b, std::size_t bStart,
                 std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        if (!juce::exactlyEqual(a[aStart + i], b[bStart + i]))
            return false;
    return true;
}

bool isSilent(const std::vector<float>& a, std::size_t start, std::size_t n) {
    for (std::size_t i = start; i < start + n; ++i)
        if (!juce::exactlyEqual(a[i], 0.0f))
            return false;
    return true;
}

Audio noise(int channels, std::size_t n, int seed) {
    juce::Random random(seed);
    Audio a(static_cast<size_t>(channels), std::vector<float>(n));
    for (auto& ch : a)
        for (size_t i = 0; i < ch.size(); ++i)
            ch[i] = 0.3f * std::sin(0.031f * float(i) * float(1 + seed % 3)) + 0.2f * (random.nextFloat() * 2.0f - 1.0f);
    return a;
}

std::vector<float> goldenInput(const juce::String& name) {
    const denoiser::TensorFile vectors =
        denoiser::TensorFile::load(golden(name + ".vectors.bin").toStdString());
    const denoiser::Tensor& input = vectors.get("input");
    return std::vector<float>(input.data, input.data + input.size());
}

std::shared_ptr<const denoiser::Model> loadModel(const juce::String& path) {
    juce::MemoryBlock bytes;
    if (!juce::File(path).loadFileAsData(bytes)) {
        fail("cannot read " + path);
        return nullptr;
    }
    return denoiser::Model::fromMemory(bytes.getData(), bytes.getSize());
}

// The core's offline result for one channel at the source's rate
std::vector<float> reference(const denoiser::Model& model, int rate, const std::vector<float>& x,
                             denoiser::HighBand highBand = denoiser::HighBand::Drop) {
    auto net = model.makeNetwork();
    return denoiser::denoiseClipAtRate(model.info(), *net, rate, x.data(), x.size(), highBand);
}

// The document settings as the plugin stores them in its ARA archive
// (DocumentController.cpp): magic, version, model path, high band
juce::MemoryBlock settingsArchive(const juce::String& modelPath, bool highBandPass = false) {
    juce::MemoryOutputStream out;
    out.writeInt(0x454e4d44);
    out.writeInt(1);
    out.writeString(modelPath);
    out.writeInt(highBandPass ? 1 : 0);
    return out.getMemoryBlock();
}

// ============================================================================
// The host side of ARA

struct TestSource {
    Audio data;
    double rate = 48000.0;
    juce::String id;
    std::int64_t length() const { return data.empty() ? 0 : std::int64_t(data[0].size()); }
};

struct Reader {
    TestSource* source;
};

using SourceConverter = HM::ConversionFunctions<TestSource*, ARA::ARAAudioSourceHostRef>;
using ReaderConverter = HM::ConversionFunctions<Reader*, ARA::ARAAudioReaderHostRef>;
using ArchiveInConverter = HM::ConversionFunctions<juce::MemoryBlock*, ARA::ARAArchiveReaderHostRef>;
using ArchiveOutConverter = HM::ConversionFunctions<juce::MemoryOutputStream*, ARA::ARAArchiveWriterHostRef>;

class AudioAccess final : public ARA::Host::AudioAccessControllerInterface {
public:
    ARA::ARAAudioReaderHostRef createAudioReaderForSource(ARA::ARAAudioSourceHostRef source,
                                                          bool use64BitSamples) noexcept override {
        if (use64BitSamples)
            fail("the plugin asked for 64-bit samples");
        auto* reader = new Reader{SourceConverter::fromHostRef(source)};
        ++readersAlive;
        return ReaderConverter::toHostRef(reader);
    }
    bool readAudioSamples(ARA::ARAAudioReaderHostRef readerRef, ARA::ARASamplePosition position,
                          ARA::ARASampleCount count, void* const buffers[]) noexcept override {
        const TestSource& s = *ReaderConverter::fromHostRef(readerRef)->source;
        waitAtGate();
        ++reads;
        for (size_t c = 0; c < s.data.size(); ++c) {
            auto* out = static_cast<float*>(buffers[c]);
            for (ARA::ARASampleCount i = 0; i < count; ++i) {
                const auto k = position + i;
                out[i] = k >= 0 && k < s.length() ? s.data[c][size_t(k)] : 0.0f;
            }
        }
        return true;
    }
    void destroyAudioReader(ARA::ARAAudioReaderHostRef readerRef) noexcept override {
        delete ReaderConverter::fromHostRef(readerRef);
        --readersAlive;
    }
    std::atomic<int> readersAlive{0};
    std::atomic<long> reads{0};

    // The host's control over its reads (a DAW's reads can be slow, e.g. from
    // disk or the network): while the gate is closed, every read waits in the
    // host until the host lets it through. Open by default.
    void closeGate() {
        const std::lock_guard<std::mutex> lock(gateMutex_);
        gateClosed_ = true;
    }
    void openGate() {
        {
            const std::lock_guard<std::mutex> lock(gateMutex_);
            gateClosed_ = false;
        }
        gate_.notify_all();
    }
    // Lets one waiting (or the next) read through, the gate staying closed
    void letOneThrough() {
        {
            const std::lock_guard<std::mutex> lock(gateMutex_);
            ++passes_;
        }
        gate_.notify_all();
    }
    // Waits until `n` reads in all have come to the closed gate (false after 20 s)
    bool waitForArrivals(int n) {
        std::unique_lock<std::mutex> lock(gateMutex_);
        return gate_.wait_for(lock, std::chrono::seconds(20), [&] { return arrivals_ >= n; });
    }

private:
    void waitAtGate() {
        std::unique_lock<std::mutex> lock(gateMutex_);
        if (!gateClosed_)
            return;
        ++arrivals_;
        gate_.notify_all();
        gate_.wait(lock, [&] { return !gateClosed_ || passes_ > 0; });
        if (gateClosed_)
            --passes_;
    }

    std::mutex gateMutex_;
    std::condition_variable gate_;
    bool gateClosed_ = false;
    int arrivals_ = 0;
    int passes_ = 0;
};

class Archiving final : public ARA::Host::ArchivingControllerInterface {
public:
    ARA::ARASize getArchiveSize(ARA::ARAArchiveReaderHostRef ref) noexcept override {
        return ARA::ARASize(ArchiveInConverter::fromHostRef(ref)->getSize());
    }
    bool readBytesFromArchive(ARA::ARAArchiveReaderHostRef ref, ARA::ARASize position, ARA::ARASize length,
                              ARA::ARAByte buffer[]) noexcept override {
        auto* block = ArchiveInConverter::fromHostRef(ref);
        if (position + length > block->getSize())
            return false;
        std::memcpy(buffer, static_cast<const char*>(block->getData()) + position, length);
        return true;
    }
    bool writeBytesToArchive(ARA::ARAArchiveWriterHostRef ref, ARA::ARASize position, ARA::ARASize length,
                             const ARA::ARAByte buffer[]) noexcept override {
        auto* out = ArchiveOutConverter::fromHostRef(ref);
        return out->setPosition(juce::int64(position)) && out->write(buffer, length);
    }
    void notifyDocumentArchivingProgress(float) noexcept override {}
    void notifyDocumentUnarchivingProgress(float) noexcept override {}
    ARA::ARAPersistentID getDocumentArchiveID(ARA::ARAArchiveReaderHostRef) noexcept override {
        return araFactory != nullptr ? araFactory->documentArchiveID : nullptr;
    }
};

struct Progress {
    int started = 0, updated = 0, completed = 0;
    float last = 0.0f;
};

class ModelUpdates final : public ARA::Host::ModelUpdateControllerInterface {
public:
    void notifyAudioSourceAnalysisProgress(ARA::ARAAudioSourceHostRef ref, ARA::ARAAnalysisProgressState state,
                                           float value) noexcept override {
        Progress& p = progress[SourceConverter::fromHostRef(ref)];
        if (state == ARA::kARAAnalysisProgressStarted)
            ++p.started;
        else if (state == ARA::kARAAnalysisProgressUpdated)
            ++p.updated;
        else
            ++p.completed;
        p.last = value;
    }
    void notifyAudioSourceContentChanged(ARA::ARAAudioSourceHostRef, const ARA::ARAContentTimeRange*,
                                         ARA::ContentUpdateScopes) noexcept override {}
    void notifyAudioModificationContentChanged(ARA::ARAAudioModificationHostRef, const ARA::ARAContentTimeRange*,
                                               ARA::ContentUpdateScopes) noexcept override {
        ++modificationChanges;
    }
    void notifyPlaybackRegionContentChanged(ARA::ARAPlaybackRegionHostRef, const ARA::ARAContentTimeRange*,
                                            ARA::ContentUpdateScopes) noexcept override {}
    void notifyDocumentDataChanged() noexcept override {}

    std::map<TestSource*, Progress> progress;  // the host's thread only (notifyModelUpdates)
    int modificationChanges = 0;
};

struct PlayHead final : juce::AudioPlayHead {
    juce::Optional<PositionInfo> getPosition() const override {
        PositionInfo info;
        info.setTimeInSamples(time);
        info.setTimeInSeconds(double(time) / rate);
        info.setIsPlaying(playing);
        return info;
    }
    std::int64_t time = 0;
    double rate = 48000.0;
    bool playing = true;
};

// One host document: one plugin instance bound as playback renderer, its
// musical context and region sequence, sources, modifications, regions
struct Document {
    std::unique_ptr<juce::AudioPluginInstance> plugin;
    std::unique_ptr<juce::ARAHostDocumentController> host;
    AudioAccess* access = nullptr;
    ModelUpdates* updates = nullptr;
    HM::PlaybackRendererInterface renderer;
    std::unique_ptr<HM::MusicalContext> context;
    std::unique_ptr<HM::RegionSequence> sequence;

    struct Source {
        std::unique_ptr<TestSource> data;
        std::unique_ptr<HM::AudioSource> source;
        std::unique_ptr<HM::AudioModification> modification;
    };
    std::vector<std::unique_ptr<Source>> sources;
    std::vector<std::unique_ptr<HM::PlaybackRegion>> regions;
    PlayHead playHead;
    int channels = 2;

    ARA::Host::DocumentController& dc() { return host->getDocumentController(); }

    ~Document() {
        if (plugin)
            plugin->releaseResources();
        // Remove the regions from the renderer one by one: JUCE 9.0.5's
        // PlaybackRegionRegistry destructor erases from the map it iterates
        // when regions are still registered (a crash), so leave it empty
        if (renderer.isValid())
            for (auto& region : regions)
                renderer.remove(*region);
        renderer = {};
        plugin.reset();  // bound instances go before the document controller
        regions.clear();
        for (auto& s : sources) {
            s->modification.reset();
            s->source.reset();
        }
        sequence.reset();
        context.reset();
        host.reset();
    }

    void notify() { dc().notifyModelUpdates(); }

    Source& addSource(Audio data, double rate, const juce::String& id) {
        auto s = std::make_unique<Source>();
        s->data = std::make_unique<TestSource>();
        s->data->data = std::move(data);
        s->data->rate = rate;
        s->data->id = id;
        auto props = HM::AudioSource::getEmptyProperties();
        props.name = s->data->id.toRawUTF8();
        props.persistentID = s->data->id.toRawUTF8();
        props.sampleCount = s->data->length();
        props.sampleRate = rate;
        props.channelCount = ARA::ARAChannelCount(s->data->data.size());
        props.merits64BitSamples = false;
        const juce::ARAEditGuard guard(dc());
        s->source = std::make_unique<HM::AudioSource>(SourceConverter::toHostRef(s->data.get()), dc(), props);
        s->source->enableAudioSourceSamplesAccess(true);
        auto modProps = HM::AudioModification::getEmptyProperties();
        const juce::String modId = id + "-mod";
        modProps.name = modId.toRawUTF8();
        modProps.persistentID = modId.toRawUTF8();
        s->modification = std::make_unique<HM::AudioModification>(
            HM::ConversionFunctions<Source*, ARA::ARAAudioModificationHostRef>::toHostRef(s.get()), dc(),
            *s->source, modProps);
        sources.push_back(std::move(s));
        return *sources.back();
    }

    static ARA::ARAPlaybackRegionProperties regionProperties(const Document& doc, double startInSource,
                                                             double duration, double startOnTimeline) {
        auto props = HM::PlaybackRegion::getEmptyProperties();
        props.transformationFlags = ARA::kARAPlaybackTransformationNoChanges;
        props.startInModificationTime = startInSource;
        props.durationInModificationTime = duration;
        props.startInPlaybackTime = startOnTimeline;
        props.durationInPlaybackTime = duration;
        props.musicalContextRef = doc.context->getPluginRef();
        props.regionSequenceRef = doc.sequence->getPluginRef();
        props.name = nullptr;
        props.color = nullptr;
        return props;
    }

    // Times in seconds. The plugin must be unprepared (ARA's rule for adding regions).
    HM::PlaybackRegion& addRegion(Source& s, double startInSource, double duration, double startOnTimeline) {
        const juce::ARAEditGuard guard(dc());
        regions.push_back(std::make_unique<HM::PlaybackRegion>(
            HM::ConversionFunctions<void*, ARA::ARAPlaybackRegionHostRef>::toHostRef(nullptr), dc(),
            *s.modification, regionProperties(*this, startInSource, duration, startOnTimeline)));
        renderer.add(*regions.back());
        return *regions.back();
    }

    void moveRegion(HM::PlaybackRegion& region, double startInSource, double duration, double startOnTimeline) {
        const juce::ARAEditGuard guard(dc());
        region.update(regionProperties(*this, startInSource, duration, startOnTimeline));
    }

    // Applies an archive of document settings, as when a project is opened
    bool restore(juce::MemoryBlock archive) {
        const juce::ARAEditGuard guard(dc());
        return dc().restoreObjectsFromArchive(ArchiveInConverter::toHostRef(&archive), nullptr);
    }

    juce::MemoryBlock store() {
        juce::MemoryOutputStream out;
        if (!dc().storeObjectsToArchive(ArchiveOutConverter::toHostRef(&out), nullptr))
            fail("storing the ARA archive failed");
        return out.getMemoryBlock();
    }

    // Prepares the plugin (a full reset) and renders the timeline range
    // [start, start + length) in blocks of `block`. Offline renders wait for
    // analysis still running; realtime ones play what is there.
    Audio render(double rate, std::int64_t start, std::int64_t length, bool realtime, int block = 512) {
        plugin->releaseResources();
        plugin->setNonRealtime(!realtime);
        plugin->setRateAndBufferSizeDetails(rate, 4096);
        plugin->prepareToPlay(rate, 4096);
        return renderPrepared(rate, start, length, block);
    }

    Audio renderPrepared(double rate, std::int64_t start, std::int64_t length, int block = 512) {
        playHead.rate = rate;
        playHead.playing = true;
        Audio out(size_t(channels), std::vector<float>(size_t(length), 0.0f));
        juce::MidiBuffer midi;
        for (std::int64_t pos = 0; pos < length;) {
            const int n = int(std::min<std::int64_t>(block, length - pos));
            juce::AudioBuffer<float> buffer(channels, n);
            for (int c = 0; c < channels; ++c)
                for (int i = 0; i < n; ++i)
                    buffer.setSample(c, i, 0.25f);  // the renderer must overwrite whatever comes in
            playHead.time = start + pos;
            plugin->processBlock(buffer, midi);
            for (int c = 0; c < channels; ++c)
                for (int i = 0; i < n; ++i)
                    out[size_t(c)][size_t(pos + i)] = buffer.getSample(c, i);
            pos += n;
        }
        notify();  // deliver analysis progress to the "host"
        return out;
    }
};

struct Host {
    juce::VST3PluginFormatHeadless format;
    juce::PluginDescription description;
    juce::ARAFactoryWrapper factory;
    // An instance that is never bound, alive for the whole test: it keeps the
    // plugin's module loaded. Otherwise destroying a document's last instance
    // can unload the module (JUCE's VST3 hosting releases it with its last
    // instance) while the document's ARA objects, which call into it, are
    // still to be destroyed. A DAW keeps the module loaded the same way.
    std::unique_ptr<juce::AudioPluginInstance> keepLoaded;

    std::unique_ptr<juce::AudioPluginInstance> createPlugin(int channels) {
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

    std::unique_ptr<Document> createDocument(int channels = 2) {
        if (!keepLoaded)
            keepLoaded = createPlugin(2);
        auto doc = std::make_unique<Document>();
        doc->channels = channels;
        doc->plugin = createPlugin(channels);
        if (!doc->plugin)
            return nullptr;
        if (!factory.get()) {
            juce::createARAFactoryAsync(*doc->plugin, [this](juce::ARAFactoryWrapper f) { factory = std::move(f); });
            araFactory = factory.get();
        }
        if (!factory.get()) {
            fail("the plugin gives no ARA factory");
            return nullptr;
        }
        auto access = std::make_unique<AudioAccess>();
        auto updates = std::make_unique<ModelUpdates>();
        doc->access = access.get();
        doc->updates = updates.get();
        doc->host = juce::ARAHostDocumentController::create(factory, "test document", std::move(access),
                                                            std::make_unique<Archiving>(), nullptr,
                                                            std::move(updates), nullptr);
        if (!doc->host) {
            fail("could not create the ARA document controller");
            return nullptr;
        }
        const auto roles = ARA::kARAPlaybackRendererRole | ARA::kARAEditorRendererRole | ARA::kARAEditorViewRole;
        const auto instance = doc->host->bindDocumentToPluginInstance(*doc->plugin, roles,
                                                                      ARA::kARAPlaybackRendererRole);
        doc->renderer = instance.getPlaybackRendererInterface();
        if (!doc->renderer.isValid())
            fail("the plugin instance gives no playback renderer");
        doc->plugin->setPlayHead(&doc->playHead);
        {
            const juce::ARAEditGuard guard(doc->dc());
            auto ctxProps = HM::MusicalContext::getEmptyProperties();
            ctxProps.name = "context";
            ctxProps.orderIndex = 0;
            ctxProps.color = nullptr;
            doc->context = std::make_unique<HM::MusicalContext>(
                HM::ConversionFunctions<void*, ARA::ARAMusicalContextHostRef>::toHostRef(nullptr), doc->dc(),
                ctxProps);
            auto seqProps = HM::RegionSequence::getEmptyProperties();
            seqProps.name = "track";
            seqProps.orderIndex = 0;
            seqProps.musicalContextRef = doc->context->getPluginRef();
            seqProps.color = nullptr;
            doc->sequence = std::make_unique<HM::RegionSequence>(
                HM::ConversionFunctions<void*, ARA::ARARegionSequenceHostRef>::toHostRef(nullptr), doc->dc(),
                seqProps);
        }
        return doc;
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
    return juce::AudioProcessor::getXmlFromBinary(component.getData(), int(component.getSize()));
}

// What the plugin says about the document (for information, in its state)
juce::String araMessage(juce::AudioPluginInstance& plugin, const juce::String& what = "message") {
    auto xml = getPluginXml(plugin);
    auto* ara = xml ? xml->getChildByName("ARA") : nullptr;
    return ara ? ara->getStringAttribute(what) : juce::String("<no ARA status>");
}

juce::String sourceStatus(juce::AudioPluginInstance& plugin) {
    auto xml = getPluginXml(plugin);
    auto* ara = xml ? xml->getChildByName("ARA") : nullptr;
    juce::String s;
    if (ara != nullptr)
        for (auto* e : ara->getChildWithTagNameIterator("Source"))
            s << e->getStringAttribute("name") << ": " << e->getStringAttribute("state") << " "
              << e->getStringAttribute("message") << "; ";
    return s;
}

double compareChannels(const Audio& out, const Audio& ref, std::size_t outStart, std::size_t refStart,
                       std::size_t n) {
    double worst = 0.0;
    for (size_t c = 0; c < out.size(); ++c)
        worst = std::max(worst, relErr(out[c], outStart, ref[std::min(c, ref.size() - 1)], refStart, n));
    return worst;
}

// Sets a parameter as a host does: the change reaches the plugin with the next block
void setParameter(Document& doc, juce::AudioProcessorParameter& p, float value, double rate) {
    p.setValue(value);
    doc.render(rate, 0, 64, false);
}

// Calls the host's notifyModelUpdates every 2 ms (as a DAW's main thread
// does) until `done` or 20 s
void pollUpdates(Document& doc, const std::function<bool()>& done) {
    for (int i = 0; i < 10000 && !done(); ++i) {
        doc.notify();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

// ============================================================================
// Tests

// A whole source as one region at timeline 0: the output is the core's
// offline result at the source's rate, sample-aligned, for each channel
void testWholeSource(Host& host, const char* modelName, int rate, int channels, bool highBandPass = false) {
    std::printf("whole source: %s, %d Hz, %d ch%s\n", modelName, rate, channels,
                highBandPass ? ", high band passed" : "");
    const juce::String path = golden(juce::String(modelName) + ".model.bin");
    auto model = loadModel(path);
    auto doc = host.createDocument(channels);
    if (!doc || !model)
        return;
    // The golden clip (resampled by plain repetition when needed: any audio
    // will do) and a synthetic one, 0.75 s
    Audio data = noise(channels, size_t(rate) * 3 / 4, 11 + rate % 7);
    const std::vector<float> g = goldenInput(modelName);
    for (size_t i = 0; i < data[0].size(); ++i)
        data[0][i] = g[(i * 16000 / size_t(rate)) % g.size()];
    auto& src = doc->addSource(data, rate, "source-1");
    doc->addRegion(src, 0.0, double(data[0].size()) / rate, 0.0);
    check(doc->restore(settingsArchive(path, highBandPass)), "restoring the settings archive failed");
    const auto n = std::int64_t(data[0].size());
    const Audio out = doc->render(rate, 0, n + 1000, false);
    const auto hb = highBandPass ? denoiser::HighBand::PassThrough : denoiser::HighBand::Drop;
    Audio ref;
    for (const auto& ch : data)
        ref.push_back(reference(*model, rate, ch, hb));
    const double err = compareChannels(out, ref, 0, 0, size_t(n));
    std::printf("  error vs core: %.2e; status: %s\n", err, sourceStatus(*doc->plugin).toRawUTF8());
    check(err <= kTolerance, juce::String(modelName) + " at " + juce::String(rate) + ": plugin != core, error " +
                                 juce::String(err));
    for (const auto& ch : out)
        check(isSilent(ch, size_t(n), 1000), "after the region: silence");
    check(doc->access->reads.load() > 0, "the plugin did not read through the host's reader");
}

// Regions moved, trimmed, split, and two regions of one source, at 48 kHz
void testRegions(Host& host) {
    std::printf("regions: moved, trimmed, split, two of one source (48 kHz stereo)\n");
    const juce::String path = golden("mlp_complex_default.model.bin");
    auto model = loadModel(path);
    auto doc = host.createDocument(2);
    if (!doc || !model)
        return;
    const int rate = 48000;
    const Audio data = noise(2, 48000, 21);  // 1 s
    auto& src = doc->addSource(data, rate, "source-r");
    check(doc->restore(settingsArchive(path)), "restore");
    Audio ref;
    for (const auto& ch : data)
        ref.push_back(reference(*model, rate, ch));

    // Moved: the whole source starting at 0.5 s on the timeline
    auto& region = doc->addRegion(src, 0.0, 1.0, 0.5);
    Audio out = doc->render(rate, 0, 96000, false, 333);
    double err = compareChannels(out, ref, 24000, 0, 48000);
    std::printf("  moved to 0.5 s: %.2e\n", err);
    check(err <= kTolerance, "moved region: error " + juce::String(err));
    check(isSilent(out[0], 0, 24000) && isSilent(out[1], 72000, 24000), "moved region: silence around it");

    // Trimmed (and moved): source 0.25 .. 0.6 s at timeline 0.1 s; changed while prepared
    doc->moveRegion(region, 0.25, 0.35, 0.1);
    out = doc->renderPrepared(rate, 0, 48000, 512);
    err = compareChannels(out, ref, 4800, 12000, 16800);
    std::printf("  trimmed (source 0.25-0.6 s at 0.1 s), edited while prepared: %.2e\n", err);
    check(err <= kTolerance, "trimmed region: error " + juce::String(err));
    check(isSilent(out[0], 0, 4800) && isSilent(out[0], 4800 + 16800, 48000 - 4800 - 16800),
          "trimmed region: silence outside");

    // Split at 0.4 s (source time): two regions, the second moved by 0.2 s;
    // plus a third region of the same source elsewhere (source 0..0.1 s at 1.5 s)
    doc->plugin->releaseResources();
    doc->moveRegion(region, 0.0, 0.4, 0.0);
    doc->addRegion(src, 0.4, 0.6, 0.6);
    doc->addRegion(src, 0.0, 0.1, 1.5);
    out = doc->render(rate, 0, 96000, false, 1024);
    const double e1 = compareChannels(out, ref, 0, 0, 19200);
    const double e2 = compareChannels(out, ref, 28800, 19200, 28800);
    const double e3 = compareChannels(out, ref, 72000, 0, 4800);
    std::printf("  split + moved + a second use of the source: %.2e, %.2e, %.2e\n", e1, e2, e3);
    check(std::max({e1, e2, e3}) <= kTolerance, "split regions");
    check(isSilent(out[0], 19200, 9600) && isSilent(out[1], 57600, 14400), "split: gap is silent");
}

// The original plays until the denoised audio is ready (realtime), then the
// denoised audio, in the same running renderer. Held by the host alone, as a
// DAW whose reads are slow could: there is no test hook in the plugin.
//
// The plugin publishes a source's original as soon as it has read it, then
// denoises. To stop it in between, the host keeps its first read waiting,
// tells the plugin meanwhile that the source's samples changed (the same
// samples here), and lets that read finish: the plugin publishes what it
// read, stops the now out-of-date denoising at its first frame and reads
// again, and the host keeps that second read waiting. The source has then
// been read and its analysis is not finished, for as long as the host likes.
void testBeforeAnalysis(Host& host) {
    std::printf("playback before the analysis finishes gives the original\n");
    const juce::String path = golden("tcn_magnitude_default.model.bin");
    auto model = loadModel(path);
    auto doc = host.createDocument(2);
    if (!doc || !model)
        return;
    AudioAccess& access = *doc->access;
    // The plugin's analysis thread must not be left waiting at the gate when the document goes
    struct OpenGateAtEnd {
        AudioAccess& access;
        ~OpenGateAtEnd() { access.openGate(); }
    } openGateAtEnd{access};
    // The gate is closed and the model given before the source exists: the
    // plugin reads a source as soon as the edit that adds it ends, so this is
    // the only order in which its first read is held whatever the timing
    access.closeGate();
    check(doc->restore(settingsArchive(path)), "restore");
    const Audio data = noise(2, 24000, 31);  // one host read (the plugin asks for 65536 samples at a time)
    auto& src = doc->addSource(data, 48000, "source-h");
    doc->addRegion(src, 0.0, 0.5, 0.0);

    // Prepared for realtime playback before the analysis finishes: what it
    // plays is replaced while it stays prepared
    doc->plugin->releaseResources();
    doc->plugin->setNonRealtime(false);
    doc->plugin->setRateAndBufferSizeDetails(48000, 4096);
    doc->plugin->prepareToPlay(48000, 4096);

    if (!access.waitForArrivals(1)) {
        fail("the plugin did not read the source");
        return;
    }
    Audio out = doc->renderPrepared(48000, 0, 24000);
    check(isSilent(out[0], 0, 24000) && isSilent(out[1], 0, 24000), "before the source was read: not silence");

    {
        const juce::ARAEditGuard guard(doc->dc());
        doc->dc().updateAudioSourceContent(src.source->getPluginRef(), nullptr,
                                           ARA::ContentUpdateScopes::samplesAreAffected());
    }
    access.letOneThrough();
    if (!access.waitForArrivals(2)) {
        fail("the plugin did not read the source again after the content update");
        return;
    }
    out = doc->renderPrepared(48000, 0, 24000);
    check(sameSamples(out[0], 0, data[0], 0, 24000) && sameSamples(out[1], 0, data[1], 0, 24000),
          "before the analysis finished: not the original audio, bit for bit");
    check(sourceStatus(*doc->plugin).contains("source-h: reading"),
          "status while the read waits: " + sourceStatus(*doc->plugin));
    // The host hears about it (a quick analysis that starts and ends between
    // two notifyModelUpdates calls is not reported at all: the ARA library
    // drops a start followed by its completion)
    Progress& p = doc->updates->progress[src.data.get()];
    pollUpdates(*doc, [&] { return p.started > 0; });
    check(p.started >= 1, "the host was not told that the analysis started");
    const int completedBefore = p.completed;  // the stopped job's, if it was reported

    access.openGate();
    pollUpdates(*doc, [&] { return sourceStatus(*doc->plugin).contains("source-h: denoised"); });
    pollUpdates(*doc, [&] { return p.completed > completedBefore; });
    std::printf("  progress notifications: %d started, %d updates (last %.2f), %d completed\n", p.started,
                p.updated, double(p.last), p.completed);
    check(p.completed > completedBefore, "the host was not told that the analysis completed");
    out = doc->renderPrepared(48000, 0, 24000);  // still realtime, not prepared again
    Audio ref;
    for (const auto& ch : data)
        ref.push_back(reference(*model, 48000, ch));
    const double err = compareChannels(out, ref, 0, 0, 24000);
    std::printf("  original while the analysis waits (exact); denoised after, same renderer: %.2e\n", err);
    check(err <= kTolerance, "after the analysis: error " + juce::String(err));
}

// Content update, model change, mix and bypass, archive store/restore, missing model
void testChanges(Host& host) {
    std::printf("content update, model change, mix/bypass, archive, missing model (44.1 kHz mono)\n");
    const juce::String mlpPath = golden("mlp_magnitude_default.model.bin");
    const juce::String snnPath = golden("snn_complex_default.model.bin");
    auto mlp = loadModel(mlpPath);
    auto snn = loadModel(snnPath);
    auto doc = host.createDocument(1);
    if (!doc || !mlp || !snn)
        return;
    const int rate = 44100;
    Audio data = noise(1, 22050, 41);
    auto& src = doc->addSource(data, rate, "source-c");
    doc->addRegion(src, 0.0, 0.5, 0.0);
    check(doc->restore(settingsArchive(mlpPath)), "restore");
    Audio out = doc->render(rate, 0, 22050, false);
    double err = compareChannels(out, {reference(*mlp, rate, data[0])}, 0, 0, 22050);
    std::printf("  MLP: %.2e\n", err);
    check(err <= kTolerance, "MLP: error " + juce::String(err));

    // The host edits the source's samples (destructively): access off, change, on, content update
    {
        src.source->enableAudioSourceSamplesAccess(false);
        data = noise(1, 22050, 42);
        src.data->data = data;
        src.source->enableAudioSourceSamplesAccess(true);
        const juce::ARAEditGuard guard(doc->dc());
        doc->dc().updateAudioSourceContent(src.source->getPluginRef(), nullptr,
                                           ARA::ContentUpdateScopes::samplesAreAffected());
    }
    out = doc->render(rate, 0, 22050, false);
    err = compareChannels(out, {reference(*mlp, rate, data[0])}, 0, 0, 22050);
    std::printf("  after a content update: %.2e\n", err);
    check(err <= kTolerance, "content update: error " + juce::String(err));

    // Another model (as from the editor or a restored project)
    check(doc->restore(settingsArchive(snnPath)), "restore SNN");
    out = doc->render(rate, 0, 22050, false);
    const std::vector<float> snnRef = reference(*snn, rate, data[0]);
    err = compareChannels(out, {snnRef}, 0, 0, 22050);
    std::printf("  after a model change (SNN): %.2e\n", err);
    check(err <= kTolerance, "model change: error " + juce::String(err));

    // Mix and bypass: this instance's parameters
    auto* params = &doc->plugin->getParameters();
    juce::AudioProcessorParameter* mix = nullptr;
    for (auto* p : *params)
        if (p->getName(20) == "Mix")
            mix = p;
    auto* bypass = doc->plugin->getBypassParameter();
    if (mix == nullptr || bypass == nullptr) {
        fail("no mix / bypass parameter");
    } else {
        setParameter(*doc, *bypass, 1.0f, rate);
        out = doc->render(rate, 0, 22050, false);
        check(sameSamples(out[0], 0, data[0], 0, 22050), "bypass: not the original, bit for bit");
        setParameter(*doc, *bypass, 0.0f, rate);
        setParameter(*doc, *mix, 0.0f, rate);
        out = doc->render(rate, 0, 22050, false);
        check(sameSamples(out[0], 0, data[0], 0, 22050), "mix 0: not the original, bit for bit");
        setParameter(*doc, *mix, 0.5f, rate);
        out = doc->render(rate, 0, 22050, false);
        std::vector<float> mean(22050);
        for (size_t i = 0; i < mean.size(); ++i)
            mean[i] = 0.5f * (data[0][i] + snnRef[i]);
        err = relErr(out[0], 0, mean, 0, 22050);
        std::printf("  mix 0.5 vs the mean of original and denoised: %.2e\n", err);
        check(err <= kTolerance, "mix 0.5: error " + juce::String(err));
        setParameter(*doc, *mix, 1.0f, rate);
    }

    // Store the document; a new document restores it (as when reopening a project)
    const juce::MemoryBlock archive = doc->store();
    out = doc->render(rate, 0, 22050, false);
    auto doc2 = host.createDocument(1);
    if (!doc2)
        return;
    auto& src2 = doc2->addSource(data, rate, "source-c");
    doc2->addRegion(src2, 0.0, 0.5, 0.0);
    check(doc2->restore(archive), "restoring the stored archive failed");
    const Audio again = doc2->render(rate, 0, 22050, false);
    err = compareChannels(again, out, 0, 0, 22050);
    std::printf("  stored -> new document -> restored: %.2e (archive %d bytes)\n", err, int(archive.getSize()));
    check(err <= kTolerance, "restored document gives other output: " + juce::String(err));
    check(araMessage(*doc2->plugin, "modelPath") == snnPath, "restored model path: " +
                                                                 araMessage(*doc2->plugin, "modelPath"));

    // A project whose model file is missing: the original plays, with a message
    auto doc3 = host.createDocument(1);
    if (!doc3)
        return;
    auto& src3 = doc3->addSource(data, rate, "source-c");
    doc3->addRegion(src3, 0.0, 0.5, 0.0);
    check(doc3->restore(settingsArchive(golden("no_such_model.model.bin"))), "restore missing");
    out = doc3->render(rate, 0, 22050, false);
    check(sameSamples(out[0], 0, data[0], 0, 22050), "missing model: not the original, bit for bit");
    check(araMessage(*doc3->plugin).contains("not found"), "missing model message: " + araMessage(*doc3->plugin));
    check(araMessage(*doc3->plugin, "modelPath").endsWith("no_such_model.model.bin"), "the path is kept");
}

// A source at 44.1 kHz played at 48 kHz: the 16 kHz result upsampled to 48 kHz
void testOtherPlaybackRate(Host& host) {
    std::printf("44.1 kHz source played at 48 kHz\n");
    const juce::String path = golden("mlp_mask_default.model.bin");
    auto model = loadModel(path);
    auto doc = host.createDocument(2);
    if (!doc || !model)
        return;
    const Audio data = noise(2, 44100, 51);
    auto& src = doc->addSource(data, 44100, "source-44");
    doc->addRegion(src, 0.0, 1.0, 0.0);
    check(doc->restore(settingsArchive(path)), "restore");
    const Audio out = doc->render(48000, 0, 48000, false);
    auto in44 = denoiser::ResampleFilter::create(44100);
    auto out48 = denoiser::ResampleFilter::create(48000);
    double worst = 0.0;
    for (size_t c = 0; c < 2; ++c) {
        auto net = model->makeNetwork();
        const auto x16 = denoiser::clipToModelRate(*in44, data[c].data(), data[c].size());
        const auto y16 = denoiser::denoiseClip(model->info(), *net, x16.data(), x16.size());
        const auto ref = denoiser::clipFromModelRate(*out48, y16.data(), y16.size(), 48000);
        worst = std::max(worst, relErr(out[c], 0, ref, 0, 48000));
    }
    std::printf("  vs core (16 kHz result upsampled to 48 kHz): %.2e\n", worst);
    check(worst <= kTolerance, "other playback rate: error " + juce::String(worst));
}

// Short, empty and unsupported sources play their original audio (or nothing)
void testEdgeSources(Host& host) {
    std::printf("short source with a centred model, empty source, 8 kHz source, no model\n");
    const juce::String centred = golden("offline_mlp_mask_centred.model.bin");
    auto model = loadModel(centred);
    auto doc = host.createDocument(1);
    if (!doc || !model)
        return;
    const std::size_t min16 = denoiser::minClipSamples(model->info());
    const Audio shortData = noise(1, 3 * (min16 - 1), 61);  // too short at 48 kHz
    const Audio emptyData{std::vector<float>{}};
    const Audio lowData = noise(1, 4000, 62);
    // Before any model: the original
    auto& s1 = doc->addSource(shortData, 48000, "short");
    doc->addRegion(s1, 0.0, double(shortData[0].size()) / 48000.0, 0.0);
    Audio out = doc->render(48000, 0, std::int64_t(shortData[0].size()), false);
    check(sameSamples(out[0], 0, shortData[0], 0, shortData[0].size()), "no model: the original");
    check(araMessage(*doc->plugin).contains("No model"), "no-model message: " + araMessage(*doc->plugin));

    doc->plugin->releaseResources();
    auto& s2 = doc->addSource(emptyData, 48000, "empty");
    auto& s3 = doc->addSource(lowData, 8000, "8k");
    doc->addRegion(s2, 0.0, 0.0, 1.0);
    doc->addRegion(s3, 0.0, 0.5, 2.0);
    check(doc->restore(settingsArchive(centred)), "restore");
    out = doc->render(48000, 0, std::int64_t(shortData[0].size()), false);
    check(sameSamples(out[0], 0, shortData[0], 0, shortData[0].size()), "too short: the original");
    const juce::String status = sourceStatus(*doc->plugin);
    std::printf("  %s\n", status.toRawUTF8());
    check(status.contains("Too short"), "too-short message: " + status);
    // The 8 kHz source can't be resampled for 48 kHz playback: silence there (and a message)
    out = doc->render(48000, 96000, 24000, false);  // waits for that source's analysis
    check(isSilent(out[0], 0, 24000), "8 kHz source at 48 kHz: silence");
    check(sourceStatus(*doc->plugin).contains("not supported"), "8 kHz message: " + sourceStatus(*doc->plugin));
    // ... and at its own rate, the original
    out = doc->render(8000, 16000, 4000, false);
    check(sameSamples(out[0], 0, lowData[0], 0, 4000), "8 kHz source at 8 kHz: the original");
}

// An audio thread renders in realtime, without pause, while the document
// thread changes the model, the high band and a source's samples, so new
// analysis results replace the ones being played (for ThreadSanitizer /
// AddressSanitizer). Every block must be finite, and in the end the output
// is the last model's.
void testRenderWhileAnalysing(Host& host) {
    std::printf("realtime rendering on another thread while the analysis replaces results\n");
    const juce::String mlp = golden("mlp_magnitude_default.model.bin");
    const juce::String snn = golden("snn_complex_default.model.bin");
    const juce::String tcn = golden("tcn_magnitude_default.model.bin");
    auto doc = host.createDocument(2);
    if (!doc)
        return;
    const Audio a = noise(2, 48000, 91), b = noise(1, 24000, 92);
    auto& sa = doc->addSource(a, 48000, "stress-a");
    auto& sb = doc->addSource(b, 48000, "stress-b");
    doc->addRegion(sa, 0.0, 1.0, 0.0);
    doc->addRegion(sb, 0.0, 0.5, 0.25);
    doc->plugin->releaseResources();
    doc->plugin->setNonRealtime(false);
    doc->plugin->prepareToPlay(48000, 512);
    std::atomic<bool> stop{false};
    std::atomic<long> blocks{0}, bad{0};
    std::thread audio([&] {
        std::vector<float> left(256), right(256);
        float* ptrs[] = {left.data(), right.data()};
        juce::MidiBuffer midi;
        std::int64_t time = 0;
        while (!stop.load()) {
            juce::AudioBuffer<float> buffer(ptrs, 2, 256);
            doc->playHead.time = time;
            doc->plugin->processBlock(buffer, midi);
            for (float v : left)
                if (!std::isfinite(v) || std::fabs(v) > 100.0f)
                    ++bad;
            time = (time + 256) % 48000;
            ++blocks;
        }
    });
    const juce::String models[] = {mlp, snn, tcn, mlp};
    for (int round = 0; round < 8; ++round) {
        check(doc->restore(settingsArchive(models[round % 4], round % 3 == 1)), "restore");
        if (round % 3 == 2) {
            sb.source->enableAudioSourceSamplesAccess(false);
            for (auto& v : sb.data->data[0])
                v *= -1.0f;
            sb.source->enableAudioSourceSamplesAccess(true);
            const juce::ARAEditGuard guard(doc->dc());
            doc->dc().updateAudioSourceContent(sb.source->getPluginRef(), nullptr,
                                               ARA::ContentUpdateScopes::samplesAreAffected());
        }
        doc->notify();
        std::this_thread::sleep_for(std::chrono::milliseconds(15 * (round % 4)));
    }
    doc->notify();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop = true;
    audio.join();
    std::printf("  %ld blocks rendered during 8 changes, %ld bad\n", blocks.load(), bad.load());
    check(bad.load() == 0, "non-finite or wild output while results were replaced");
    // Settles on the last settings (round 7: the MLP, high band passed through)
    const Audio out = doc->render(48000, 0, 48000, false);
    auto model = loadModel(mlp);
    if (model) {
        const auto hb = denoiser::HighBand::PassThrough;
        const double err = compareChannels(out, {reference(*model, 48000, a[0], hb), reference(*model, 48000, a[1], hb)},
                                           0, 0, 12000);
        check(err <= kTolerance, "after the changes: not the last model's output, error " + juce::String(err));
    }
}

#if DENOISER_COUNT_WRAPPER_ALLOCATIONS
void testAllocations(Host& host) {
    std::printf("allocations while rendering ARA regions through the VST3 interface\n");
    const juce::String path = golden("mlp_magnitude_default.model.bin");
    auto doc = host.createDocument(2);
    if (!doc)
        return;
    const Audio data = noise(2, 48000, 71);
    auto& src = doc->addSource(data, 48000, "source-a");
    doc->addRegion(src, 0.0, 0.5, 0.0);
    doc->addRegion(src, 0.5, 0.5, 0.7);
    check(doc->restore(settingsArchive(path)), "restore");
    doc->render(48000, 0, 48000, false);  // offline: analysis done
    doc->plugin->releaseResources();
    doc->plugin->setNonRealtime(false);
    doc->plugin->prepareToPlay(48000, 512);
    std::vector<float> left(512), right(512);
    float* ptrs[] = {left.data(), right.data()};
    juce::MidiBuffer midi;
    juce::Random random(3);
    const auto run = [&](int blocks) {
        return alloc::countOnThisThread([&] {
            for (int b = 0; b < blocks; ++b) {
                const int n = 1 + random.nextInt(512);
                juce::AudioBuffer<float> buffer(ptrs, 2, n);
                doc->playHead.time += n;
                doc->plugin->processBlock(buffer, midi);
            }
        });
    };
    run(10);
    const long count = run(400);
    std::printf("  %ld\n", count);
    check(count == 0, "ARA rendering allocated " + juce::String(count) + " times");
}
#endif

// Not a test: how long the analysis of a 48 kHz stereo source takes
void bench(Host& host, const juce::String& modelPath, double seconds) {
    auto doc = host.createDocument(2);
    if (!doc)
        return;
    const auto n = size_t(48000.0 * seconds);
    auto& src = doc->addSource(noise(2, n, 81), 48000, "bench");
    doc->addRegion(src, 0.0, seconds, 0.0);
    const auto start = std::chrono::steady_clock::now();
    check(doc->restore(settingsArchive(modelPath)), "restore");
    doc->render(48000, 0, 512, false);  // offline: returns when the analysis is done
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("%s: %.0f s of 48 kHz stereo analysed in %.2f s (%.1fx real time); %s\n",
                juce::File(modelPath).getFileName().toRawUTF8(), seconds, elapsed, seconds / elapsed,
                sourceStatus(*doc->plugin).toRawUTF8());
}

}  // namespace

int main(int argc, char** argv) {
    const bool benchMode = argc == 6 && juce::String(argv[3]) == "bench";
    if (argc != 3 && !benchMode) {
        std::printf("usage: %s <plugin.vst3> <golden dir> [bench <model.bin> <seconds>]\n", argv[0]);
        return 2;
    }
    const auto cwd = juce::File::getCurrentWorkingDirectory();
    goldenDir = cwd.getChildFile(juce::String::fromUTF8(argv[2])).getFullPathName();
    juce::ScopedJuceInitialiser_GUI juce;

    Host host;
    juce::OwnedArray<juce::PluginDescription> found;
    host.format.findAllTypesForFile(found, juce::String::fromUTF8(argv[1]));
    if (found.isEmpty()) {
        std::printf("FAIL: no plugin found in %s\n", argv[1]);
        return 1;
    }
    host.description = *found.getFirst();
    if (!host.description.hasARAExtension) {
        std::printf("FAIL: the plugin has no ARA factory\n");
        return 1;
    }

    if (benchMode) {
        bench(host, cwd.getChildFile(juce::String::fromUTF8(argv[4])).getFullPathName(), std::atof(argv[5]));
        host.factory = {};  // releases the plugin's ARA factory while the module is loaded
        host.keepLoaded.reset();
        return failures == 0 ? 0 : 1;
    }

    try {
        testWholeSource(host, "mlp_magnitude_default", 16000, 2);
        testWholeSource(host, "snn_complex_default", 16000, 1);
        testWholeSource(host, "tcn_magnitude_default", 16000, 2);
        testWholeSource(host, "mlp_complex_default", 48000, 2);
        testWholeSource(host, "tcn_complex_plain", 44100, 1, true);
        testWholeSource(host, "offline_mlp_mask_centred", 16000, 1);
        testWholeSource(host, "offline_tcn_complex_centred", 48000, 2);
        testRegions(host);
        testBeforeAnalysis(host);
        testChanges(host);
        testOtherPlaybackRate(host);
        testEdgeSources(host);
        testRenderWhileAnalysing(host);
#if DENOISER_COUNT_WRAPPER_ALLOCATIONS
        testAllocations(host);
#endif
    } catch (const std::exception& e) {
        fail(juce::String("exception: ") + e.what());
    }

    host.factory = {};  // releases the plugin's ARA factory while the module is loaded
    host.keepLoaded.reset();
    std::printf(failures == 0 ? "all checks passed\n" : "%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
