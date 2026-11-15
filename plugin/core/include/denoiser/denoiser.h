#pragma once
// The denoiser as the plugin and the command-line tool use it: a model loaded
// once (Model) and one ChannelDenoiser per audio channel, which runs the
// causal STFT and the network hop by hop and plugs into HostRateAdapter.
//
//   // off the audio thread (allocates, throws FormatError / std::runtime_error):
//   std::shared_ptr<const denoiser::Model> model = denoiser::Model::fromMemory(bytes, size);
//   if (!model->info().realtime) ...                       // centred model: offline (ARA) only
//   std::vector<std::unique_ptr<denoiser::ChannelDenoiser>> ch;
//   for (int c = 0; c < channels; ++c)
//       ch.push_back(std::make_unique<denoiser::ChannelDenoiser>(model));
//   denoiser::HostRateAdapter adapter;
//   adapter.prepare(ch[0]->hostRateConfig(hostRate, channels));
//   for (int c = 0; c < channels; ++c) adapter.setProcessor(c, ch[c].get());
//   setLatencySamples(adapter.latencySamples());
//   // audio thread (no allocation, locks or exceptions):
//   adapter.process(in, out, numSamples);  adapter.reset();
//
// Offline (whole clip at the model's rate, either framing):
//   auto net = model->makeNetwork();
//   std::vector<float> y = denoiser::denoiseClip(model->info(), *net, x, n);
//
// Threads: a Model is immutable after loading; one shared_ptr<const Model>
// may be used by any number of ChannelDenoisers / networks on any threads.
// The weights are not copied per channel. Each ChannelDenoiser (and each
// network) is used by one thread at a time.

#include "denoiser/host_rate.h"
#include "denoiser/model_file.h"
#include "denoiser/network.h"
#include "denoiser/stream.h"

#include <cstddef>
#include <memory>
#include <string>

namespace denoiser {

class SnnWeights;

class Model {
public:
    // All of these allocate and throw: FormatError for a malformed or
    // inconsistent file, std::runtime_error for a network type this build
    // can't run (e.g. "tcn" for now). A Model that loads can make networks.
    static std::shared_ptr<const Model> load(const std::string& path);  // narrow path (see README on Windows)
    static std::shared_ptr<const Model> fromMemory(const void* bytes, std::size_t size);
    static std::shared_ptr<const Model> fromFile(ModelFile file);

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    ~Model();

    const ModelFile& file() const noexcept { return file_; }
    const ModelInfo& info() const noexcept { return file_.info(); }

    // A new network with its own state, sharing this model's weights. It
    // reads them in place: keep the Model alive at least as long (a
    // ChannelDenoiser holds a shared_ptr for that). Allocates.
    std::unique_ptr<FrameNetwork> makeNetwork() const;

    // One line for a UI or log, e.g. "mlp 130 -> 1024 -> 1024 -> 65, magnitude;
    // n_fft 128, hop 80, long window 1280; 16000 Hz; real-time"
    std::string describe() const;

private:
    explicit Model(ModelFile file);

    ModelFile file_;
    std::shared_ptr<const SnnWeights> snn_;  // SNN only: transposed copies, shared by its networks
};

// One channel of real-time denoising at the model's rate: StreamProcessor +
// network, as a HopProcessor for HostRateAdapter. Its output is the offline
// result delayed by processorDelay() = n_fft - hop samples.
class ChannelDenoiser final : public HopProcessor {
public:
    // Allocates. Throws std::invalid_argument for a centred (realtime =
    // false) model, which can't run live.
    explicit ChannelDenoiser(std::shared_ptr<const Model> model);

    // hop() samples in, hop() out; for any other hop the output is silence
    // (a configuration bug). Real-time safe.
    void processHop(const float* in, float* out, int hop) noexcept override;
    // Back to the start of a recording: STFT history, overlap-add buffers,
    // network state. Real-time safe.
    void reset() noexcept override;

    int hop() const noexcept { return stream_.hop(); }
    // The delay of processHop's output (n_fft - hop): HostRateConfig::processorDelay
    int processorDelay() const noexcept { return stream_.delay(); }
    // A HostRateConfig for this model at a host rate
    HostRateConfig hostRateConfig(int hostRate, int channels, HighBand highBand = HighBand::Drop) const noexcept;

    // The first processorDelay() samples after a reset belong to time before
    // the input (stream.h, "pre-roll") and can be very loud. Muted (zeros) by
    // default; off reproduces StreamingDenoiser.process sample for sample.
    void setMutePreRoll(bool mute) noexcept { mutePreRoll_ = mute; }
    bool mutePreRoll() const noexcept { return mutePreRoll_; }

    const Model& model() const noexcept { return *model_; }
    const std::shared_ptr<const Model>& sharedModel() const noexcept { return model_; }
    FrameNetwork& network() noexcept { return *network_; }
    const StreamProcessor& stream() const noexcept { return stream_; }

private:
    std::shared_ptr<const Model> model_;
    StreamProcessor stream_;
    std::unique_ptr<FrameNetwork> network_;
    bool mutePreRoll_ = true;
};

}  // namespace denoiser
