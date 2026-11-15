// denoise: the command-line denoiser, built on the same core as the plugin.
//
//   denoise [options] model.bin in.wav out.wav
//   denoise --version | --help
//
// Default (offline): the file must be at the model's sample rate; every
// channel is denoised on its own with denoiseClip (pipeline.enhance: same
// length, no delay; works for centred models too).
// --stream: every channel runs the way the real-time plugin runs it, at the
// file's own sample rate (any rate HostRateAdapter supports): a
// ChannelDenoiser behind a HostRateAdapter, with the latency removed
// (processClipAligned). Real-time (causal) models only.
//
// Exit status: 0 done, 1 error (unreadable or malformed file, unsupported
// rate or model, ...), 2 bad arguments.
#include "wav.h"

#include "denoiser/core.h"
#include "denoiser/denoiser.h"
#include "denoiser/host_rate.h"
#include "denoiser/offline.h"
#include "denoiser/resample.h"

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* const kUsage =
    "usage: denoise [options] model.bin in.wav out.wav\n"
    "       denoise --version | --help\n"
    "\n"
    "Denoises every channel of in.wav with the model (a file written by\n"
    "plugin/tools/export.py) and writes out.wav, the same length and aligned.\n"
    "\n"
    "options:\n"
    "  --stream          process as the real-time plugin does: per channel, the\n"
    "                    hop-by-hop denoiser behind the host-rate adapter at the\n"
    "                    file's own sample rate (16 kHz, or 22.05 to 384 kHz with\n"
    "                    resampling), latency removed. Real-time models only.\n"
    "                    Default: offline, whole clip, at the model's rate only.\n"
    "  --high-band=MODE  with --stream above 16 kHz: 'drop' (default, as the plugin)\n"
    "                    outputs only the denoised band below ~7.5 kHz; 'pass' adds\n"
    "                    back the band above 8 kHz, untouched\n"
    "  --format=FMT      out.wav sample format: same (default: as in.wav), pcm16,\n"
    "                    pcm24, pcm32, float32, float64\n"
    "  --float           the same as --format=float32\n"
    "  -q, --quiet       no report on stderr\n";

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Options {
    bool stream = false;
    bool quiet = false;
    denoiser::HighBand highBand = denoiser::HighBand::Drop;
    bool sameFormat = true;
    wav::Format format = wav::Format::Float32;
    std::vector<std::filesystem::path> files;
};

bool startsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

Options parse(const std::vector<std::filesystem::path>& args) {
    Options o;
    bool optionsDone = false;
    for (const auto& arg : args) {
        const std::string a = wav::display(arg);
        if (optionsDone || a.empty() || a[0] != '-' || a == "-") {
            o.files.push_back(arg);
        } else if (a == "--") {
            optionsDone = true;
        } else if (a == "--stream") {
            o.stream = true;
        } else if (a == "-q" || a == "--quiet") {
            o.quiet = true;
        } else if (a == "--float") {
            o.sameFormat = false;
            o.format = wav::Format::Float32;
        } else if (startsWith(a, "--format=")) {
            const std::string f = a.substr(9);
            o.sameFormat = f == "same";
            if (f == "pcm16") o.format = wav::Format::Pcm16;
            else if (f == "pcm24") o.format = wav::Format::Pcm24;
            else if (f == "pcm32") o.format = wav::Format::Pcm32;
            else if (f == "float32" || f == "float") o.format = wav::Format::Float32;
            else if (f == "float64") o.format = wav::Format::Float64;
            else if (f != "same") throw UsageError("unknown --format '" + f + "'");
        } else if (startsWith(a, "--high-band=")) {
            const std::string m = a.substr(12);
            if (m == "drop") o.highBand = denoiser::HighBand::Drop;
            else if (m == "pass") o.highBand = denoiser::HighBand::PassThrough;
            else throw UsageError("unknown --high-band '" + m + "' (drop or pass)");
        } else {
            throw UsageError("unknown option " + a);
        }
    }
    if (o.files.size() != 3)
        throw UsageError("expected 3 file names (model.bin in.wav out.wav), got " + std::to_string(o.files.size()));
    return o;
}

std::shared_ptr<const denoiser::Model> loadModel(const std::filesystem::path& path) {
    // Read the bytes here (std::filesystem::path: non-ASCII names work on Windows too)
    const std::vector<unsigned char> bytes = wav::readFile(path);
    try {
        return denoiser::Model::fromMemory(bytes.data(), bytes.size());
    } catch (const denoiser::FormatError& e) {
        throw std::runtime_error(wav::display(path) + ": not a valid model file: " + e.what());
    } catch (const std::exception& e) {
        throw std::runtime_error(wav::display(path) + ": " + e.what());
    }
}

std::string supportedRates() {
    std::string s;
    for (int r : {16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000})
        s += (s.empty() ? "" : ", ") + std::to_string(r);
    return s;
}

int run(const Options& o) {
    using Clock = std::chrono::steady_clock;
    const auto model = loadModel(o.files[0]);
    const denoiser::ModelInfo& info = model->info();
    wav::Audio in = wav::read(o.files[1]);
    const std::string inName = wav::display(o.files[1]);
    const std::size_t frames = in.frames();
    const int channels = int(in.channels.size());
    if (!o.quiet) {
        std::fprintf(stderr, "model: %s\n", model->describe().c_str());
        std::fprintf(stderr, "input: %s: %d Hz, %d channel(s), %s, %zu samples (%.2f s)\n", inName.c_str(),
                     in.sampleRate, channels, wav::formatName(in.format), frames, double(frames) / in.sampleRate);
    }

    wav::Audio out;
    out.sampleRate = in.sampleRate;
    out.channels.resize(in.channels.size());
    const auto start = Clock::now();
    if (!o.stream) {
        if (in.sampleRate != info.sampleRate)
            throw std::runtime_error(inName + " is " + std::to_string(in.sampleRate) + " Hz; the model runs at " +
                                     std::to_string(info.sampleRate) +
                                     " Hz. Resample it first, or use --stream (resamples like the plugin).");
        const std::unique_ptr<denoiser::FrameNetwork> net = model->makeNetwork();
        for (int c = 0; c < channels; ++c) {
            const auto& x = in.channels[std::size_t(c)];
            try {
                out.channels[std::size_t(c)] = denoiser::denoiseClip(info, *net, x.data(), x.size());
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error(inName + ": " + e.what());
            }
        }
        if (!o.quiet)
            std::fprintf(stderr, "mode: offline (whole clip%s)\n",
                         info.framing == denoiser::Framing::Centred ? ", centred frames" : ", causal frames");
    } else {
        if (!info.realtime)
            throw std::runtime_error(wav::display(o.files[0]) +
                                     ": this model uses centred frames (realtime = false): it looks ahead and can't run "
                                     "in real time. Run it without --stream (offline).");
        if (!denoiser::isSupportedHostRate(in.sampleRate))
            throw std::runtime_error(inName + ": sample rate " + std::to_string(in.sampleRate) +
                                     " Hz is not supported for --stream (supported: " + supportedRates() + " Hz)");
        std::vector<std::unique_ptr<denoiser::ChannelDenoiser>> denoisers;
        for (int c = 0; c < channels; ++c)
            denoisers.push_back(std::make_unique<denoiser::ChannelDenoiser>(model));
        denoiser::HostRateAdapter adapter;
        adapter.prepare(denoisers[0]->hostRateConfig(in.sampleRate, channels, o.highBand));
        for (int c = 0; c < channels; ++c)
            adapter.setProcessor(c, denoisers[std::size_t(c)].get());
        for (int c = 0; c < channels; ++c) {
            auto& y = out.channels[std::size_t(c)];
            y.assign(frames, 0.0f);
            denoiser::processClipAligned(adapter, c, in.channels[std::size_t(c)].data(), y.data(),
                                         std::int64_t(frames));
        }
        if (!o.quiet)
            std::fprintf(stderr, "mode: stream at %d Hz (%s), plugin latency %d samples removed\n", in.sampleRate,
                         in.sampleRate == denoiser::kModelRate ? "no resampling"
                         : o.highBand == denoiser::HighBand::Drop ? "resampled, band above 8 kHz dropped"
                                                                  : "resampled, band above 8 kHz passed through",
                         adapter.latencySamples());
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();

    const wav::Format format = o.sameFormat ? in.format : o.format;
    std::size_t clipped = 0;
    wav::write(o.files[2], out, format, &clipped);
    if (!o.quiet) {
        const double audio = double(frames) / in.sampleRate;
        std::fprintf(stderr, "output: %s (%s)\n", wav::display(o.files[2]).c_str(), wav::formatName(format));
        std::fprintf(stderr, "time: %.3f s for %.2f s of audio x %d channel(s): %.1fx faster than real time\n", seconds,
                     audio, channels, seconds > 0 ? audio / seconds : 0.0);
    }
    if (clipped)
        std::fprintf(stderr, "warning: %zu sample(s) clipped to the range of %s\n", clipped, wav::formatName(format));
    return 0;
}

int mainWith(const std::vector<std::filesystem::path>& args) {
    if (args.size() == 1) {
        const std::string a = wav::display(args[0]);
        if (a == "--version" || a == "-V") {
            std::printf("denoise %s\n", denoiser::version());
            return 0;
        }
        if (a == "--help" || a == "-h") {
            std::printf("%s", kUsage);
            return 0;
        }
    }
    Options o;
    try {
        o = parse(args);
    } catch (const UsageError& e) {
        std::fprintf(stderr, "denoise: %s\n\n%s", e.what(), kUsage);
        return 2;
    }
    try {
        return run(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "denoise: error: %s\n", e.what());
        return 1;
    }
}

}  // namespace

#if defined(_WIN32) && defined(_MSC_VER)
// Wide arguments, so file names outside the ANSI code page work
int wmain(int argc, wchar_t** argv) {
    std::vector<std::filesystem::path> args;
    for (int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return mainWith(args);
}
#else
int main(int argc, char** argv) {
    std::vector<std::filesystem::path> args;
    for (int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return mainWith(args);
}
#endif
