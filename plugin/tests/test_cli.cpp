// The command-line tool, run as a program, on WAV files made from the golden
// inputs, plus its WAV reader/writer on their own.
//
//   test_cli <denoise executable> <golden dir> <work dir>
//
// For every golden case: the input as a 32-bit float WAV (stereo:
// the same clip twice) through `denoise` (offline) and, for real-time models,
// `denoise --stream`; each written channel vs golden offline (1e-5 of
// max(1, max |golden|)). As 16-bit PCM in and out: within 1 LSB of the
// in-process result for the quantised input. Error cases: exit status 2 for
// bad arguments, 1 for a missing or malformed file, a wrong or unsupported
// sample rate, or a centred model with --stream.
// Set DENOISE_LAUNCHER (e.g. "qemu-aarch64 -L /usr/aarch64-linux-gnu") to run
// the tool through an emulator.
#include "check.h"
#include "golden.h"

#include "../cli/wav.h"

#include "denoiser/denoiser.h"
#include "denoiser/offline.h"

#include <cstdlib>
#include <filesystem>
#include <string>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

using namespace denoiser;
namespace fs = std::filesystem;

namespace {

constexpr double kTol = 1e-5;

std::string exe;
fs::path work;

std::string quote(const std::string& s) { return "\"" + s + "\""; }

// Runs denoise with the arguments; returns its exit status
int denoise(const std::string& args) {
    std::string cmd;
    if (const char* launcher = std::getenv("DENOISE_LAUNCHER"))
        cmd = std::string(launcher) + " ";
    cmd += quote(exe) + " " + args;
#if defined(_WIN32)
    cmd = "\"" + cmd + " >NUL 2>&1\"";  // cmd.exe strips one outer pair of quotes
    return std::system(cmd.c_str());
#else
    cmd += " >/dev/null 2>&1";
    const int rc = std::system(cmd.c_str());
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
#endif
}

std::string path(const fs::path& p) { return quote(wav::display(p)); }

void writeWav(const fs::path& p, int rate, const std::vector<std::vector<float>>& channels, wav::Format f) {
    wav::Audio a;
    a.sampleRate = rate;
    a.channels = channels;
    wav::write(p, a, f);
}

// ---- WAV on its own ---------------------------------------------------------------

void checkWav() {
    const std::vector<float> ramp = {0.0f, 0.5f, -0.5f, 0.99f, -1.0f, 0.25f, -0.125f, 1e-6f, 0.3333f};
    wav::Audio a;
    a.sampleRate = 44100;
    a.channels = {ramp, std::vector<float>(ramp.rbegin(), ramp.rend()), std::vector<float>(ramp.size(), 0.1f)};
    for (wav::Format f : {wav::Format::Pcm8, wav::Format::Pcm16, wav::Format::Pcm24, wav::Format::Pcm32,
                          wav::Format::Float32, wav::Format::Float64}) {
        std::size_t clipped = 99;
        const wav::Audio b = wav::decode(wav::encode(a, f, &clipped), "mem");
        CHECK(clipped == 0);
        CHECK(b.sampleRate == 44100 && b.format == f && b.channels.size() == 3 && b.frames() == ramp.size());
        const double lsb = f == wav::Format::Pcm8 ? 1.0 / 128 : f == wav::Format::Pcm16 ? 1.0 / 32768
                         : f == wav::Format::Pcm24 ? 1.0 / 8388608 : f == wav::Format::Pcm32 ? 1.0 / 2147483648.0 : 0.0;
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t t = 0; t < ramp.size(); ++t)
                CHECK_NEAR(b.channels[c][t], a.channels[c][t], lsb / 2 + 1e-9);
    }
    // clipping
    wav::Audio loud;
    loud.sampleRate = 16000;
    loud.channels = {{1.0f, -1.0f, 2.0f, -3.0f, 0.0f}};
    std::size_t clipped = 0;
    const wav::Audio q = wav::decode(wav::encode(loud, wav::Format::Pcm16, &clipped), "mem");
    CHECK(clipped == 3);  // 1.0 is just above 32767 / 32768; -1.0 fits
    CHECK(q.channels[0][0] == 32767.0f / 32768.0f && q.channels[0][1] == -1.0f && q.channels[0][3] == -1.0f);

    // a hand-made WAVE_FORMAT_EXTENSIBLE 24-bit file with an odd-sized chunk before fmt
    std::vector<unsigned char> x = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'j', 'u', 'n', 'k', 3, 0, 0, 0, 1, 2, 3, 0,
                                    'f', 'm', 't', ' ', 40, 0, 0, 0, 0xFE, 0xFF, 1, 0, 0x80, 0x3E, 0, 0, 0x80, 0xBB, 0, 0,
                                    3, 0, 24, 0, 22, 0, 24, 0, 4, 0, 0, 0, 1, 0, 0, 0, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00,
                                    0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71, 'd', 'a', 't', 'a', 6, 0, 0, 0, 0, 0, 0x40, 0, 0, 0xC0};
    const wav::Audio e = wav::decode(x, "ext");
    CHECK(e.sampleRate == 16000 && e.format == wav::Format::Pcm24 && e.frames() == 2);
    CHECK(e.channels[0][0] == 0.5f && e.channels[0][1] == -0.5f);

    // malformed input: errors, never a crash
    for (const std::vector<unsigned char>& bad : {std::vector<unsigned char>{}, std::vector<unsigned char>(x.begin(), x.begin() + 30),
                                                 std::vector<unsigned char>(x.begin(), x.begin() + 68)}) {
        bool threw = false;
        try {
            wav::decode(bad, "bad");
        } catch (const wav::Error&) {
            threw = true;
        }
        CHECK(threw);
    }
    std::vector<unsigned char> alaw = x;
    alaw[32] = 6;  // format code A-law, not extensible
    alaw[33] = 0;
    bool threw = false;
    try {
        wav::decode(alaw, "alaw");
    } catch (const wav::Error& err) {
        threw = std::string(err.what()).find("unsupported") != std::string::npos;
    }
    CHECK(threw);
}

// ---- the tool on golden cases ------------------------------------------------------

double channelErr(const wav::Audio& out, std::size_t c, const Tensor& ref) {
    if (out.channels.size() <= c || out.frames() != ref.size())
        return 1e9;
    return golden::relErr(out.channels[c].data(), ref.data, ref.size());
}

void checkCase(const std::string& name) {
    const golden::Case c = golden::load(name);
    const ModelInfo& info = c.info();
    const Tensor& in = c.get("input");
    const Tensor& offline = c.get("offline");
    const std::vector<float> clip(in.data, in.data + in.size());
    const fs::path model = fs::path(golden::dir()) / (name + ".model.bin");
    const fs::path inWav = work / (name + ".in.wav"), outWav = work / (name + ".out.wav");
    writeWav(inWav, 16000, {clip, clip}, wav::Format::Float32);

    std::printf("%-28s", name.c_str());
    CHECK(denoise("-q " + path(model) + " " + path(inWav) + " " + path(outWav)) == 0);
    const wav::Audio out = wav::read(outWav);
    CHECK(out.format == wav::Format::Float32 && out.sampleRate == 16000);
    const double err = std::max(channelErr(out, 0, offline), channelErr(out, 1, offline));
    CHECK(err <= kTol);
    std::printf(" offline (float WAV) %.1e", err);

    if (info.realtime) {
        CHECK(denoise("-q --stream " + path(model) + " " + path(inWav) + " " + path(outWav)) == 0);
        const wav::Audio s = wav::read(outWav);
        const double serr = std::max(channelErr(s, 0, offline), channelErr(s, 1, offline));
        CHECK(serr <= kTol);
        std::printf(", --stream %.1e", serr);
    }

    // 16-bit in, 16-bit out (the default: same format as the input)
    const fs::path in16 = work / (name + ".in16.wav"), out16 = work / (name + ".out16.wav");
    writeWav(in16, 16000, {clip}, wav::Format::Pcm16);
    const wav::Audio q = wav::read(in16);
    CHECK(denoise("-q " + path(model) + " " + path(in16) + " " + path(out16)) == 0);
    const wav::Audio o16 = wav::read(out16);
    CHECK(o16.format == wav::Format::Pcm16);
    const auto m = Model::load(model.string());
    auto net = m->makeNetwork();
    const std::vector<float> ref = denoiseClip(info, *net, q.channels[0].data(), q.frames());
    double worst = 0.0;
    bool sameLength = o16.frames() == ref.size();
    CHECK(sameLength);
    for (std::size_t i = 0; sameLength && i < ref.size(); ++i) {
        const double r = std::max(-1.0, std::min(32767.0 / 32768.0, double(ref[i])));
        worst = std::max(worst, std::fabs(double(o16.channels[0][i]) - r) * 32768.0);
    }
    CHECK(worst <= 0.5001 + 1e-3);  // rounding to the nearest step (+ float noise)
    std::printf(", 16-bit within %.2f LSB\n", worst);
}

void checkErrors() {
    const fs::path mlp = fs::path(golden::dir()) / "mlp_mask_default.model.bin";
    const fs::path centred = fs::path(golden::dir()) / "offline_mlp_mask_centred.model.bin";
    const fs::path in = work / "mlp_mask_default.in.wav", out = work / "err.out.wav";
    std::vector<float> tone(4800);
    for (std::size_t i = 0; i < tone.size(); ++i)
        tone[i] = 0.3f * float(std::sin(0.05 * double(i)));
    const fs::path in48 = work / "tone48k.wav", in8 = work / "tone8k.wav";
    writeWav(in48, 48000, {tone, tone}, wav::Format::Pcm24);
    writeWav(in8, 8000, {tone}, wav::Format::Pcm16);
    // a truncated model and a text file posing as a WAV
    const std::vector<unsigned char> bytes = wav::readFile(mlp);
    const fs::path truncated = work / "truncated.model.bin", notWav = work / "not.wav";
    wav::writeFile(truncated, std::vector<unsigned char>(bytes.begin(), bytes.begin() + std::ptrdiff_t(bytes.size() / 2)));
    wav::writeFile(notWav, std::vector<unsigned char>{'h', 'e', 'l', 'l', 'o'});

    struct Run {
        std::string args;
        int status;
    };
    const Run runs[] = {
        {"", 2},
        {path(mlp) + " " + path(in), 2},
        {"--bogus " + path(mlp) + " " + path(in) + " " + path(out), 2},
        {"--format=mp3 " + path(mlp) + " " + path(in) + " " + path(out), 2},
        {path(work / "missing.bin") + " " + path(in) + " " + path(out), 1},
        {path(mlp) + " " + path(work / "missing.wav") + " " + path(out), 1},
        {path(truncated) + " " + path(in) + " " + path(out), 1},
        {path(mlp) + " " + path(notWav) + " " + path(out), 1},
        {path(mlp) + " " + path(in48) + " " + path(out), 1},             // not the model's rate, offline
        {"--stream " + path(mlp) + " " + path(in8) + " " + path(out), 1},  // below 16 kHz
        {"--stream " + path(centred) + " " + path(in) + " " + path(out), 1},
        {path(mlp) + " " + path(in) + " " + path(work / "no_such_dir" / "x.wav"), 1},
        {"--help", 0},
        {"--version", 0},
    };
    for (const Run& r : runs) {
        const int status = denoise(r.args);
        if (status != r.status)
            std::printf("denoise %s: exit status %d, expected %d\n", r.args.c_str(), status, r.status);
        CHECK(status == r.status);
    }
    // --stream at 48 kHz, stereo 24-bit: same length and format, and finite
    for (const char* band : {"drop", "pass"}) {
        CHECK(denoise("-q --stream --high-band=" + std::string(band) + " " + path(mlp) + " " + path(in48) + " " + path(out)) == 0);
        const wav::Audio a = wav::read(out);
        CHECK(a.sampleRate == 48000 && a.channels.size() == 2 && a.frames() == tone.size() && a.format == wav::Format::Pcm24);
    }
    std::printf("error cases and --stream at 48 kHz: done\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: test_cli <denoise executable> <golden dir> <work dir>\n");
        return 2;
    }
    exe = argv[1];
    golden::dir() = argv[2];
    work = argv[3];
    fs::create_directories(work);
    checkWav();
    int cases = 0;
    for (const std::string& name : golden::caseNames()) {
        checkCase(name);
        ++cases;
    }
    CHECK(cases == 17);
    checkErrors();
    return check::finish();
}
