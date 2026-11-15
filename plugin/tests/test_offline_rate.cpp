// Whole clips at other rates (denoiser/offline_rate.h), for the ARA analysis:
//   - modelRateLength
//   - 16 kHz: both directions copy, denoiseClipAtRate == denoiseClip bit for bit
//   - alignment: a band-limited signal (sines up to 6 kHz) taken to 16 kHz
//     equals the same signal computed at 16 kHz, and back again equals the
//     input, away from the clip edges (-80 dB; one sample off is -7 to -20 dB)
//   - denoiseClipAtRate == the parts composed by hand; PassThrough - Drop ==
//     input - its round trip
//   (Not compared with the real-time path: that one frames the 16 kHz signal
//   on a grid shifted by the resampler's delay, so its STFT frames fall
//   elsewhere and the result differs, as between any two frame grids. This
//   path frames the clip from its first sample, like the CLI on a 16 kHz file.)
//   - errors: unsupported rates, a clip too short for a centred model
#include "check.h"
#include "golden.h"

#include "denoiser/denoiser.h"
#include "denoiser/offline.h"
#include "denoiser/offline_rate.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

using namespace denoiser;

namespace {

const double kPi = std::acos(-1.0);

std::shared_ptr<const Model> loadModel(const std::string& name) {
    const std::vector<char> bytes = golden::readBytes(golden::dir() + "/" + name + ".model.bin");
    return Model::fromMemory(bytes.data(), bytes.size());
}

// Sum of sines below 6 kHz, evaluated at time i / rate
std::vector<float> sines(int rate, std::size_t n) {
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = double(i) / rate;
        double v = 0.0;
        for (int k = 1; k <= 12; ++k)
            v += 0.05 * std::sin(2.0 * kPi * (480.0 * k + 13.0) * t + 0.7 * k);
        x[i] = float(v);
    }
    return x;
}

double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b, std::size_t from, std::size_t to) {
    double d = 0.0;
    for (std::size_t i = from; i < to; ++i)
        d = std::max(d, std::fabs(double(a[i]) - double(b[i])));
    return d;
}

double db(double x) { return 20.0 * std::log10(std::max(x, 1e-30)); }

void testLengths() {
    CHECK(modelRateLength(48000, 0) == 0);
    CHECK(modelRateLength(48000, 3) == 1);
    CHECK(modelRateLength(48000, 4) == 2);
    CHECK(modelRateLength(48000, 48000) == 16000);
    CHECK(modelRateLength(44100, 44100) == 16000);
    CHECK(modelRateLength(44100, 1) == 1);
    CHECK(modelRateLength(16000, 1234) == 1234);
    bool threw = false;
    try {
        modelRateLength(8000, 10);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void testModelRate() {
    auto f = ResampleFilter::create(16000);
    const std::vector<float> x = sines(16000, 1000);
    CHECK(clipToModelRate(*f, x.data(), x.size()) == x);
    CHECK(clipFromModelRate(*f, x.data(), x.size(), x.size()) == x);
    auto model = loadModel("mlp_magnitude_default");
    auto net = model->makeNetwork();
    const std::vector<float> a = denoiseClip(model->info(), *net, x.data(), x.size());
    const std::vector<float> b = denoiseClipAtRate(model->info(), *net, 16000, x.data(), x.size());
    CHECK(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

void testAlignment(int rate) {
    auto f = ResampleFilter::create(rate);
    const std::size_t n = std::size_t(rate) / 2;  // 0.5 s
    const std::vector<float> x = sines(rate, n);
    const std::vector<float> x16 = clipToModelRate(*f, x.data(), n);
    const std::vector<float> ref16 = sines(16000, x16.size());
    CHECK(x16.size() == modelRateLength(rate, n));
    // Away from the edges (the filter spans 48 samples at 16 kHz each way)
    const double down = maxAbsDiff(x16, ref16, 200, x16.size() - 200);
    const std::vector<float> back = clipFromModelRate(*f, x16.data(), x16.size(), n);
    CHECK(back.size() == n);
    const std::size_t edge = std::size_t(400.0 * rate / 16000.0);
    const double round = maxAbsDiff(back, x, edge, n - edge);
    // One sample late at the host rate, for comparison
    std::vector<float> late(n, 0.0f);
    for (std::size_t i = 1; i < n; ++i)
        late[i] = x[i - 1];
    const double shifted = maxAbsDiff(back, late, edge, n - edge);
    std::printf("  %d Hz: to 16 kHz %.1f dB, round trip %.1f dB (one sample off: %.1f dB)\n", rate, db(down),
                db(round), db(shifted));
    CHECK(db(down) < -80.0);
    CHECK(db(round) < -80.0);
    CHECK(db(shifted) > -40.0);
}

void testComposition(int rate) {
    auto model = loadModel("mlp_mask_default");
    auto net = model->makeNetwork();
    auto f = ResampleFilter::create(rate);
    const std::size_t n = std::size_t(rate) / 4 + 7;
    std::vector<float> x = sines(rate, n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] += 0.1f * float(std::sin(0.37 * double(i)) * std::cos(0.0011 * double(i * i % 9973)));
    const std::vector<float> x16 = clipToModelRate(*f, x.data(), n);
    const std::vector<float> y16 = denoiseClip(model->info(), *net, x16.data(), x16.size());
    const std::vector<float> manual = clipFromModelRate(*f, y16.data(), y16.size(), n);
    const std::vector<float> drop = denoiseClipAtRate(model->info(), *net, rate, x.data(), n, HighBand::Drop);
    CHECK(drop.size() == n && std::memcmp(drop.data(), manual.data(), n * sizeof(float)) == 0);
    const std::vector<float> pass = denoiseClipAtRate(model->info(), *net, rate, x.data(), n, HighBand::PassThrough);
    const std::vector<float> round = clipFromModelRate(*f, x16.data(), x16.size(), n);
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        worst = std::max(worst, std::fabs(double(pass[i] - drop[i]) - double(x[i] - round[i])));
    CHECK(worst < 1e-6);
}

void testErrors() {
    auto centred = loadModel("offline_mlp_mask_centred");
    auto net = centred->makeNetwork();
    const std::size_t min16 = minClipSamples(centred->info());
    // Just too short at 48 kHz, and just long enough
    std::vector<float> x(3 * (min16 - 1), 0.1f);
    bool threw = false;
    try {
        denoiseClipAtRate(centred->info(), *net, 48000, x.data(), x.size());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    x.assign(3 * min16, 0.1f);
    CHECK(denoiseClipAtRate(centred->info(), *net, 48000, x.data(), x.size()).size() == x.size());
    threw = false;
    try {
        denoiseClipAtRate(centred->info(), *net, 11025, x.data(), x.size());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    // Empty clips
    auto f = ResampleFilter::create(44100);
    CHECK(clipToModelRate(*f, x.data(), 0).empty());
    CHECK(clipFromModelRate(*f, x.data(), 0, 5) == std::vector<float>(5, 0.0f));
    // Cancelled: empty result
    auto mlp = loadModel("mlp_magnitude_default");
    auto mlpNet = mlp->makeNetwork();
    const auto cancelled = denoiseClipAtRate(mlp->info(), *mlpNet, 48000, x.data(), x.size(), HighBand::Drop,
                                             [](std::size_t done, std::size_t) { return done < 3; });
    CHECK(cancelled.empty());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1)
        golden::dir() = argv[1];
    try {
        testLengths();
        testModelRate();
        for (int rate : {48000, 44100, 22050, 96000})
            testAlignment(rate);
        testComposition(48000);
        testComposition(44100);
        testErrors();
    } catch (const std::exception& e) {
        std::printf("exception: %s\n", e.what());
        ++check::failures();
    }
    return check::finish();
}
