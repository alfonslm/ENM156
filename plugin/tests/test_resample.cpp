// Resampler building blocks (denoiser/resample.h): which rates are
// supported, the filter's measured response (passband ripple, stopband
// attenuation) for every supported family of rates, and that each direction
// on its own delays by exactly kResampleHalfLength samples at 16 kHz.
#include "check.h"
#include "denoiser/fft.h"
#include "denoiser/resample.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

const double kPi = std::acos(-1.0);

double db(double x) { return 20.0 * std::log10(std::max(x, 1e-30)); }

void testRates() {
    using denoiser::isSupportedHostRate;
    for (int rate : {16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000})
        CHECK(isSupportedHostRate(rate));
    for (double rate : {8000.0, 11025.0, 15999.0, 47999.0, 44100.5, 400000.0, 0.0, -48000.0})
        CHECK(!isSupportedHostRate(rate));
    CHECK(!isSupportedHostRate(std::nan("")));

    auto r = denoiser::hostRateRatio(44100);
    CHECK(r.up == 160 && r.down == 441);
    r = denoiser::hostRateRatio(48000);
    CHECK(r.up == 1 && r.down == 3);
    r = denoiser::hostRateRatio(16000);
    CHECK(r.up == 1 && r.down == 1);

    bool threw = false;
    try {
        denoiser::ResampleFilter::create(8000);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

// The prototype's frequency response, from a zero-padded FFT
void testDesign(int rate) {
    auto f = denoiser::ResampleFilter::create(rate);
    const int M = f->down(), L = f->up();
    const int N = f->length();
    const auto& h = f->prototype();
    const double grid = double(M) * denoiser::kModelRate;
    CHECK(N == 2 * M * denoiser::kResampleHalfLength + 1);
    CHECK(int(h.size()) == N);
    CHECK(f->centre() == std::int64_t(M) * denoiser::kResampleHalfLength);

    double sum = 0.0, asym = 0.0;
    for (int k = 0; k < N; ++k) {
        sum += h[k];
        asym = std::max(asym, std::abs(h[k] - h[N - 1 - k]));
    }
    CHECK_NEAR(sum, 1.0, 1e-12);
    CHECK(asym < 1e-15);  // linear phase

    int size = 1 << 16;
    while (size < 16 * N || grid / size > 8.0)
        size *= 2;
    std::vector<float> padded(size, 0.0f);
    for (int k = 0; k < N; ++k)
        padded[k] = float(h[k]);
    denoiser::RealFft fft(size);
    std::vector<std::complex<float>> spec(fft.bins());
    fft.forward(padded.data(), spec.data());

    double passMax = 0.0, stopMax = 0.0, at7k5 = 0.0;
    for (int b = 0; b < fft.bins(); ++b) {
        const double hz = b * grid / size;
        const double mag = std::abs(spec[b]);
        if (hz <= denoiser::kResamplePassbandHz)
            passMax = std::max(passMax, std::abs(db(mag)));
        else if (hz >= denoiser::kResampleStopbandHz)
            stopMax = std::max(stopMax, mag);
        if (std::abs(hz - 7500.0) < grid / size / 2)
            at7k5 = mag;
    }
    // Each polyphase table, as stored (float, with gain L or M): DC gain of
    // every phase. Their spread is a ripple at the host / 16 kHz rate.
    double phaseDc = 0.0;
    for (int p = 0; p < L; ++p) {
        double s = 0.0;
        for (int i = 0; i < f->downTaps(); ++i)
            s += f->downPhase(p)[i];
        phaseDc = std::max(phaseDc, std::abs(s - 1.0));
    }
    for (int p = 0; p < M; ++p) {
        double s = 0.0;
        for (int i = 0; i < f->upTaps(); ++i)
            s += f->upPhase(p)[i];
        phaseDc = std::max(phaseDc, std::abs(s - 1.0));
    }
    std::printf("%6d Hz: L %3d M %3d, %5d taps (down %3d x %3d phases, up %3d x %3d), "
                "passband ripple +-%.2g dB, stopband %.1f dB, %.2f dB at 7.5 kHz, phase DC error %.1g\n",
                rate, L, M, N, f->downTaps(), L, f->upTaps(), M, passMax, -db(stopMax), db(at7k5), phaseDc);
    CHECK(passMax < 0.001);    // dB
    CHECK(db(stopMax) < -90);  // dB
    CHECK(phaseDc < 1e-4);
}

// Each direction alone: a sine comes out delayed by exactly Q at 16 kHz
void testDirections(int rate) {
    auto f = denoiser::ResampleFilter::create(rate);
    const int Q = denoiser::kResampleHalfLength;
    const double hz = 1234.5;
    const int seconds16 = 16000 / 2;  // half a second at 16 kHz

    // Down
    denoiser::Downsampler down;
    down.prepare(f);
    std::vector<float> y;
    for (int m = 0; int(y.size()) < seconds16; ++m) {
        float v;
        if (down.push(float(std::sin(2.0 * kPi * hz * m / rate)), &v))
            y.push_back(v);
    }
    double errDown = 0.0;
    for (int n = 4 * Q; n < int(y.size()); ++n)
        errDown = std::max(errDown, std::abs(y[n] - std::sin(2.0 * kPi * hz * (n - Q) / 16000.0)));

    // Up, with offset 0
    denoiser::Upsampler up;
    up.prepare(f, 0);
    std::vector<float> z;
    int n = 0;
    while (int(z.size()) < rate / 2) {
        while (!up.ready())
            up.push(float(std::sin(2.0 * kPi * hz * n++ / 16000.0)));
        z.push_back(up.pull());
    }
    double errUp = 0.0;
    const int skip = 4 * Q * rate / 16000;
    for (int m = skip; m < int(z.size()); ++m)
        errUp = std::max(errUp, std::abs(z[m] - std::sin(2.0 * kPi * hz * (double(m) / rate - double(Q) / 16000.0))));
    std::printf("%6d Hz: 1234.5 Hz sine, max error vs. exact delay of %d samples at 16 kHz: down %.2g, up %.2g\n",
                rate, Q, errDown, errUp);
    CHECK(errDown < 1e-4);
    CHECK(errUp < 1e-4);
    CHECK(up.underruns() == 0);
}

}  // namespace

int main() {
    testRates();
    for (int rate : {22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000, 384000})
        testDesign(rate);
    for (int rate : {44100, 48000, 96000})
        testDirections(rate);

    // 16 kHz: both directions copy samples unchanged, with no delay
    auto f = denoiser::ResampleFilter::create(16000);
    CHECK(f->bypass());
    denoiser::Downsampler down;
    denoiser::Upsampler up;
    down.prepare(f);
    up.prepare(f, 0);
    for (float x : {0.25f, -0.0f, 1e-30f, -3.0f}) {
        float y = 7.0f;
        CHECK(down.push(x, &y));
        up.push(y);
        CHECK(up.ready());
        const float z = up.pull();
        CHECK(std::signbit(z) == std::signbit(x) && z == x);
    }
    return check::finish();
}
