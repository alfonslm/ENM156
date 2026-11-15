// RealFft: the sizes the default model uses (n_fft 128, long window 1280 =
// 2^8 * 5, not a power of two) and an odd-factor one. Checks the forward
// transform against a direct DFT in double precision, and that inverse()
// undoes forward().
#include "check.h"
#include "denoiser/fft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void testSize(int n) {
    std::printf("size %d\n", n);
    std::mt19937 rng(1234 + n);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> x(n);
    for (float& v : x)
        v = uniform(rng);

    denoiser::RealFft fft(n);
    CHECK(fft.size() == n);
    CHECK(fft.bins() == n / 2 + 1);

    std::vector<std::complex<float>> spec(fft.bins());
    fft.forward(x.data(), spec.data());

    // Direct DFT, double precision
    const double pi = std::acos(-1.0);
    double maxErr = 0.0;
    for (int k = 0; k < fft.bins(); ++k) {
        std::complex<double> sum = 0.0;
        for (int t = 0; t < n; ++t)
            sum += double(x[t]) * std::polar(1.0, -2.0 * pi * double(k) * double(t) / double(n));
        maxErr = std::max(maxErr, std::abs(sum - std::complex<double>(spec[k])));
    }
    // Values are O(sqrt(n)); float rounding gives errors well below 1e-3
    std::printf("  forward vs direct DFT: max abs error %.3g\n", maxErr);
    CHECK(maxErr < 1e-3);

    std::vector<float> back(n, 0.0f);
    fft.inverse(spec.data(), back.data());
    double maxRoundTrip = 0.0;
    for (int t = 0; t < n; ++t)
        maxRoundTrip = std::max(maxRoundTrip, double(std::abs(back[t] - x[t])));
    std::printf("  inverse(forward(x)) - x: max abs error %.3g\n", maxRoundTrip);
    CHECK(maxRoundTrip < 1e-5);
}

}  // namespace

int main() {
    testSize(1280);  // long window with the defaults in config.toml
    testSize(128);   // n_fft with the defaults
    testSize(160);   // 2^5 * 5: another non-power-of-two
    testSize(2);

    bool threw = false;
    try {
        denoiser::RealFft odd(129);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // Moving keeps the plan usable
    denoiser::RealFft a(1280);
    denoiser::RealFft b = std::move(a);
    CHECK(b.size() == 1280);
    CHECK(a.size() == 0);

    return check::finish();
}
