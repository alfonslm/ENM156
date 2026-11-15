#include "denoiser/resample.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace denoiser {

namespace {

constexpr int kMinHostRate = kModelRate;
constexpr int kMaxHostRate = 384000;
constexpr int kMaxDown = 1000;  // most phases an up table may have (44.1 kHz family: 441)

// Modified Bessel function of the first kind, order 0 (power series)
double besselI0(double x) {
    const double q = 0.25 * x * x;
    double term = 1.0, sum = 1.0;
    for (int k = 1; k < 500; ++k) {
        term *= q / (double(k) * double(k));
        sum += term;
        if (term < 1e-17 * sum)
            break;
    }
    return sum;
}

// Sum of a[i] * b[i]. Four partial sums in a fixed order: fast, and the
// result only depends on the inputs (not on how audio was split into blocks).
inline float dot(const float* a, const float* b, int n) noexcept {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i)
        s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

}  // namespace

bool isSupportedHostRate(double hostRate) noexcept {
    if (!(hostRate >= kMinHostRate && hostRate <= kMaxHostRate))
        return false;
    if (hostRate != std::floor(hostRate))
        return false;
    const int rate = static_cast<int>(hostRate);
    return rate / std::gcd(rate, kModelRate) <= kMaxDown;
}

RateRatio hostRateRatio(int hostRate) {
    if (!isSupportedHostRate(hostRate))
        throw std::invalid_argument("unsupported host sample rate " + std::to_string(hostRate) +
                                    " Hz (supported: 16 to 384 kHz, e.g. 44.1, 48, 88.2, 96 kHz)");
    const int g = std::gcd(hostRate, kModelRate);
    RateRatio r;
    r.up = kModelRate / g;
    r.down = hostRate / g;
    return r;
}

std::shared_ptr<const ResampleFilter> ResampleFilter::create(int hostRate) {
    std::shared_ptr<ResampleFilter> f(new ResampleFilter());
    f->ratio_ = hostRateRatio(hostRate);  // throws if unsupported
    f->hostRate_ = hostRate;
    if (f->bypass())
        return f;

    const int L = f->ratio_.up;
    const int M = f->ratio_.down;
    const int N = 2 * M * kResampleHalfLength + 1;
    const double grid = double(M) * kModelRate;  // F, Hz
    const double pi = std::acos(-1.0);
    const int c = M * kResampleHalfLength;  // centre

    // Kaiser's estimate of the attenuation this length reaches over the
    // transition band, and the window shape for it
    const double transition = 2.0 * pi * (kResampleStopbandHz - kResamplePassbandHz) / grid;
    const double atten = 2.285 * transition * double(N - 1) + 7.95;
    const double beta = atten > 50.0 ? 0.1102 * (atten - 8.7)
                                     : 0.5842 * std::pow(atten - 21.0, 0.4) + 0.07886 * (atten - 21.0);
    const double fc = kResampleCutoffHz / grid;  // cycles per grid step
    const double i0beta = besselI0(beta);

    f->length_ = N;
    f->prototype_.resize(N);
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        const double t = double(k - c);
        const double x = 2.0 * pi * fc * t;
        const double sinc = (k == c) ? 1.0 : std::sin(x) / x;
        const double r = t / double(c);
        const double w = besselI0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0beta;
        f->prototype_[k] = 2.0 * fc * sinc * w;
        sum += f->prototype_[k];
    }
    for (double& h : f->prototype_)
        h /= sum;  // DC gain 1 on the grid

    auto tap = [&](int k) { return (k >= 0 && k < N) ? f->prototype_[k] : 0.0; };

    // Down: output at grid time t, phase p = t mod L, newest input m_hi =
    // t div L. Input m_hi - j meets tap p + j L. Stored oldest first, with
    // gain L (zero-stuffing by L divides the level by L).
    f->downTaps_ = (N + L - 1) / L;
    f->downTable_.resize(std::size_t(L) * f->downTaps_);
    for (int p = 0; p < L; ++p)
        for (int i = 0; i < f->downTaps_; ++i)
            f->downTable_[std::size_t(p) * f->downTaps_ + i] =
                float(double(L) * tap(p + (f->downTaps_ - 1 - i) * L));

    // Up: the same with the roles of L and M swapped, gain M
    f->upTaps_ = (N + M - 1) / M;
    f->upTable_.resize(std::size_t(M) * f->upTaps_);
    for (int p = 0; p < M; ++p)
        for (int i = 0; i < f->upTaps_; ++i)
            f->upTable_[std::size_t(p) * f->upTaps_ + i] =
                float(double(M) * tap(p + (f->upTaps_ - 1 - i) * M));
    return f;
}

// --- Downsampler ---

void Downsampler::prepare(std::shared_ptr<const ResampleFilter> filter) {
    if (!filter)
        throw std::invalid_argument("Downsampler: no filter");
    filter_ = std::move(filter);
    history_.assign(filter_->bypass() ? 0 : 2 * std::size_t(filter_->downTaps()), 0.0f);
    reset();
}

void Downsampler::reset() noexcept {
    std::fill(history_.begin(), history_.end(), 0.0f);
    write_ = 0;
    pushed_ = 0;
    nextTime_ = 0;
}

bool Downsampler::push(float in, float* out) noexcept {
    const ResampleFilter& f = *filter_;
    if (f.bypass()) {
        *out = in;
        return true;
    }
    const int K = f.downTaps();
    history_[write_] = in;
    history_[write_ + K] = in;
    if (++write_ == K)
        write_ = 0;
    ++pushed_;

    const int L = f.up();
    // Since hostRate >= 16 kHz, consecutive outputs need different newest
    // inputs, so when one is ready its newest input is the one just pushed.
    if (nextTime_ / L > pushed_ - 1)
        return false;
    const int phase = int(nextTime_ % L);
    *out = dot(history_.data() + write_, f.downPhase(phase), K);
    nextTime_ += f.down();
    return true;
}

// --- Upsampler ---

void Upsampler::prepare(std::shared_ptr<const ResampleFilter> filter, int offset) {
    if (!filter)
        throw std::invalid_argument("Upsampler: no filter");
    if (offset < 0 || offset >= filter->up())
        throw std::invalid_argument("Upsampler: offset must be in [0, L)");
    filter_ = std::move(filter);
    offset_ = offset;
    // The newest sample needed lags the newest pushed by at most one
    capacity_ = filter_->bypass() ? 2 : filter_->upTaps() + 2;
    history_.assign(2 * std::size_t(capacity_), 0.0f);
    reset();
}

void Upsampler::reset() noexcept {
    std::fill(history_.begin(), history_.end(), 0.0f);
    write_ = 0;
    pushed_ = 0;
    nextTime_ = offset_;
    underruns_ = 0;
}

void Upsampler::push(float in) noexcept {
    history_[write_] = in;
    history_[write_ + capacity_] = in;
    if (++write_ == capacity_)
        write_ = 0;
    ++pushed_;
}

bool Upsampler::ready() const noexcept {
    return nextTime_ / filter_->down() <= pushed_ - 1;
}

float Upsampler::pull() noexcept {
    const ResampleFilter& f = *filter_;
    const int M = f.down();
    const std::int64_t newest = nextTime_ / M;
    const std::int64_t lag = (pushed_ - 1) - newest;  // samples pushed beyond the newest needed
    const int K = f.bypass() ? 1 : f.upTaps();
    const int phase = int(nextTime_ % M);
    nextTime_ += f.up();
    if (lag < 0 || lag > capacity_ - K) {
        ++underruns_;
        return 0.0f;
    }
    // Slot of the oldest sample of the K-sample window ending at `newest`
    int start = write_ - int(lag) - K;
    if (start < 0)
        start += capacity_;
    if (f.bypass())
        return history_[start];  // bit-exact copy
    return dot(history_.data() + start, f.upPhase(phase), K);
}

}  // namespace denoiser
