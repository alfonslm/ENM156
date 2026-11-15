// HostRateAdapter (denoiser/host_rate.h): the reported latency is exact for
// every supported rate and every way of splitting the audio into blocks;
// output does not depend on block sizes; round-trip accuracy below 7 kHz and
// attenuation above 8 kHz at 44.1 and 48 kHz; 16 kHz is a bit-exact bypass;
// reset() reproduces the first run; no allocation while processing; stereo
// channels are independent; the high band pass-through (option B). Also
// prints the cost per second of 48 kHz stereo audio.
#include "check.h"
#include "denoiser/host_rate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <stdexcept>
#include <vector>

// --- Allocation counting: every operator new in this program goes through here
static long long gAllocations = 0;

void* operator new(std::size_t size) {
    ++gAllocations;
    if (void* p = std::malloc(size ? size : 1))
        return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

using denoiser::HighBand;
using denoiser::HopProcessor;
using denoiser::HostRateAdapter;
using denoiser::HostRateConfig;

const double kPi = std::acos(-1.0);
const int kAllRates[] = {16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000};

double db(double x) { return 20.0 * std::log10(std::max(x, 1e-30)); }

// --- Stand-ins for the 16 kHz denoiser

// out = gain * in delayed by `delay` samples (delay 0 + gain 1 = identity).
// Carries state across hops like the real one, and resets it.
class DelayGain : public HopProcessor {
public:
    DelayGain(int delay, float gain) : line_(std::size_t(delay) + 1, 0.0f), gain_(gain) {}
    void processHop(const float* in, float* out, int hop) noexcept override {
        ++calls;
        for (int i = 0; i < hop; ++i) {
            line_[pos_] = in[i];
            pos_ = (pos_ + 1) % int(line_.size());
            out[i] = gain_ * line_[pos_];  // the oldest: in delayed by size - 1
        }
    }
    void reset() noexcept override {
        std::fill(line_.begin(), line_.end(), 0.0f);
        pos_ = 0;
        ++resets;
    }
    long long calls = 0, resets = 0;

private:
    std::vector<float> line_;
    int pos_ = 0;
    float gain_;
};

// Stateful and nonlinear, so a wrong regrouping or reset shows: a one-pole
// lowpass whose coefficient depends on the hop's energy
class Weird : public HopProcessor {
public:
    void processHop(const float* in, float* out, int hop) noexcept override {
        float e = 0.0f;
        for (int i = 0; i < hop; ++i)
            e += in[i] * in[i];
        const float a = 0.5f + 0.4f * std::tanh(e);
        for (int i = 0; i < hop; ++i) {
            s_ = a * s_ + (1.0f - a) * in[i];
            out[i] = s_ + 0.1f * in[i];
        }
    }
    void reset() noexcept override { s_ = 0.0f; }

private:
    float s_ = 0.0f;
};

// --- Block size patterns

std::vector<int> pattern(int kind, std::size_t total, unsigned seed) {
    std::vector<int> blocks;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> any(0, 700);  // includes 0-sample calls
    std::size_t done = 0;
    while (done < total) {
        int b = kind > 0 ? kind : any(rng);
        b = int(std::min<std::size_t>(std::size_t(b), total - done));
        blocks.push_back(b);
        done += std::size_t(b);
    }
    return blocks;
}

const int kBlockKinds[] = {1, 7, 64, 80, 333, 512, 4096, 0 /* random */};

std::vector<float> run(HostRateAdapter& a, int channel, const std::vector<float>& x, const std::vector<int>& blocks,
                       bool inPlace = false) {
    std::vector<float> y(x.size());
    if (inPlace)
        y = x;
    std::size_t at = 0;
    for (int b : blocks) {
        a.processChannel(channel, inPlace ? y.data() + at : x.data() + at, y.data() + at, b);
        at += std::size_t(b);
    }
    return y;
}

// Sum of random sines from 50 Hz to `topHz`, peak about 0.5
std::vector<float> sines(int rate, std::size_t n, double topHz, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> freq(50.0, topHz), phase(0.0, 2.0 * kPi);
    std::vector<double> f(40), p(40);
    for (int k = 0; k < 40; ++k) {
        f[k] = freq(rng);
        p[k] = phase(rng);
    }
    std::vector<float> x(n);
    for (std::size_t m = 0; m < n; ++m) {
        double s = 0.0;
        for (int k = 0; k < 40; ++k)
            s += std::sin(2.0 * kPi * f[k] * double(m) / rate + p[k]);
        x[m] = float(s / 40.0 * 2.0);
    }
    return x;
}

// Largest |y[m] - g x[m - delay]| over m >= skip, relative to max |x|
double delayError(const std::vector<float>& x, const std::vector<float>& y, int delay, std::size_t skip,
                  double g = 1.0) {
    double err = 0.0, peak = 0.0;
    for (std::size_t m = skip; m < y.size(); ++m) {
        const long long k = (long long)m - delay;
        const double ref = k >= 0 && k < (long long)x.size() ? x[std::size_t(k)] : 0.0;
        err = std::max(err, std::abs(double(y[m]) - g * ref));
        peak = std::max(peak, std::abs(ref));
    }
    return err / std::max(peak, 1e-30);
}

// --- Tests

void testLatencyFunction() {
    using denoiser::hostLatencySamples;
    // hop 80, n_fft 128: processor delay 48, regrouping 79, resamplers 2 x 48
    CHECK(hostLatencySamples(16000, 80, 48) == 127);
    CHECK(hostLatencySamples(48000, 80, 48) == 3 * (96 + 127));
    CHECK(hostLatencySamples(96000, 80, 48) == 6 * (96 + 127));
    CHECK(hostLatencySamples(44100, 80, 48) == 614);  // floor(223 * 2.75625 = 614.64)
    CHECK(hostLatencySamples(88200, 80, 48) == 1229);
    std::printf("latency, hop 80, processor delay 48 (n_fft 128):\n");
    for (int rate : kAllRates) {
        const int l = hostLatencySamples(rate, 80, 48);
        std::printf("  %6d Hz: %4d samples = %.3f ms\n", rate, l, 1000.0 * l / rate);
    }
    bool threw = false;
    try {
        hostLatencySamples(8000, 80, 48);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        hostLatencySamples(48000, 0, 48);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        HostRateAdapter a;
        a.prepare({47999, 1, 80, 48, HighBand::Drop});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

// Impulses and a broadband signal through pass-through processors come out
// at exactly the reported latency, the same for every block pattern
void testLatencyAndBlocks(int rate, int hop, int delay, bool useNull, HighBand band) {
    HostRateAdapter a;
    a.prepare({rate, 1, hop, delay, band});
    DelayGain proc(delay, 1.0f);
    if (!useNull)
        a.setProcessor(0, &proc);
    const int lat = a.latencySamples();
    CHECK(lat == denoiser::hostLatencySamples(rate, hop, delay));

    // a quarter second (at least the impulse positions) + tail
    const std::size_t n = std::size_t(std::max(rate / 4, 4200)) + 3 * std::size_t(lat) + 400;
    // Impulses at several positions (so several resampler phases)
    std::vector<float> imp(n, 0.0f);
    const int pos[] = {lat + 3, lat + 1001, lat + 2502, lat + 4103};
    for (int p : pos)
        imp[std::size_t(p)] = 1.0f;
    // Broadband: sines up to 6 kHz (the passband, below the 16 kHz Nyquist)
    std::vector<float> bb = sines(rate, n, 6000.0, unsigned(rate));

    std::vector<float> impRef, bbRef;
    bool sameImp = true, sameBb = true;
    for (int kind : kBlockKinds) {
        a.reset();
        std::vector<float> yi = run(a, 0, imp, pattern(kind, n, 17u + unsigned(kind)));
        a.reset();
        std::vector<float> yb = run(a, 0, bb, pattern(kind, n, 99u + unsigned(kind)), kind == 333);
        if (impRef.empty()) {
            impRef = yi;
            bbRef = yb;
        } else {
            sameImp = sameImp && std::memcmp(yi.data(), impRef.data(), n * sizeof(float)) == 0;
            sameBb = sameBb && std::memcmp(yb.data(), bbRef.data(), n * sizeof(float)) == 0;
        }
    }
    CHECK(sameImp);
    CHECK(sameBb);
    CHECK(a.underruns() == 0);

    // Each impulse's response peaks exactly `lat` samples later
    bool peaksOk = true;
    for (int p : pos) {
        const int lo = p + lat - 200, hi = p + lat + 200;
        int best = lo;
        for (int m = lo; m <= hi; ++m)
            if (std::abs(impRef[std::size_t(m)]) > std::abs(impRef[std::size_t(best)]))
                best = m;
        if (best != p + lat) {
            std::printf("  impulse at %d: peak at %d, expected %d\n", p, best, p + lat);
            peaksOk = false;
        }
    }
    CHECK(peaksOk);
    // Nothing comes out before the impulse plus the latency minus the filter's
    // half length (causality and no early leakage)
    double early = 0.0;
    for (int m = 0; m < pos[0]; ++m)
        early = std::max(early, double(std::abs(impRef[std::size_t(m)])));
    CHECK(early == 0.0);

    // Broadband: matches the input delayed by lat, not by lat +- 1
    const std::size_t skip = std::size_t(2 * lat);
    const double e0 = delayError(bb, bbRef, lat, skip);
    const double em = delayError(bb, bbRef, lat - 1, skip);
    const double ep = delayError(bb, bbRef, lat + 1, skip);
    std::printf("  %6d Hz hop %3d delay %3d %s %s: latency %4d, broadband error %.1f dB (at latency -1: %.1f, +1: %.1f)\n",
                rate, hop, delay, useNull ? "null    " : "stand-in", band == HighBand::Drop ? "A" : "B", lat, db(e0),
                db(em), db(ep));
    CHECK(db(e0) < -70.0);
    CHECK(e0 < em / 100 && e0 < ep / 100);
}

// Per-band round trip (sines) at a host rate, identity processor
void testRoundTrip(int rate, HighBand band, double passTolDb) {
    HostRateAdapter a;
    a.prepare({rate, 1, 80, 48, band});
    const int lat = a.latencySamples();
    std::printf("  %d Hz, option %s, round trip with a pass-through model:\n", rate, band == HighBand::Drop ? "A" : "B");
    double worstPass = -1000.0;
    const double freqs[] = {50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 6500, 7000,
                            7500, 8000, 8500, 9000, 10000, 12000, 15000, 20000};
    for (double hz : freqs) {
        if (hz >= rate / 2.0)
            continue;
        const std::size_t n = std::size_t(rate) / 2;
        std::vector<float> x(n);
        for (std::size_t m = 0; m < n; ++m)
            x[m] = float(0.5 * std::sin(2.0 * kPi * hz * double(m) / rate + 0.3));
        a.reset();
        std::vector<float> y = run(a, 0, x, pattern(512, n, 1));
        const std::size_t skip = std::size_t(2 * lat) + 1000;
        // Error vs. the delayed input, and output level (RMS ratio)
        double se = 0.0, sy = 0.0, sx = 0.0;
        for (std::size_t m = skip; m < n; ++m) {
            const double d = double(y[m]) - double(x[m - std::size_t(lat)]);
            se += d * d;
            sy += double(y[m]) * y[m];
            sx += double(x[m - std::size_t(lat)]) * x[m - std::size_t(lat)];
        }
        const double err = std::sqrt(se / sx), gain = std::sqrt(sy / sx);
        std::printf("    %7.0f Hz: error %7.1f dB, output level %8.2f dB\n", hz, db(err), db(gain));
        if (hz <= 7000.0)
            worstPass = std::max(worstPass, db(err));
        if (band == HighBand::Drop && hz >= 8000.0)
            CHECK(db(gain) < -85.0);
        if (band == HighBand::PassThrough)
            CHECK(db(err) < -100.0);  // everything passes unchanged
    }
    std::printf("    worst error below 7 kHz: %.1f dB (tolerance %.0f dB)\n", worstPass, passTolDb);
    CHECK(worstPass < passTolDb);
}

// 16 kHz: no resampling, output == the hop-by-hop ("command-line") result
// delayed by hop - 1, bit for bit, for a stateful processor
void testBypass() {
    const int hop = 80;
    const std::size_t n = 16000;
    std::vector<float> x = sines(16000, n, 7900.0, 5);
    x[100] = -0.0f;
    // The command-line way: whole hops through the processor
    Weird cli;
    std::vector<float> ref(n, 0.0f);
    for (std::size_t at = 0; at + hop <= n; at += hop)
        cli.processHop(x.data() + at, ref.data() + at, hop);

    for (HighBand band : {HighBand::Drop, HighBand::PassThrough}) {
        HostRateAdapter a;
        Weird proc;
        a.prepare({16000, 1, hop, 0, band});
        a.setProcessor(0, &proc);
        CHECK(a.latencySamples() == hop - 1);
        for (int kind : kBlockKinds) {
            a.reset();
            std::vector<float> y = run(a, 0, x, pattern(kind, n, 3));
            bool same = true;
            for (std::size_t m = hop - 1; m + hop < n; ++m)
                same = same && std::memcmp(&y[m], &ref[m - (hop - 1)], sizeof(float)) == 0;
            CHECK(same);
        }
    }
    // Pass-through at 16 kHz: the input, delayed, bit for bit
    HostRateAdapter a;
    a.prepare({16000, 1, 80, 48, HighBand::Drop});
    std::vector<float> y = run(a, 0, x, pattern(0, n, 4));
    bool same = true;
    for (std::size_t m = 127; m < n; ++m)
        same = same && std::memcmp(&y[m], &x[m - 127], sizeof(float)) == 0;
    CHECK(same);
}

// reset() gives exactly the first run again, and resets the processors
void testReset() {
    for (int rate : {44100, 48000}) {
        HostRateAdapter a;
        Weird proc;
        a.prepare({rate, 1, 80, 48, HighBand::PassThrough});
        a.setProcessor(0, &proc);
        const std::size_t n = std::size_t(rate) / 3;
        std::vector<float> x = sines(rate, n, 15000.0, 8);
        std::vector<float> first = run(a, 0, x, pattern(0, n, 1));
        a.reset();
        std::vector<float> second = run(a, 0, x, pattern(256, n, 1));
        CHECK(std::memcmp(first.data(), second.data(), n * sizeof(float)) == 0);
        // processClipAligned == the real-time render shifted by the latency
        std::vector<float> aligned(n);
        processClipAligned(a, 0, x.data(), aligned.data(), std::int64_t(n));
        const std::size_t lat = std::size_t(a.latencySamples());
        CHECK(std::memcmp(aligned.data(), first.data() + lat, (n - lat) * sizeof(float)) == 0);
    }
    DelayGain p(5, 1.0f);
    HostRateAdapter a;
    a.prepare({48000, 2, 80, 5, HighBand::Drop});
    a.setProcessor(1, &p);
    a.reset();
    CHECK(p.resets == 1);
}

// Two channels, different inputs and processors: each equals its mono run
void testStereo() {
    const int rate = 44100;
    const std::size_t n = 20000;
    std::vector<float> l = sines(rate, n, 6000.0, 1), r = sines(rate, n, 12000.0, 2);
    Weird pl, pr, ml, mr;
    HostRateAdapter st;
    st.prepare({rate, 2, 80, 48, HighBand::Drop});
    st.setProcessor(0, &pl);
    st.setProcessor(1, &pr);
    std::vector<float> ol(n), orr(n);
    std::vector<float> il = l, ir = r;
    std::size_t at = 0;
    for (int b : pattern(0, n, 77)) {
        const float* in[2] = {il.data() + at, ir.data() + at};
        float* out[2] = {ol.data() + at, orr.data() + at};
        st.process(in, out, b);
        at += std::size_t(b);
    }
    HostRateAdapter mono;
    mono.prepare({rate, 1, 80, 48, HighBand::Drop});
    mono.setProcessor(0, &ml);
    std::vector<float> el = run(mono, 0, l, pattern(64, n, 0));
    mono.setProcessor(0, &mr);
    mono.reset();
    std::vector<float> er = run(mono, 0, r, pattern(64, n, 0));
    CHECK(std::memcmp(el.data(), ol.data(), n * sizeof(float)) == 0);
    CHECK(std::memcmp(er.data(), orr.data(), n * sizeof(float)) == 0);
    CHECK(std::memcmp(el.data(), er.data(), n * sizeof(float)) != 0);
}

// No allocation in process / reset / setProcessor, after prepare
void testNoAllocation() {
    for (HighBand band : {HighBand::Drop, HighBand::PassThrough})
        for (int rate : {16000, 44100, 48000, 96000}) {
            HostRateAdapter a;
            a.prepare({rate, 2, 80, 48, band});
            DelayGain p0(48, 1.0f), p1(48, 0.5f);
            std::vector<float> x = sines(rate, 50000, 15000.0, 1), y0(50000), y1(50000);
            std::vector<int> blocks = pattern(0, 50000, 5);
            const long long before = gAllocations;
            a.setProcessor(0, &p0);
            a.setProcessor(1, &p1);
            std::size_t at = 0;
            for (int b : blocks) {
                const float* in[2] = {x.data() + at, x.data() + at};
                float* out[2] = {y0.data() + at, y1.data() + at};
                a.process(in, out, b);
                at += std::size_t(b);
            }
            a.reset();
            a.setProcessor(0, nullptr);
            a.processChannel(0, x.data(), y0.data(), 4096);
            processClipAligned(a, 1, x.data(), y1.data(), 10000);
            CHECK(gAllocations == before);
            CHECK(p0.calls > 0);
        }
}

// Option B with a "model" that removes everything: only the band the model
// can't see is left, at unit gain, and the low band is gone
void testHighBandSplit(int rate) {
    HostRateAdapter a;
    DelayGain zero(48, 0.0f);
    a.prepare({rate, 1, 80, 48, HighBand::PassThrough});
    a.setProcessor(0, &zero);
    const int lat = a.latencySamples();
    std::printf("  %d Hz, option B, model outputs silence:\n", rate);
    for (double hz : {1000.0, 5000.0, 7000.0, 7500.0, 8000.0, 9000.0, 12000.0, 18000.0}) {
        const std::size_t n = std::size_t(rate) / 2;
        std::vector<float> x(n);
        for (std::size_t m = 0; m < n; ++m)
            x[m] = float(0.5 * std::sin(2.0 * kPi * hz * double(m) / rate));
        a.reset();
        std::vector<float> y = run(a, 0, x, pattern(480, n, 1));
        const std::size_t skip = std::size_t(2 * lat) + 1000;
        double sy = 0.0, sx = 0.0;
        for (std::size_t m = skip; m < n; ++m) {
            sy += double(y[m]) * y[m];
            sx += double(x[m]) * x[m];
        }
        const double level = db(std::sqrt(sy / sx));
        std::printf("    %6.0f Hz: output level %8.2f dB, error vs. delayed input %7.1f dB\n", hz, level,
                    db(delayError(x, y, lat, skip)));
        if (hz <= 7000.0)
            CHECK(level < -85.0);
        if (hz >= 8000.0)
            CHECK(std::abs(level) < 0.01 && db(delayError(x, y, lat, skip)) < -80.0);
    }
}

void benchmark() {
  for (int rate : {48000, 44100}) {
    const int seconds = 10, block = 512;
    const std::size_t n = std::size_t(rate) * seconds;
    std::vector<float> l = sines(rate, n, 20000.0, 1), r = sines(rate, n, 20000.0, 2);
    std::printf("cost, %d Hz stereo, blocks of %d, %d s of audio, pass-through stand-in:\n", rate, block, seconds);
    for (HighBand band : {HighBand::Drop, HighBand::PassThrough}) {
        HostRateAdapter a;
        a.prepare({rate, 2, 80, 48, band});
        DelayGain p0(48, 1.0f), p1(48, 1.0f);
        a.setProcessor(0, &p0);
        a.setProcessor(1, &p1);
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t at = 0; at < n; at += block) {
            float* io[2] = {l.data() + at, r.data() + at};
            a.process(io, io, int(std::min<std::size_t>(block, n - at)));
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  option %s: %.2f ms per second of audio (%.2f%% of one core)\n",
                    band == HighBand::Drop ? "A" : "B", 1000.0 * s / seconds, 100.0 * s / seconds);
    }
  }
}

}  // namespace

int main() {
    testLatencyFunction();

    std::printf("latency and block-size independence:\n");
    for (int rate : kAllRates) {
        testLatencyAndBlocks(rate, 80, 48, false, HighBand::Drop);
        testLatencyAndBlocks(rate, 80, 48, true, HighBand::Drop);
    }
    for (int rate : {44100, 48000, 96000}) {
        testLatencyAndBlocks(rate, 80, 48, false, HighBand::PassThrough);
        testLatencyAndBlocks(rate, 160, 96, false, HighBand::Drop);  // n_fft 256, hop 160
        testLatencyAndBlocks(rate, 64, 0, false, HighBand::Drop);
        testLatencyAndBlocks(rate, 1, 3, false, HighBand::Drop);
    }

    std::printf("round trips:\n");
    for (int rate : {44100, 48000}) {
        testRoundTrip(rate, HighBand::Drop, -80.0);
        testRoundTrip(rate, HighBand::PassThrough, -80.0);
    }
    for (int rate : {44100, 48000})
        testHighBandSplit(rate);

    testBypass();
    testReset();
    testStereo();
    testNoAllocation();
    benchmark();
    return check::finish();
}
