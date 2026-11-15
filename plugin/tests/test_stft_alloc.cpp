// The per-hop calls don't allocate: this binary replaces the global operator
// new / delete (all forms) and, on glibc without a sanitizer, also malloc /
// calloc / realloc / free, and counts calls while armed. Every golden model's
// StreamProcessor (and FrameCodec for the centred ones) runs hops, reset()
// and the pass-through path while armed; the count must stay 0.
#include "check.h"
#include "golden.h"

#include "denoiser/features.h"
#include "denoiser/stream.h"

#include <atomic>
#include <cstdlib>
#include <new>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace {
std::atomic<bool> armed{false};
std::atomic<long> allocations{0};

inline void note() {
    if (armed.load(std::memory_order_relaxed))
        allocations.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

// ---- operator new / delete ---------------------------------------------------
// They forward to malloc / aligned allocation, so only count here when malloc
// itself is not hooked below (otherwise each new would count twice)
#if defined(__GLIBC__) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
#if defined(__has_feature)
#if !__has_feature(address_sanitizer) && !__has_feature(memory_sanitizer) && !__has_feature(thread_sanitizer)
#define HOOK_MALLOC 1
#endif
#else
#define HOOK_MALLOC 1
#endif
#endif

#ifndef HOOK_MALLOC
#define HOOK_MALLOC 0
#endif

namespace {
void* allocate(std::size_t n) {
#if !HOOK_MALLOC
    note();
#endif
    if (void* p = std::malloc(n ? n : 1))
        return p;
    throw std::bad_alloc();
}

void* allocateAligned(std::size_t n, std::size_t align) {
    note();  // aligned allocation doesn't go through malloc
    void* p = nullptr;
    n = (n + align - 1) / align * align;
#if defined(_WIN32)
    p = _aligned_malloc(n ? n : align, align);
#else
    p = std::aligned_alloc(align, n ? n : align);
#endif
    if (!p)
        throw std::bad_alloc();
    return p;
}

void freeAligned(void* p) noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}
}  // namespace

void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocate(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocate(n); } catch (...) { return nullptr; }
}
void* operator new(std::size_t n, std::align_val_t a) { return allocateAligned(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocateAligned(n, std::size_t(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { freeAligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { freeAligned(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { freeAligned(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { freeAligned(p); }

// ---- malloc family (glibc, no sanitizer): catches C code such as KISS FFT ---
#if HOOK_MALLOC
extern "C" {
void* __libc_malloc(std::size_t);
void* __libc_calloc(std::size_t, std::size_t);
void* __libc_realloc(void*, std::size_t);
void __libc_free(void*);

void* malloc(std::size_t n) {
    note();
    return __libc_malloc(n);
}
void* calloc(std::size_t a, std::size_t b) {
    note();
    return __libc_calloc(a, b);
}
void* realloc(void* p, std::size_t n) {
    note();
    return __libc_realloc(p, n);
}
void free(void* p) {
    note();
    __libc_free(p);
}
}
#endif

namespace {

// A stand-in network that touches its input and output, without allocating
class SumNetwork : public denoiser::FrameNetwork {
public:
    SumNetwork(int nIn, int nOut) : nIn_(nIn), nOut_(nOut) {}
    void reset() noexcept override { state_ = 0.0f; }
    void process(const float* x, float* y) noexcept override {
        float s = 0.0f;
        for (int i = 0; i < nIn_; ++i) s += x[i];
        state_ = 0.5f * state_ + 1e-3f * s;
        for (int i = 0; i < nOut_; ++i) y[i] = 0.5f + 0.01f * float(i % 7) + 1e-3f * state_;
    }

private:
    int nIn_, nOut_;
    float state_ = 0.0f;
};

}  // namespace

int main(int argc, char** argv) {
    golden::init(argc, argv);
    std::printf("malloc family hooked: %s\n", HOOK_MALLOC ? "yes" : "no (operator new only)");

    // The counter works (the pointer escapes into a volatile, so the
    // compiler may not elide the allocation)
    static int* volatile probe = nullptr;
    armed = true;
    probe = new int(1);
    armed = false;
    CHECK(allocations.load() >= 1);
    delete probe;
#if HOOK_MALLOC
    allocations = 0;
    armed = true;
    void* volatile m = std::malloc(16);
    armed = false;
    CHECK(allocations.load() == 1);
    std::free(m);
#endif

    for (const std::string& name : golden::caseNames()) {
        const golden::Case c = golden::load(name);
        const denoiser::ModelInfo& info = c.info();
        const denoiser::Tensor& in = c.get("input");
        SumNetwork net(info.nIn, info.nOut);
        std::vector<float> pred(std::size_t(info.nOut), 0.5f), out(std::size_t(info.hop));
        const std::size_t hops = in.size() / std::size_t(info.hop);

        if (info.framing == denoiser::Framing::Causal) {
            denoiser::StreamProcessor stream(info);  // allocates: not armed
            allocations = 0;
            armed = true;
            for (int pass = 0; pass < 2; ++pass) {
                for (std::size_t k = 0; k < hops; ++k) {
                    const float* hopIn = in.data + k * std::size_t(info.hop);
                    stream.process(hopIn, out.data(), net);
                    stream.analyse(hopIn);
                    stream.synthesise(pred.data(), out.data());
                    stream.analyse(hopIn);
                    stream.synthesisePassThrough(out.data());
                }
                stream.reset();
                net.reset();
            }
            armed = false;
        } else {
            denoiser::FrameCodec codec(info);
            std::vector<float> longFrame(std::size_t(info.widestWindow()), 0.1f);
            allocations = 0;
            armed = true;
            for (std::size_t k = 0; k + std::size_t(info.nFft) <= in.size(); k += std::size_t(info.hop)) {
                codec.analyse(in.data + k, info.longFactor ? longFrame.data() : nullptr);
                codec.synthesise(pred.data());
            }
            armed = false;
        }
        std::printf("%-28s allocations in per-hop calls: %ld\n", name.c_str(), allocations.load());
        CHECK(allocations.load() == 0);
    }
    return check::finish();
}
