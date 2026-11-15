// TcnNetwork::process() and reset() don't allocate: this binary replaces the
// global operator new / delete (all forms) and, on glibc without a sanitizer,
// also malloc / calloc / realloc / free, and counts calls while armed (the
// technique of test_stft_alloc.cpp). Every golden TCN model and a default
// size synthetic TCN run frames, reset() and frames again while armed, also
// through StreamProcessor::process; the count must stay 0.
#include "check.h"
#include "golden.h"
#include "tcn_synth.h"

#include "denoiser/stream.h"
#include "denoiser/tcn.h"

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

// Frames, reset(), frames again, all armed; returns the allocations counted
long countAllocations(const denoiser::ModelFile& model, const float* x, std::size_t rows) {
    const denoiser::ModelInfo& info = model.info();
    denoiser::TcnNetwork net(model);  // allocates: not armed
    std::vector<float> y(std::size_t(info.nOut)), out(std::size_t(info.hop));
    std::vector<float> chunk(std::size_t(info.hop), 0.01f);
    denoiser::StreamProcessor* stream = nullptr;
    std::unique_ptr<denoiser::StreamProcessor> owner;
    if (info.framing == denoiser::Framing::Causal) {
        owner = std::make_unique<denoiser::StreamProcessor>(info);
        stream = owner.get();
    }
    allocations = 0;
    armed = true;
    for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t t = 0; t < rows; ++t)
            net.process(x + t * std::size_t(info.nIn), y.data());
        net.reset();
        if (stream) {
            for (std::size_t t = 0; t < rows; ++t)
                stream->process(chunk.data(), out.data(), net);
            stream->reset();
            net.reset();
        }
    }
    armed = false;
    return allocations.load();
}

}  // namespace

int main(int argc, char** argv) {
    golden::init(argc, argv);
    std::printf("malloc family hooked: %s\n", HOOK_MALLOC ? "yes" : "no (operator new only)");

    // The counter works
    static int* volatile probe = nullptr;
    armed = true;
    probe = new int(1);
    armed = false;
    CHECK(allocations.load() >= 1);
    delete probe;

    for (const std::string& name : golden::caseNames()) {
        const golden::Case c = golden::load(name);
        if (c.info().netType != "tcn")
            continue;
        const denoiser::Tensor& x = c.get("x");
        const long n = countAllocations(c.model, x.data, x.shape[0]);
        std::printf("%-28s allocations in process/reset: %ld\n", name.c_str(), n);
        CHECK(n == 0);
    }

    const denoiser::ModelFile model = tcn_synth::model(tcn_synth::Spec{});
    std::vector<float> x(std::size_t(tcn_synth::kNIn) * 20, 0.5f);
    const long n = countAllocations(model, x.data(), 20);
    std::printf("%-28s allocations in process/reset: %ld\n", "synthetic default size", n);
    CHECK(n == 0);
    return check::finish();
}
