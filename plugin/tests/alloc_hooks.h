#pragma once
// Counts heap allocations while armed: replaces the global operator new /
// delete (all forms) and, on glibc without a sanitizer, also malloc / calloc /
// realloc / free. Include in exactly one source file of a test executable.
// The same hooks as test_stft_alloc.cpp (which has them inline).
//
//   alloc::count([&] { ...code that must not allocate... })   -> number of allocations
//   alloc::countOnThisThread([&] { ... })   -> the same, ignoring other threads
//                                              (e.g. a framework's timer thread)
#include <atomic>
#include <cstdlib>
#include <new>
#include <thread>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace alloc {
inline std::atomic<bool> armed{false};
inline std::atomic<long> allocations{0};
inline std::atomic<bool> oneThread{false};
inline std::thread::id countedThread;  // written only while not armed
// operator new throws std::bad_alloc for larger requests (a machine short of
// memory, for tests of what happens then); no limit by default
inline std::atomic<std::size_t> failAbove{static_cast<std::size_t>(-1)};

inline void note() {
    if (armed.load(std::memory_order_relaxed) &&
        (!oneThread.load(std::memory_order_relaxed) || std::this_thread::get_id() == countedThread))
        allocations.fetch_add(1, std::memory_order_relaxed);
}

template <class F>
long count(F&& f) {
    allocations = 0;
    armed = true;
    f();
    armed = false;
    return allocations.load();
}

template <class F>
long countOnThisThread(F&& f) {
    countedThread = std::this_thread::get_id();
    oneThread = true;
    const long n = count(f);
    oneThread = false;
    return n;
}
}  // namespace alloc

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

namespace alloc {
inline void* allocate(std::size_t n) {
#if !HOOK_MALLOC
    note();
#endif
    if (n > failAbove.load(std::memory_order_relaxed))
        throw std::bad_alloc();
    if (void* p = std::malloc(n ? n : 1))
        return p;
    throw std::bad_alloc();
}

inline void* allocateAligned(std::size_t n, std::size_t align) {
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

inline void freeAligned(void* p) noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}
}  // namespace alloc

void* operator new(std::size_t n) { return alloc::allocate(n); }
void* operator new[](std::size_t n) { return alloc::allocate(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return alloc::allocate(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return alloc::allocate(n); } catch (...) { return nullptr; }
}
void* operator new(std::size_t n, std::align_val_t a) { return alloc::allocateAligned(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return alloc::allocateAligned(n, std::size_t(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { alloc::freeAligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { alloc::freeAligned(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { alloc::freeAligned(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { alloc::freeAligned(p); }

// ---- malloc family (glibc, no sanitizer): catches C code such as KISS FFT ---
#if HOOK_MALLOC
extern "C" {
void* __libc_malloc(std::size_t);
void* __libc_calloc(std::size_t, std::size_t);
void* __libc_realloc(void*, std::size_t);
void __libc_free(void*);

void* malloc(std::size_t n) {
    alloc::note();
    return __libc_malloc(n);
}
void* calloc(std::size_t a, std::size_t b) {
    alloc::note();
    return __libc_calloc(a, b);
}
void* realloc(void* p, std::size_t n) {
    alloc::note();
    return __libc_realloc(p, n);
}
void free(void* p) {
    if (p != nullptr)  // free(nullptr) does nothing (JUCE calls it for empty buffers)
        alloc::note();
    __libc_free(p);
}
}
#endif
