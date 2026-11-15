#pragma once
// Dense and spike-driven kernels for the networks (internal to the core).
//
// Written for auto-vectorisation without -ffast-math: a float reduction is
// only vectorised by a compiler when it may reorder it, so the dot products
// keep kLanes independent partial sums by hand (one SIMD register's worth
// per 4 or 8 lanes), and the column additions have no reduction at all.
// Baseline x86-64 (SSE2) or ARM64 (NEON) code; no -march needed.

#include <cmath>
#include <cstddef>

namespace denoiser {
namespace kernels {

constexpr std::size_t kLanes = 16;

// sum_i a[i] * b[i]
inline float dot(const float* a, const float* b, std::size_t n) noexcept {
    float acc[kLanes] = {};
    std::size_t i = 0;
    for (; i + kLanes <= n; i += kLanes)
        for (std::size_t k = 0; k < kLanes; ++k)
            acc[k] += a[i + k] * b[i + k];
    float tail = 0.0f;
    for (; i < n; ++i)
        tail += a[i] * b[i];
    // pairwise, so the order doesn't depend on n
    for (std::size_t w = kLanes / 2; w > 0; w /= 2)
        for (std::size_t k = 0; k < w; ++k)
            acc[k] += acc[k + w];
    return acc[0] + tail;
}

// y = W x + b (b may be nullptr), W row-major [rows, cols]. kRows rows at a
// time share the loads of x, each with kRowLanes partial sums. Of the forms
// measured (one row with 16 lanes, 2 or 4 rows with 8 or 16 lanes, the
// transposed axpy form; GCC 13 and Clang 18, -O3, SSE2) this one was the
// fastest or close to it with both compilers: ~9-15 GMAC/s when W fits in
// the cache (256 x 256), ~4-7 when it doesn't (1024 x 1024).
constexpr std::size_t kRows = 4, kRowLanes = 8;

inline void matVec(const float* w, const float* b, const float* x, float* y, std::size_t rows,
                   std::size_t cols) noexcept {
    std::size_t r = 0;
    for (; r + kRows <= rows; r += kRows) {
        float acc[kRows][kRowLanes] = {};
        std::size_t i = 0;
        for (; i + kRowLanes <= cols; i += kRowLanes)
            for (std::size_t q = 0; q < kRows; ++q)
                for (std::size_t k = 0; k < kRowLanes; ++k)
                    acc[q][k] += w[(r + q) * cols + i + k] * x[i + k];
        for (std::size_t q = 0; q < kRows; ++q) {
            float tail = 0.0f;
            for (std::size_t j = i; j < cols; ++j)
                tail += w[(r + q) * cols + j] * x[j];
            for (std::size_t s = kRowLanes / 2; s > 0; s /= 2)
                for (std::size_t k = 0; k < s; ++k)
                    acc[q][k] += acc[q][k + s];
            y[r + q] = (acc[q][0] + tail) + (b ? b[r + q] : 0.0f);
        }
    }
    for (; r < rows; ++r)
        y[r] = dot(w + r * cols, x, cols) + (b ? b[r] : 0.0f);
}

// y += row (n floats): one column of W for an input that is exactly 1
inline void addRow(const float* row, float* y, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        y[i] += row[i];
}

// y += sum over the active inputs j of wt[j * n ...]: W s for spikes s in {0, 1},
// with W stored transposed ([inputs, n], so column j of W is contiguous)
inline void addActiveRows(const float* wt, const int* active, int count, float* y, std::size_t n) noexcept {
    for (int k = 0; k < count; ++k)
        addRow(wt + std::size_t(active[k]) * n, y, n);
}

inline void relu(float* x, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

inline void sigmoid(float* x, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

}  // namespace kernels
}  // namespace denoiser
