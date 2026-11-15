#pragma once
// Placeholder API of the denoiser core. The real API (model loading, the
// causal STFT, inference and a hop-by-hop processor) comes in later issues;
// see ../../README.md. Nothing in core/ may depend on JUCE or on host code.

#include <cstddef>

namespace denoiser {

// Version of the core library, e.g. "0.1.0"
const char* version() noexcept;

// Stand-in for the denoiser until the real one exists: copies n samples from
// in to out (in == out is allowed). Real-time safe.
void passThrough(const float* in, float* out, std::size_t n) noexcept;

}  // namespace denoiser
