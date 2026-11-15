#include "denoiser/features.h"

#include <algorithm>
#include <cmath>

namespace denoiser {

std::vector<float> hannWindow(int n) {
    // As torch.hann_window(n) (periodic): torch builds a symmetric window of
    // n + 1 points, i.e. step 2 pi / n as a float32 scalar, cos in float32,
    // * -0.5 + 0.5, and drops the last point
    std::vector<float> w(static_cast<std::size_t>(n));
    constexpr double pi = 3.14159265358979323846;  // M_PI is not standard C++
    const float step = static_cast<float>(2.0 * pi / n);
    for (int i = 0; i < n; ++i)
        w[static_cast<std::size_t>(i)] = std::cos(static_cast<float>(i) * step) * -0.5f + 0.5f;
    return w;
}

namespace {
// torch.polar(mag, angle): mag * cos(angle), mag * sin(angle)
inline std::complex<float> polar(float mag, float angle) {
    return {mag * std::cos(angle), mag * std::sin(angle)};
}
}  // namespace

FrameCodec::FrameCodec(const ModelInfo& info)
    : info_(info),
      fft_(info.nFft),
      longFft_(info.longFactor ? info.longNFft : 2),
      window_(hannWindow(info.nFft)),
      longWindow_(info.longFactor ? hannWindow(info.longNFft) : std::vector<float>()),
      buffer_(static_cast<std::size_t>(info.widestWindow())),
      spec_(static_cast<std::size_t>(info.nFreq)),
      longSpec_(info.longFactor ? static_cast<std::size_t>(info.longNFft / 2 + 1) : 0),
      outSpec_(static_cast<std::size_t>(info.nFreq)),
      mag_(static_cast<std::size_t>(info.nFreq)),
      compReal_(info.output == OutputKind::ComplexMask ? static_cast<std::size_t>(info.nFreq) : 0),
      compImag_(compReal_.size()),
      longMag_(info.longFactor ? static_cast<std::size_t>(info.nFreq) : 0),
      x_(static_cast<std::size_t>(info.nIn)),
      frame_(static_cast<std::size_t>(info.nFft)) {}

const float* FrameCodec::analyse(const float* shortFrame, const float* longFrame) noexcept {
    const int nFft = info_.nFft, F = info_.nFreq;

    // Short spectrum and its magnitude
    for (int i = 0; i < nFft; ++i)
        buffer_[i] = shortFrame[i] * window_[i];
    fft_.forward(buffer_.data(), spec_.data());
    for (int k = 0; k < F; ++k)
        mag_[k] = std::abs(spec_[k]);

    // complex_mask: magnitude ^ compress at the noisy phase
    if (info_.output == OutputKind::ComplexMask) {
        for (int k = 0; k < F; ++k) {
            const std::complex<float> c =
                polar(std::pow(mag_[k], info_.compress), std::atan2(spec_[k].imag(), spec_[k].real()));
            compReal_[k] = c.real();
            compImag_[k] = c.imag();
        }
    }

    // Long window: power of the finer bins, pooled onto the short bins
    // (avg_pool1d, kernel = stride = long_factor, padding long_factor / 2,
    // count_include_pad=False), * long_factor, sqrt, / long_factor
    if (info_.longFactor) {
        const int L = info_.longNFft, lf = info_.longFactor, pad = lf / 2, bins = L / 2 + 1;
        for (int i = 0; i < L; ++i)
            buffer_[i] = longFrame[i] * longWindow_[i];
        longFft_.forward(buffer_.data(), longSpec_.data());
        const float lff = static_cast<float>(lf);
        for (int k = 0; k < F; ++k) {
            const int lo = std::max(k * lf - pad, 0), hi = std::min(k * lf - pad + lf, bins);
            float sum = 0.0f;
            for (int j = lo; j < hi; ++j) {
                const float a = std::abs(longSpec_[j]);
                sum += a * a;
            }
            const float mean = sum / static_cast<float>(hi - lo);
            longMag_[k] = std::sqrt(mean * lff) / lff;
        }
    }

    // The model input, part by part in the file's order
    float* out = x_.data();
    for (const auto& part : info_.inputLayout) {
        switch (part.first) {
            case InputPart::Log1pMag:
                for (int k = 0; k < F; ++k) out[k] = std::log1p(mag_[k]);
                break;
            case InputPart::CompressedReal:
                std::copy(compReal_.begin(), compReal_.end(), out);
                break;
            case InputPart::CompressedImag:
                std::copy(compImag_.begin(), compImag_.end(), out);
                break;
            case InputPart::Log1pLongMag:
                for (int k = 0; k < F; ++k) out[k] = std::log1p(longMag_[k]);
                break;
        }
        out += part.second;
    }
    return x_.data();
}

const float* FrameCodec::synthesise(const float* pred) noexcept {
    const int F = info_.nFreq;
    switch (info_.output) {
        case OutputKind::Magnitude:
            // expm1(pred) clamped at 0, at the noisy phase
            for (int k = 0; k < F; ++k)
                outSpec_[k] = polar(std::max(std::expm1(pred[k]), 0.0f), std::atan2(spec_[k].imag(), spec_[k].real()));
            break;
        case OutputKind::Mask:
            // noisy magnitude * gain, at the noisy phase
            for (int k = 0; k < F; ++k)
                outSpec_[k] = polar(mag_[k] * pred[k], std::atan2(spec_[k].imag(), spec_[k].real()));
            break;
        case OutputKind::ComplexMask: {
            const float bound = info_.maskBound;
            for (int k = 0; k < F; ++k) {
                float mr = pred[k], mi = pred[F + k];
                if (bound != 0.0f) {  // pipeline.bound_mask
                    const float size = std::sqrt(mr * mr + mi * mi + 1e-8f);
                    const float scale = bound * std::tanh(size / bound) / size;
                    mr *= scale;
                    mi *= scale;
                }
                // complex multiply with the compressed noisy bin, then undo the compression
                const float xr = compReal_[k], xi = compImag_[k];
                const float zr = mr * xr - mi * xi, zi = mr * xi + mi * xr;
                const float zmag = std::abs(std::complex<float>(zr, zi));
                outSpec_[k] = polar(std::pow(zmag, info_.decompress), std::atan2(zi, zr));
            }
            break;
        }
    }
    return synthesiseSpectrum(outSpec_.data());
}

const float* FrameCodec::synthesiseSpectrum(const std::complex<float>* spec) noexcept {
    fft_.inverse(spec, frame_.data());  // ignores the imaginary parts of DC and Nyquist, like irfft
    for (int i = 0; i < info_.nFft; ++i)
        frame_[i] *= window_[i];
    return frame_.data();
}

}  // namespace denoiser
