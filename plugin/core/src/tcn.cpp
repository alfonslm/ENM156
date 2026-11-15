#include "denoiser/tcn.h"

#include <algorithm>
#include <cmath>
#include <string>

#if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
#define DENOISER_RESTRICT __restrict
#else
#define DENOISER_RESTRICT
#endif

namespace denoiser {

namespace {

// ---- kernels -------------------------------------------------------------------
//
// Every layer is y += W^T v over the nonzero entries of its input v: the
// weights are stored transposed, one row of n outputs per input, so the inner
// loop runs over contiguous outputs with independent lanes. Compilers
// vectorise it without -ffast-math (no reduction to reassociate). Most inputs
// come out of a ReLU, so about half are zero; their rows are never loaded.

// Collects the nonzero entries of x[0..n) as (base + i, x[i]); returns how many
inline std::size_t gather(const float* DENOISER_RESTRICT x, std::size_t n, int base, int* DENOISER_RESTRICT index,
                          float* DENOISER_RESTRICT value) noexcept {
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        index[count] = base + static_cast<int>(i);
        value[count] = x[i];
        count += x[i] != 0.0f ? 1u : 0u;  // branch-free
    }
    return count;
}

// y[o] += sum_k value[k] * w[index[k] * n + o], o < n; eight rows at a time,
// so y is loaded and stored once per eight rows (measured: one row at a time
// is ~1.7x slower, four at a time about the same or slightly slower than eight)
inline void addRows(float* DENOISER_RESTRICT y, std::size_t n, const float* DENOISER_RESTRICT w, const int* index,
                    const float* value, std::size_t count) noexcept {
    std::size_t k = 0;
    for (; k + 8 <= count; k += 8) {
        const float* DENOISER_RESTRICT r0 = w + static_cast<std::size_t>(index[k]) * n;
        const float* DENOISER_RESTRICT r1 = w + static_cast<std::size_t>(index[k + 1]) * n;
        const float* DENOISER_RESTRICT r2 = w + static_cast<std::size_t>(index[k + 2]) * n;
        const float* DENOISER_RESTRICT r3 = w + static_cast<std::size_t>(index[k + 3]) * n;
        const float* DENOISER_RESTRICT r4 = w + static_cast<std::size_t>(index[k + 4]) * n;
        const float* DENOISER_RESTRICT r5 = w + static_cast<std::size_t>(index[k + 5]) * n;
        const float* DENOISER_RESTRICT r6 = w + static_cast<std::size_t>(index[k + 6]) * n;
        const float* DENOISER_RESTRICT r7 = w + static_cast<std::size_t>(index[k + 7]) * n;
        const float v0 = value[k], v1 = value[k + 1], v2 = value[k + 2], v3 = value[k + 3];
        const float v4 = value[k + 4], v5 = value[k + 5], v6 = value[k + 6], v7 = value[k + 7];
        for (std::size_t o = 0; o < n; ++o)
            y[o] += ((v0 * r0[o] + v1 * r1[o]) + (v2 * r2[o] + v3 * r3[o])) +
                    ((v4 * r4[o] + v5 * r5[o]) + (v6 * r6[o] + v7 * r7[o]));
    }
    for (; k < count; ++k) {
        const float* DENOISER_RESTRICT r = w + static_cast<std::size_t>(index[k]) * n;
        const float v = value[k];
        for (std::size_t o = 0; o < n; ++o)
            y[o] += v * r[o];
    }
}

// relu(LayerNorm(h)) over n channels: biased variance; mean and variance
// accumulated in double (n is small, so this costs nothing)
inline void normRelu(const float* DENOISER_RESTRICT h, const float* DENOISER_RESTRICT weight,
                     const float* DENOISER_RESTRICT bias, float eps, std::size_t n,
                     float* DENOISER_RESTRICT out) noexcept {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        sum += h[i];
    const double mean = sum / static_cast<double>(n);
    double sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = h[i] - mean;
        sq += d * d;
    }
    const double rstd = 1.0 / std::sqrt(sq / static_cast<double>(n) + static_cast<double>(eps));
    for (std::size_t i = 0; i < n; ++i) {
        const float z = static_cast<float>((h[i] - mean) * rstd) * weight[i] + bias[i];
        out[i] = z > 0.0f ? z : 0.0f;
    }
}

// ---- loading -----------------------------------------------------------------

constexpr long long kMaxChannels = 1 << 14;
constexpr long long kMaxKernel = 64;
constexpr long long kMaxDilation = 1 << 16;
constexpr long long kMaxBlocks = 256;
constexpr std::size_t kMaxStateFloats = std::size_t(1) << 28;

[[noreturn]] void fail(const std::string& what) { throw FormatError("tcn: " + what); }

std::vector<float> copyOf(const Tensor& t) { return std::vector<float>(t.data, t.data + t.size()); }

// [rows, cols] row-major -> [cols, rows]
std::vector<float> transposed(const Tensor& t, std::size_t rows, std::size_t cols) {
    std::vector<float> out(rows * cols);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            out[c * rows + r] = t.data[r * cols + c];
    return out;
}

}  // namespace

// ---- TcnModel ------------------------------------------------------------------

TcnModel::TcnModel(const ModelFile& model) {
    const ModelInfo& info = model.info();
    const Json& net = model.net();
    if (net.at("type").string() != "tcn")
        fail("net.type is \"" + net.at("type").string() + "\", not \"tcn\"");
    nIn_ = info.nIn;
    nOut_ = info.nOut;
    outActivation_ = info.outActivation;
    channels_ = static_cast<int>(net.at("channels").integer(1, kMaxChannels));
    kernel_ = static_cast<int>(net.at("kernel").integer(1, kMaxKernel));
    const double eps = net.at("layer_norm_eps").number();
    if (!(eps > 0.0 && eps < 1.0))
        fail("layer_norm_eps must be in (0, 1)");
    eps_ = static_cast<float>(eps);
    if (nIn_ < 1 || nOut_ < 1)
        fail("n_in and n_out must be positive");

    const auto C = static_cast<std::size_t>(channels_), K = static_cast<std::size_t>(kernel_);
    const auto nIn = static_cast<std::size_t>(nIn_), nOut = static_cast<std::size_t>(nOut_);
    const TensorFile& t = model.tensors();

    inWeight_ = transposed(t.get("input.weight", {C, nIn}), C, nIn);
    inBias_ = copyOf(t.get("input.bias", {C}));

    const std::vector<Json>& blocks = net.at("blocks").array();
    if (static_cast<long long>(blocks.size()) > kMaxBlocks)
        fail("too many blocks");
    long long context = 1;
    std::size_t offset = 0;
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        const std::string p = "block" + std::to_string(b) + ".";
        Block block;
        block.dilation = static_cast<int>(blocks[b].at("dilation").integer(1, kMaxDilation));
        const long long trim = static_cast<long long>(kernel_ - 1) * block.dilation;
        if (blocks[b].at("trim").integer() != trim)
            fail("blocks[" + std::to_string(b) + "].trim is not (kernel - 1) * dilation");
        context += trim;
        block.ringFrames = static_cast<int>(trim);
        block.ringOffset = offset;
        offset += static_cast<std::size_t>(block.ringFrames) * C;
        if (offset > kMaxStateFloats)
            fail("state too large");

        block.normWeight = copyOf(t.get(p + "norm.weight", {C}));
        block.normBias = copyOf(t.get(p + "norm.bias", {C}));
        // [out, in, tap] -> [tap][in][out]
        const Tensor& conv = t.get(p + "conv.weight", {C, C, K});
        block.convWeight.resize(K * C * C);
        for (std::size_t o = 0; o < C; ++o)
            for (std::size_t i = 0; i < C; ++i)
                for (std::size_t j = 0; j < K; ++j)
                    block.convWeight[(j * C + i) * C + o] = conv.data[(o * C + i) * K + j];
        block.convBias = copyOf(t.get(p + "conv.bias", {C}));
        block.mixWeight = transposed(t.get(p + "mix.weight", {C, C}), C, C);
        block.mixBias = copyOf(t.get(p + "mix.bias", {C}));
        blocks_.push_back(std::move(block));
    }
    context_ = static_cast<int>(context);
    if (net.at("context").integer() != context)
        fail("net.context is not 1 + sum of the blocks' trim");
    if (info.contextFrames != context)
        fail("derived.context_frames is not 1 + sum of the blocks' trim");
    stateFloats_ = offset;

    finalWeight_ = copyOf(t.get("final_norm.weight", {C}));
    finalBias_ = copyOf(t.get("final_norm.bias", {C}));
    outWeight_ = transposed(t.get("output.weight", {nOut, C}), nOut, C);
    outBias_ = copyOf(t.get("output.bias", {nOut}));

    // The start-up of FrameModel.forward: context - 1 zero input frames
    start_ = makeState();
    Work work = makeWork();
    for (int f = 0; f + 1 < context_; ++f)
        step(start_, work, work.zeros.data(), work.out.data());
}

std::size_t TcnModel::macsPerFrame() const noexcept {
    const auto C = static_cast<std::size_t>(channels_);
    return static_cast<std::size_t>(nIn_) * C + blocks_.size() * (static_cast<std::size_t>(kernel_) + 1) * C * C +
           C * static_cast<std::size_t>(nOut_);
}

TcnModel::Work TcnModel::makeWork() const {
    const auto C = static_cast<std::size_t>(channels_);
    Work w;
    w.hidden.assign((blocks_.size() + 1) * C, 0.0f);
    w.conv.assign(C, 0.0f);
    w.current.assign(C, 0.0f);
    w.normed.assign(C, 0.0f);
    w.out.assign(static_cast<std::size_t>(nOut_), 0.0f);
    w.zeros.assign(static_cast<std::size_t>(nIn_), 0.0f);
    const std::size_t inputs = std::max(static_cast<std::size_t>(nIn_), static_cast<std::size_t>(kernel_) * C);
    w.index.assign(inputs, 0);
    w.value.assign(inputs, 0.0f);
    return w;
}

TcnState TcnModel::makeState() const {
    TcnState s;
    s.rings.assign(stateFloats_, 0.0f);
    s.position.assign(blocks_.size(), 0);
    return s;
}

void TcnModel::step(TcnState& state, Work& work, const float* x, float* y) const noexcept {
    const auto C = static_cast<std::size_t>(channels_);
    int* index = work.index.data();
    float* value = work.value.data();

    // input layer (1x1 conv)
    float* h = work.hidden.data();
    std::copy(inBias_.begin(), inBias_.end(), h);
    std::size_t n = gather(x, static_cast<std::size_t>(nIn_), 0, index, value);
    addRows(h, C, inWeight_.data(), index, value, n);

    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const Block& block = blocks_[b];
        float* ring = state.rings.data() + block.ringOffset;  // frames t - trim .. t - 1
        const int frames = block.ringFrames;                    // trim
        int& pos = state.position[b];                           // slot of frame t - trim
        float* current = work.current.data();
        normRelu(h, block.normWeight.data(), block.normBias.data(), eps_, C, current);

        // dilated conv: tap j sees the frame m = (K - 1 - j) * dilation back
        n = 0;
        for (int j = 0; j < kernel_; ++j) {
            const int m = (kernel_ - 1 - j) * block.dilation;
            const float* frame = current;
            if (m > 0) {
                const int f = pos - m < 0 ? pos - m + frames : pos - m;  // frame t - m is in slot pos + trim - m
                frame = ring + static_cast<std::size_t>(f) * C;
            }
            n += gather(frame, C, j * channels_, index + n, value + n);
        }
        // frame t - trim is not needed any more: frame t takes its slot
        if (frames > 0) {
            std::copy(current, current + C, ring + static_cast<std::size_t>(pos) * C);
            pos = pos + 1 == frames ? 0 : pos + 1;
        }
        float* conv = work.conv.data();
        std::copy(block.convBias.begin(), block.convBias.end(), conv);
        addRows(conv, C, block.convWeight.data(), index, value, n);
        for (std::size_t i = 0; i < C; ++i)
            conv[i] = conv[i] > 0.0f ? conv[i] : 0.0f;

        // channel mix (1x1 conv) plus the residual
        float* next = h + C;
        for (std::size_t i = 0; i < C; ++i)
            next[i] = h[i] + block.mixBias[i];
        n = gather(conv, C, 0, index, value);
        addRows(next, C, block.mixWeight.data(), index, value, n);

        h = next;
    }

    // final norm, output layer, activation
    normRelu(h, finalWeight_.data(), finalBias_.data(), eps_, C, work.normed.data());
    const auto nOut = static_cast<std::size_t>(nOut_);
    std::copy(outBias_.begin(), outBias_.end(), y);
    n = gather(work.normed.data(), C, 0, index, value);
    addRows(y, nOut, outWeight_.data(), index, value, n);
    switch (outActivation_) {
        case Activation::None: break;
        case Activation::Relu:
            for (std::size_t o = 0; o < nOut; ++o)
                y[o] = y[o] > 0.0f ? y[o] : 0.0f;
            break;
        case Activation::Sigmoid:
            for (std::size_t o = 0; o < nOut; ++o)
                y[o] = 1.0f / (1.0f + std::exp(-y[o]));
            break;
    }
}

// ---- TcnNetwork ----------------------------------------------------------------

TcnNetwork::TcnNetwork(const ModelFile& model) : TcnNetwork(std::make_shared<const TcnModel>(model)) {}

TcnNetwork::TcnNetwork(std::shared_ptr<const TcnModel> model)
    : model_(std::move(model)), state_(model_->startState()), work_(model_->makeWork()) {}

void TcnNetwork::reset() noexcept {
    const TcnState& start = model_->startState();
    std::copy(start.rings.begin(), start.rings.end(), state_.rings.begin());
    std::copy(start.position.begin(), start.position.end(), state_.position.begin());
}

void TcnNetwork::process(const float* x, float* y) noexcept { model_->step(state_, work_, x, y); }

std::vector<float> TcnNetwork::orderedState() const {
    const auto C = static_cast<std::size_t>(model_->channels());
    std::vector<float> out;
    out.reserve(state_.rings.size());
    for (int b = 0; b < model_->blocks(); ++b) {
        const float* ring = state_.rings.data() + model_->ringOffset(b);
        const int frames = model_->ringFrames(b);
        for (int k = 0; k < frames; ++k) {
            const auto f = static_cast<std::size_t>((state_.position[static_cast<std::size_t>(b)] + k) % frames);
            out.insert(out.end(), ring + f * C, ring + (f + 1) * C);
        }
    }
    return out;
}

void TcnNetwork::scrambleState(std::uint32_t seed) noexcept {
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 8;
    };
    for (float& v : state_.rings)
        v = static_cast<float>(next()) / static_cast<float>(1u << 24) * 100.0f - 50.0f;
    for (int b = 0; b < model_->blocks(); ++b)
        state_.position[static_cast<std::size_t>(b)] =
            model_->ringFrames(b) > 0 ? static_cast<int>(next() % static_cast<std::uint32_t>(model_->ringFrames(b))) : 0;
}

void TcnNetwork::startUp() noexcept {
    for (int f = 0; f + 1 < model_->context(); ++f)
        model_->step(state_, work_, work_.zeros.data(), work_.out.data());
}

std::unique_ptr<FrameNetwork> makeTcnNetwork(const ModelFile& model) { return std::make_unique<TcnNetwork>(model); }

}  // namespace denoiser
