#include "denoiser/networks.h"

#include "kernels.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace denoiser {

namespace {

constexpr long long kMaxWidth = 1 << 16;  // neurons per layer: far above anything trainable here

Activation parseActivation(const Json& j, const std::string& where) {
    const std::string& name = j.string();
    if (name == "relu") return Activation::Relu;
    if (name == "sigmoid") return Activation::Sigmoid;
    if (name == "none") return Activation::None;
    throw FormatError(where + ": unknown activation \"" + name + "\"");
}

void applyActivation(Activation a, float* x, std::size_t n) noexcept {
    if (a == Activation::Relu)
        kernels::relu(x, n);
    else if (a == Activation::Sigmoid)
        kernels::sigmoid(x, n);
}

std::size_t sz(int n) { return static_cast<std::size_t>(n); }

}  // namespace

// ---- MLP ----------------------------------------------------------------------

MlpNetwork::MlpNetwork(const ModelFile& model) {
    const ModelInfo& info = model.info();
    const Json& net = model.net();
    if (net.at("type").string() != "mlp")
        throw FormatError("net.type is \"" + net.at("type").string() + "\", not \"mlp\"");
    const auto& layers = net.at("layers").array();
    if (layers.empty())
        throw FormatError("net.layers: an MLP needs at least one layer");
    int width = 0, prev = info.nIn;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const std::string where = "net.layers[" + std::to_string(i) + "]";
        Layer l;
        l.in = static_cast<int>(layers[i].at("in").integer(1, kMaxWidth));
        l.out = static_cast<int>(layers[i].at("out").integer(1, kMaxWidth));
        l.activation = parseActivation(layers[i].at("activation"), where + ".activation");
        if (l.in != prev)
            throw FormatError(where + ".in is " + std::to_string(l.in) + ", expected " + std::to_string(prev));
        const bool last = i + 1 == layers.size();
        if (last && l.out != info.nOut)
            throw FormatError(where + ".out is " + std::to_string(l.out) + ", but n_out is " + std::to_string(info.nOut));
        if (last && l.activation != info.outActivation)
            throw FormatError(where + ".activation differs from derived.out_activation");
        if (!last && l.activation != Activation::Relu)
            throw FormatError(where + ".activation: hidden layers use relu");
        const std::string p = "linear" + std::to_string(i);
        l.weight = model.tensors().get(p + ".weight", {sz(l.out), sz(l.in)}).data;
        l.bias = model.tensors().get(p + ".bias", {sz(l.out)}).data;
        if (!last)
            width = std::max(width, l.out);
        prev = l.out;
        layers_.push_back(l);
    }
    ping_.assign(sz(std::max(width, 1)), 0.0f);
    pong_.assign(sz(std::max(width, 1)), 0.0f);
}

void MlpNetwork::process(const float* x, float* y) noexcept {
    const float* in = x;
    float* bufs[2] = {ping_.data(), pong_.data()};
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const Layer& l = layers_[i];
        float* out = i + 1 == layers_.size() ? y : bufs[i % 2];
        kernels::matVec(l.weight, l.bias, in, out, sz(l.out), sz(l.in));
        applyActivation(l.activation, out, sz(l.out));
        in = out;
    }
}

long long MlpNetwork::macsPerFrame() const noexcept {
    long long n = 0;
    for (const Layer& l : layers_)
        n += static_cast<long long>(l.in) * l.out;
    return n;
}

// ---- SNN weights ----------------------------------------------------------------

namespace {

// [rows, cols] row-major -> [cols, rows]
std::vector<float> transposed(const float* m, int rows, int cols) {
    std::vector<float> t(sz(rows) * sz(cols));
    for (std::size_t r = 0; r < sz(rows); ++r)
        for (std::size_t c = 0; c < sz(cols); ++c)
            t[c * sz(rows) + r] = m[r * sz(cols) + c];
    return t;
}

}  // namespace

std::shared_ptr<const SnnWeights> SnnWeights::build(const ModelFile& model) {
    const ModelInfo& info = model.info();
    const Json& net = model.net();
    if (net.at("type").string() != "snn")
        throw FormatError("net.type is \"" + net.at("type").string() + "\", not \"snn\"");
    auto w = std::make_shared<SnnWeights>();
    w->nIn = info.nIn;
    w->nOut = info.nOut;
    w->steps = static_cast<int>(net.at("steps_per_frame").integer(1, 1000));
    const double threshold = net.at("threshold").number();
    if (!(threshold > 0.0) || threshold > 1e30)
        throw FormatError("net.threshold must be positive, got " + std::to_string(threshold));
    w->threshold = static_cast<float>(threshold);
    w->recurrent = net.at("recurrent").boolean();
    w->outActivation = info.outActivation;
    const auto& hidden = net.at("hidden").array();
    if (hidden.empty())
        throw FormatError("net.hidden: an SNN needs at least one spiking layer");
    const TensorFile& t = model.tensors();
    int prev = info.nIn;
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        Layer l;
        l.in = prev;
        l.size = static_cast<int>(hidden[i].integer(1, kMaxWidth));
        const std::string p = "layer" + std::to_string(i);
        l.weight = t.get(p + ".weight", {sz(l.size), sz(l.in)}).data;
        l.bias = t.get(p + ".bias", {sz(l.size)}).data;
        l.decay = t.get(p + ".decay", {sz(l.size)}).data;
        if (w->recurrent) {
            l.recurrent = t.get(p + ".recurrent", {sz(l.size), sz(l.size)}).data;
            l.recurrentT = transposed(l.recurrent, l.size, l.size);
        }
        if (i > 0)
            l.weightT = transposed(l.weight, l.size, l.in);
        prev = l.size;
        w->layers.push_back(std::move(l));
    }
    w->readout = t.get("readout.weight", {sz(info.nOut), sz(prev)}).data;
    w->readoutT = transposed(w->readout, info.nOut, prev);
    w->readoutDecay = t.get("readout.decay", {sz(info.nOut)}).data;
    w->outputBias = t.get("output.bias", {sz(info.nOut)}).data;
    return w;
}

long long SnnWeights::denseMacsPerFrame() const noexcept {
    long long perStep = 0;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const Layer& l = layers[i];
        if (i > 0)
            perStep += static_cast<long long>(l.in) * l.size;
        if (recurrent)
            perStep += static_cast<long long>(l.size) * l.size;
    }
    perStep += static_cast<long long>(layers.back().size) * nOut;
    return static_cast<long long>(layers.front().in) * layers.front().size + perStep * steps;
}

// ---- SNN --------------------------------------------------------------------------

SnnNetwork::SnnNetwork(const ModelFile& model, Kernel kernel) : SnnNetwork(SnnWeights::build(model), kernel) {}

SnnNetwork::SnnNetwork(std::shared_ptr<const SnnWeights> weights, Kernel kernel)
    : w_(std::move(weights)), kernel_(kernel) {
    if (!w_)
        throw std::invalid_argument("SnnNetwork: no weights");
    int widest = 1;
    for (const SnnWeights::Layer& l : w_->layers) {
        State s;
        s.v.assign(sz(l.size), 0.0f);
        s.s.assign(sz(l.size), 0.0f);
        s.active.assign(sz(l.size), 0);
        state_.push_back(std::move(s));
        widest = std::max(widest, l.size);
    }
    u_.assign(sz(w_->nOut), 0.0f);
    first_.assign(sz(w_->layers.front().size), 0.0f);
    cur_.assign(sz(widest), 0.0f);
    tmp_.assign(sz(std::max(widest, w_->nOut)), 0.0f);
    reset();
}

void SnnNetwork::reset() noexcept {
    for (State& s : state_) {
        std::fill(s.v.begin(), s.v.end(), 0.0f);
        std::fill(s.s.begin(), s.s.end(), 0.0f);
        s.count = 0;
    }
    std::fill(u_.begin(), u_.end(), 0.0f);
    spikeCount_ = 0;
    neuronSteps_ = 0;
}

void SnnNetwork::process(const float* x, float* y) noexcept {
    const SnnWeights& w = *w_;
    const std::size_t nOut = sz(w.nOut);
    const bool spikes = kernel_ == Kernel::Spikes;
    // The first layer's input current is the same in every time step of the frame
    kernels::matVec(w.layers[0].weight, w.layers[0].bias, x, first_.data(), sz(w.layers[0].size), sz(w.nIn));

    for (int step = 0; step < w.steps; ++step) {
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            const SnnWeights::Layer& l = w.layers[i];
            State& st = state_[i];
            const std::size_t n = sz(l.size);
            float* cur = cur_.data();
            if (i == 0) {
                std::copy(first_.begin(), first_.end(), cur);
            } else {
                const State& below = state_[i - 1];
                if (spikes) {
                    std::copy(l.bias, l.bias + n, cur);
                    kernels::addActiveRows(l.weightT.data(), below.active.data(), below.count, cur, n);
                } else {
                    kernels::matVec(l.weight, l.bias, below.s.data(), cur, n, sz(l.in));
                }
            }
            if (w.recurrent) {  // own spikes of the step before
                if (spikes) {
                    kernels::addActiveRows(l.recurrentT.data(), st.active.data(), st.count, cur, n);
                } else {
                    kernels::matVec(l.recurrent, nullptr, st.s.data(), tmp_.data(), n, n);
                    for (std::size_t k = 0; k < n; ++k)
                        cur[k] += tmp_[k];
                }
            }
            // LIF: leak and integrate, spike strictly above the threshold, reset by subtraction
            const float threshold = w.threshold;
            float* v = st.v.data();
            float* s = st.s.data();
            int count = 0;
            for (std::size_t k = 0; k < n; ++k) {
                const float vk = cur[k] + l.decay[k] * v[k];
                const bool fire = vk - threshold > 0.0f;
                s[k] = fire ? 1.0f : 0.0f;
                v[k] = fire ? vk - threshold : vk;
                if (fire)
                    st.active[sz(count++)] = static_cast<int>(k);
            }
            st.count = count;
            spikeCount_ += count;
            neuronSteps_ += l.size;
            if (observer_)
                observer_->spikes(static_cast<int>(i), s, l.size);
        }
        // Leaky readout of the last layer's spikes (no bias inside)
        const State& last = state_.back();
        float* r = tmp_.data();
        if (spikes) {
            std::fill(r, r + nOut, 0.0f);
            kernels::addActiveRows(w.readoutT.data(), last.active.data(), last.count, r, nOut);
        } else {
            kernels::matVec(w.readout, nullptr, last.s.data(), r, nOut, sz(w.layers.back().size));
        }
        for (std::size_t k = 0; k < nOut; ++k)
            u_[k] = r[k] + w.readoutDecay[k] * u_[k];
    }
    for (std::size_t k = 0; k < nOut; ++k)
        y[k] = u_[k] + w.outputBias[k];
    applyActivation(w.outActivation, y, nOut);
}

// ---- factory ------------------------------------------------------------------------

std::unique_ptr<FrameNetwork> makeNetwork(const ModelFile& model) {
    const std::string& type = model.info().netType;
    if (type == "mlp")
        return std::make_unique<MlpNetwork>(model);
    if (type == "snn")
        return std::make_unique<SnnNetwork>(model);
    if (type == "tcn")
        throw std::runtime_error("TCN models are not supported by this build yet (net.type \"tcn\")");
    throw FormatError("unknown network type net.type = \"" + type + "\"");
}

}  // namespace denoiser
