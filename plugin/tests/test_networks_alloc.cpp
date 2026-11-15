// No heap allocation on the audio path: for every golden MLP / SNN model,
// counted with alloc_hooks.h while armed:
//   - FrameNetwork::process and reset (MLP; SNN with both kernels)
//   - ChannelDenoiser::processHop and reset
//   - HostRateAdapter::process / reset / processClipAligned with
//     ChannelDenoisers as the processors, at 16 and 48 kHz, stereo
// Also: after other input, reset() reproduces a fresh ChannelDenoiser's
// output bit for bit, and so does HostRateAdapter::reset() at 48 kHz.
#include "alloc_hooks.h"

#include "check.h"
#include "golden.h"

#include "denoiser/denoiser.h"
#include "denoiser/networks.h"

#include <cstdint>
#include <cstring>

using namespace denoiser;

namespace {

std::vector<float> noise(std::size_t n, std::uint32_t seed) {
    std::vector<float> v(n);
    for (float& s : v) {
        seed = seed * 1664525u + 1013904223u;
        s = (float(seed >> 8) / float(1u << 24) - 0.5f) * 0.6f;
    }
    return v;
}

void checkCase(const std::string& name) {
    const std::vector<char> bytes = golden::readBytes(golden::dir() + "/" + name + ".model.bin");
    const std::shared_ptr<const Model> model = Model::fromMemory(bytes.data(), bytes.size());
    const ModelInfo& info = model->info();
    const golden::Case c = golden::load(name);
    const Tensor& x = c.get("x");
    std::vector<float> y(std::size_t(info.nOut));

    std::vector<std::unique_ptr<FrameNetwork>> nets;
    nets.push_back(model->makeNetwork());
    if (info.netType == "snn")
        nets.push_back(std::make_unique<SnnNetwork>(model->file(), SnnNetwork::Kernel::Dense));
    long netAllocs = 0;
    for (auto& net : nets)
        netAllocs += alloc::count([&] {
            for (int pass = 0; pass < 2; ++pass) {
                for (std::size_t t = 0; t < x.shape[0]; ++t)
                    net->process(x.data + t * x.shape[1], y.data());
                net->reset();
            }
        });
    CHECK(netAllocs == 0);
    std::printf("%-28s network: %ld", name.c_str(), netAllocs);
    if (!info.realtime) {
        std::printf("\n");
        return;
    }

    // ChannelDenoiser, and reset() reproducing a fresh one
    const std::size_t hop = std::size_t(info.hop);
    const std::vector<float> signal = noise(40 * hop, 5), junk = noise(17 * hop, 9);
    ChannelDenoiser fresh(model), used(model);
    std::vector<float> first(signal.size()), second(signal.size()), out(hop);
    const long chAllocs = alloc::count([&] {
        for (std::size_t k = 0; k < 40; ++k)
            fresh.processHop(signal.data() + k * hop, first.data() + k * hop, int(hop));
        for (std::size_t k = 0; k < 17; ++k)
            used.processHop(junk.data() + k * hop, out.data(), int(hop));
        used.reset();
        for (std::size_t k = 0; k < 40; ++k)
            used.processHop(signal.data() + k * hop, second.data() + k * hop, int(hop));
    });
    CHECK(chAllocs == 0);
    CHECK(std::memcmp(first.data(), second.data(), first.size() * sizeof(float)) == 0);
    std::printf(", ChannelDenoiser: %ld", chAllocs);

    // behind the adapter, stereo
    for (int rate : {16000, 48000}) {
        ChannelDenoiser left(model), right(model);
        HostRateAdapter adapter;
        adapter.prepare(left.hostRateConfig(rate, 2));
        adapter.setProcessor(0, &left);
        adapter.setProcessor(1, &right);
        const std::vector<float> l = noise(std::size_t(rate) / 4, 1), r = noise(std::size_t(rate) / 4, 2);
        std::vector<float> lo(l.size()), ro(r.size()), lo2(l.size()), ro2(r.size()), aligned(l.size());
        const long adAllocs = alloc::count([&] {
            for (int pass = 0; pass < 2; ++pass) {
                float* outs[2] = {pass ? lo2.data() : lo.data(), pass ? ro2.data() : ro.data()};
                for (std::size_t pos = 0; pos < l.size(); pos += 333) {
                    const int n = int(std::min<std::size_t>(333, l.size() - pos));
                    const float* ins[2] = {l.data() + pos, r.data() + pos};
                    float* o[2] = {outs[0] + pos, outs[1] + pos};
                    adapter.process(ins, o, n);
                }
                adapter.reset();
            }
            processClipAligned(adapter, 0, l.data(), aligned.data(), std::int64_t(l.size()));
        });
        CHECK(adAllocs == 0);
        CHECK(std::memcmp(lo.data(), lo2.data(), lo.size() * sizeof(float)) == 0);
        CHECK(std::memcmp(ro.data(), ro2.data(), ro.size() * sizeof(float)) == 0);
        CHECK(adapter.underruns() == 0);
        std::printf(", adapter %d Hz: %ld", rate, adAllocs);
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    golden::init(argc, argv);
    std::printf("malloc family hooked: %s; allocations counted in per-frame / per-block calls:\n",
                HOOK_MALLOC ? "yes" : "no (operator new only)");
    static int* volatile probe = nullptr;
    CHECK(alloc::count([] { probe = new int(1); }) >= 1);  // the counter works
    delete probe;
    int cases = 0;
    for (const std::string& name : golden::caseNames()) {
        const golden::Case c = golden::load(name);
        if (c.info().netType != "mlp" && c.info().netType != "snn")
            continue;
        checkCase(name);
        ++cases;
    }
    CHECK(cases == 12);
    return check::finish();
}
