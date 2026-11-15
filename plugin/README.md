# Denoiser plugin (VST3 + ARA)

C++ port of the causal speech denoiser in this repo, to run inside a DAW:

- as a **VST3 effect**, in real time, like `denoiser/stream.py`;
- as an **ARA 2 plugin**, offline on whole audio sources, like `pipeline.enhance()`.

Everything for the plugin lives in this folder (plus
`.github/workflows/plugin.yml`), so syncing the fork with upstream never
conflicts. Nothing outside it is changed.

**Status:** the core runs MLP and SNN models end to end and matches Python
(golden vectors and full-size models; see "Inference, the per-channel
denoiser and the command-line tool" below), with host-rate resampling for
the plugin and a working `denoise` command-line tool (milestone M1). The
TCN, the real-time processor in the VST3 and the ARA document controller
come in later issues; the plugin itself is still a pass-through.

## Decisions

These are the proposed defaults from the decision issue, taken from its
recommendations. **Merging the pull request that adds this file confirms them.**

| | choice | why |
|---|---|---|
| Plugin framework | **JUCE 9.0.5** with CMake | One codebase for VST3 and ARA (JUCE wraps the ARA SDK; ARA is the hard half). AU and standalone are nearly free later. The raw VST3 + ARA SDKs would mean writing that glue by hand; iPlug2 / DPF have no ARA support to speak of. |
| ARA SDK | **2.3.0** | The version JUCE 9.0.5 names in its `docs/ARA.md`. |
| Inference | **hand-written C++** in `core/` | The networks use a handful of layer types (`Linear`, `Conv1d`, `LayerNorm`, ReLU/sigmoid, the LIF step). Plain loops give no runtime to ship, no allocation or threads on the audio thread by construction, and exact control over the SNN state and an incremental TCN. The default MLP is ~1.25 M multiply-adds per frame (~250 M/s at 200 frames/s): fine on one core. ONNX Runtime and libtorch were rejected as large per-platform dependencies that need work to be real-time safe. |
| How the plugin gets a model | **loaded from a file** the user picks | Written by the export script in [`tools/`](tools/README.md), which documents the format. The repo ships no trained weights (`*.pt` is git-ignored). An embedded default model can be added later. |
| FFT | **KISS FFT 131.2.0**, single precision | Any even size for real transforms: 1280 (= 2^8 · 5, the long window) and 128 work, and so do the other sizes a model file may ask for. All memory is allocated when the plan is made, so transforms are real-time safe. Plain C, no JUCE, BSD-3-Clause. Rejected: JUCE's `dsp::FFT` (powers of two only); pffft (real transforms need a multiple of 32 with SSE, so e.g. n_fft = 200 would fail); pocketfft's C++ header (allocates a scratch buffer inside every transform, so not real-time safe as is). |
| JSON (model file headers) | **hand-written reader** in `core/src/json.cpp` (~300 lines) | Only the small headers of our own files are parsed, once, at load time. The full JSON grammar fits in a few hundred lines with what matters here: a `FormatError` (never UB) on any malformed input, a nesting limit, and numbers read independently of the C locale (a host may set a decimal comma). No dependency to pin, fetch and license; nlohmann/json (MIT, one ~25k-line header) would also work but is far more than needed. |
| C++ standard | **C++17** | What JUCE 9 requires as a minimum; nothing in the core needs more. |
| First target OS | **Linux x86-64** for development (it is what was built and tested so far) | CI builds and tests on Linux, Windows and macOS. |
| Plugin formats | **VST3** (with ARA) | AU (macOS) can be added to `FORMATS` in `vst/CMakeLists.txt` when someone can test it. VST 2 is not included and not wanted. |
| WAV files (command-line tool) | **hand-written reader/writer** in `cli/wav.cpp` (~300 lines) | Reads PCM 8/16/24/32-bit and float 32/64-bit, plain or `WAVE_FORMAT_EXTENSIBLE`, any number of channels; writes the same formats. Only the CLI and its test use it (the plugin gets audio from the host), so a dependency to pin and fetch (dr_wav, libsndfile) wasn't worth it. RF64 (> 4 GB) and compressed WAVs are refused with a message. |
| Dense kernels | **plain loops written to auto-vectorise** (`core/src/kernels.h`), no BLAS | Without `-ffast-math` a compiler won't reorder a float sum, so the matrix-vector product keeps 4 rows × 8 partial sums by hand; measured against 4 other loop forms with GCC 13 and Clang 18 (baseline SSE2, no `-march`). The default MLP runs at 4000–5600 frames/s on one core (20–28× what live use needs), largely limited by streaming its 5 MB of weights through the caches each frame (the same loop does ~2× more multiply-adds per second on a 256 × 256 matrix that fits). A BLAS would add a per-platform dependency for little gain on one frame at a time. |
| SNN kernel | **spike-driven**: add the weight column of each input that spiked | Spikes are 0/1, so the layers after the first, the recurrent weights and the readout add columns instead of multiplying (transposed copies made once per model, shared by all channels). Measured 1.4–3× faster than the dense product at every spike rate tried (0.02–0.45); `SnnNetwork::Kernel::Dense` is kept for comparison. |

## Licences (read from the pinned versions, not summarised from elsewhere)

This repo is AGPL-3.0 (root `LICENSE`).

| dependency | pinned at | licence, as its files say | fits AGPL-3.0? |
|---|---|---|---|
| JUCE | tag `9.0.5`, commit `d60b7a78030661574a615b586d9325ba7a7fa7dd` | `LICENSE.md`: "The JUCE Framework modules are dual-licensed under the AGPLv3 and the commercial JUCE licence." Its `JUCE.spdx.json` gives every module as `AGPL-3.0-only OR LicenseRef-JUCE-Commercial`. The commercial JUCE 9 EULA only applies "if you are not licensing the JUCE Framework modules under the AGPLv3". JUCE's examples are ISC. | Yes, using the AGPLv3 option. Note it is AGPL-3.0-**only**. |
| VST3 SDK | 3.8.0, bundled inside JUCE (`modules/juce_audio_processors_headless/format_types/VST3_SDK`) | `LICENSE.txt` there: "MIT License, Copyright (c) 2025, Steinberg Media Technologies GmbH". JUCE's SBOM agrees (`VST3 SDK 3.8.0 MIT`). | Yes (MIT is permissive). |
| ARA SDK | tag `releases/2.3.0`, commit `0005c532bb03eb210e7bb490329fc6d7ba1792c7` (submodules: ARA_API `65ec5c43`, ARA_Library `d18a6a5e`, ARA_Examples `abd7c8aa`) | `NOTICE.txt`: the ARA API, Library and Examples are "available under the Apache License 2.0"; each of the three submodules has the same Apache-2.0 `LICENSE.txt`, and the headers say "Licensed under the Apache License, Version 2.0". The examples bundle zlib-, MIT- and BSD-2-licensed third-party code, which the plugin does not use. | Yes: Apache-2.0 is compatible with (A)GPLv3 (one way: the combination is distributed under the AGPL). |
| KISS FFT | tag `131.2.0`, commit `7bce4153c6bc8aba2db0e889e576f9d00505cbe1` | `COPYING`: "KISS FFT is provided under: SPDX-License-Identifier: BSD-3-Clause". | Yes (permissive). |

Consequences:

- **Plugin binaries are AGPL-3.0** (only, because of JUCE). Whoever distributes a
  binary must offer its complete corresponding source, this folder included.
  The Apache-2.0 (ARA), MIT (VST3) and BSD-3-Clause (KISS FFT) notices must
  ship with binaries too.
- The core library and command-line tool do not use JUCE, so they carry only
  this repo's licence plus KISS FFT's BSD notice.
- Anyone wanting a closed-source build would need JUCE's commercial licence;
  that is not a goal here.

## Building

Needs CMake ≥ 3.22, a C++17 compiler and git (dependencies are cloned by
git at the pinned versions above into `<build>/_deps`; no archive downloads).

### Core, command-line tool and tests (no JUCE): the fast path

```bash
cmake -S plugin -B plugin/build -G Ninja      # -G is optional
cmake --build plugin/build
ctest --test-dir plugin/build --output-on-failure
```

From a clean build folder this takes about 3 seconds on a 4-core Linux machine
(it clones only KISS FFT). This is the default: `DENOISER_BUILD_PLUGIN` is `OFF`.

### The VST3 plugin

On Linux, first install JUCE's dependencies (Ubuntu 24.04; the plugin is built
without curl, WebKit, OpenGL and JACK, so those are not needed):

```bash
sudo apt-get install -y libasound2-dev libfreetype-dev libfontconfig1-dev libx11-dev libxcomposite-dev libxcursor-dev libxext-dev libxinerama-dev libxrandr-dev libxrender-dev libxi-dev pkg-config
```

Then:

```bash
cmake -S plugin -B plugin/build-plugin -G Ninja -DDENOISER_BUILD_PLUGIN=ON
cmake --build plugin/build-plugin
ctest --test-dir plugin/build-plugin --output-on-failure
```

From a clean folder on a 4-core Linux machine: configure ~50 s (cloning JUCE
and the ARA SDK, building JUCE's `juceaide` helper), build ~3 min. Release
builds use link-time optimisation, so any change relinks the plugin (~2 min);
`-DCMAKE_BUILD_TYPE=Debug` is quicker to iterate with.

The plugin ends up at

```
plugin/build-plugin/vst/DenoiserPlugin_artefacts/Release/VST3/ENM156 Denoiser.vst3/
  Contents/x86_64-linux/ENM156 Denoiser.so
  Contents/Resources/moduleinfo.json
```

To try it in a DAW, copy the `.vst3` folder to `~/.vst3/` (Linux),
`C:\Program Files\Common Files\VST3\` (Windows) or `~/Library/Audio/Plug-Ins/VST3/` (macOS).

On Windows and macOS (multi-config generators) add `--config Release` to the
build and `-C Release` to ctest, as the CI workflow does.

### Options

| option | default | |
|---|---|---|
| `DENOISER_BUILD_PLUGIN` | `OFF` | fetch JUCE + the ARA SDK and build the VST3 |
| `DENOISER_ENABLE_ARA` | `ON` | build the plugin as an ARA 2 plugin |
| `DENOISER_BUILD_CLI` | `ON` | the `denoise` command-line tool |
| `DENOISER_BUILD_TESTS` | `ON` | the tests (and, with the plugin, the host test) |

Offline, or to share one checkout between build folders, point a dependency at
a local clone of the same version: `-DFETCHCONTENT_SOURCE_DIR_JUCE=/path/to/JUCE`,
`-DFETCHCONTENT_SOURCE_DIR_ARA_SDK=/path/to/ARA_SDK` (cloned with `--recursive`),
`-DFETCHCONTENT_SOURCE_DIR_KISSFFT=/path/to/kissfft`.

### Checking the plugin

- `ctest` in a plugin build runs `plugin_passthrough`: a small headless host
  (`vst/HostTest.cpp`, JUCE's VST3 hosting) loads the built `.vst3`, checks
  that it exposes an ARA factory, and that stereo and mono audio in blocks of
  1 to 512 samples come out bit-identical.
- Steinberg's validator (from the VST3 SDK, MIT, tag `v3.8.0_build_66`) can be
  built with `-DSMTG_ENABLE_VST3_PLUGIN_EXAMPLES=OFF -DSMTG_ENABLE_VSTGUI_SUPPORT=OFF`
  and target `validator` (on Linux its hosting examples also need
  `libgtkmm-3.0-dev` to configure); run it as `validator "<path>/ENM156 Denoiser.vst3"`.
- [pluginval](https://github.com/Tracktion/pluginval) v1.0.4 (needs `ladspa-sdk`
  on Linux to build): `pluginval --strictness-level 10 --skip-gui-tests --validate "<path>/ENM156 Denoiser.vst3"`.
  Don't pass it `--vst3validator` with the 3.8 validator: pluginval hands it
  the `.so` inside the bundle, which that validator rejects ("is not a module
  directory").

## Layout

```
plugin/
  CMakeLists.txt        options, C++17, adds the folders below
  cmake/Dependencies.cmake   KISS FFT always; JUCE + ARA SDK with DENOISER_BUILD_PLUGIN
  core/                 denoiser_core (alias denoiser::core): static library, no JUCE
    include/denoiser/   public headers: fft.h (RealFft), model_file.h, features.h, stream.h,
                        offline.h, host_rate.h, resample.h, networks.h (MLP, SNN),
                        denoiser.h (Model, ChannelDenoiser)
    src/                sources
  cli/                  denoiser_cli, built as `denoise`: links the core
    main.cpp            argument handling, offline and --stream modes
    wav.h / wav.cpp     WAV reader/writer
  tests/                core tests, one executable each, run by ctest
    check.h             CHECK / CHECK_NEAR, no test framework
    golden/             golden test vectors written by tools/ (see tools/README.md)
  tools/                Python: checkpoint export and golden vectors (see tools/README.md)
  vst/                  DenoiserPlugin: JUCE AudioProcessor, VST3 + ARA
    PluginProcessor.*   the processor (pass-through for now)
    DocumentController.cpp   the ARA entry point (createARAFactory), empty for now
    HostTest.cpp        headless host for the plugin_passthrough test
```

## Working on it

- **New core source**: add it to `add_library(denoiser_core ...)` in
  `core/CMakeLists.txt`; public headers go in `core/include/denoiser/`. The core
  must not include JUCE or host headers.
- **New test**: write `tests/test_<name>.cpp` with a `main()` that uses
  `CHECK`/`CHECK_NEAR` and ends with `return check::finish();`, then add
  `<name>` to `DENOISER_TESTS` in `tests/CMakeLists.txt`. Golden files are in
  `tests/golden/`.
- **New CLI source**: add it to `add_executable(denoiser_cli ...)` in `cli/CMakeLists.txt`.
- **New plugin source**: add it to `target_sources(DenoiserPlugin ...)` in `vst/CMakeLists.txt`.

### FFT

`#include "denoiser/fft.h"`; it follows `torch.fft.rfft` / `irfft`:

```cpp
denoiser::RealFft fft(n);                    // n even; allocates (do it when a model loads)
std::vector<std::complex<float>> spec(fft.bins());   // n / 2 + 1
fft.forward(frame, spec.data());             // n floats -> bins, unnormalised
fft.inverse(spec.data(), frame);             // bins -> n floats, scaled by 1/n
```

`forward` and `inverse` don't allocate, lock or throw. A `RealFft` keeps
scratch memory, so use one per thread. `tests/test_fft.cpp` checks sizes 1280,
128, 160 and 2 against a direct DFT in double precision (max abs error 8e-6
for 1280 on random input in [-1, 1]; the test allows 1e-3) and the round trip
inverse(forward(x)) (max abs error 3e-7; the test allows 1e-5).

## Host sample rate, the band above 8 kHz, and stereo

The model runs at 16 kHz mono; hosts run at 44.1–192 kHz, mostly stereo.
`core/include/denoiser/host_rate.h` (`HostRateAdapter`) wraps the 16 kHz
hop-by-hop processor for the plugin; `core/include/denoiser/resample.h` has
the resampler it is built on. Tests: `tests/test_resample.cpp`,
`tests/test_host_rate.cpp`.

### Decisions

| | choice | trade-off |
|---|---|---|
| Band above 8 kHz | **A: output only the 16 kHz result**, resampled back (`HighBand::Drop`, the default). Nothing above ~7.5 kHz comes out. | Simplest, and what the model was trained for, but speech sounds duller than the dry signal. The UI should say so. |
| | **B is implemented too**, behind `HighBand::PassThrough`: the band the model can't see passes through, delayed to match. | Full bandwidth, but noise above 8 kHz stays. +50% adapter cost (one more upsampler per channel). It is built as *input delayed by the latency − the input's own unprocessed round trip*, so with a pass-through model the output is exactly the delayed input (bit for bit), and the crossover (7–8 kHz) is complementary by construction. |
| | C (scale the high band by the model's top bins) and D (train at 48 kHz) are not done; D belongs upstream. | |
| Channels | **One independent 16 kHz processor per channel** (`setProcessor(channel, ...)`). | CPU × channels. Independent masks can shift the stereo image; mid/side or a shared mask are possible later. |
| Host rates | **Integers from 16 to 384 kHz whose ratio to 16 kHz needs at most 1000 phases**: 22.05, 24, 32, 44.1, 48, 88.2, 96, 176.4, 192, 352.8, 384 kHz (`isSupportedHostRate`). At **16 kHz the resamplers are bypassed** (samples copied, bit-exact). | **Below 16 kHz (8, 11.025 kHz…) is refused**: the model needs the band up to 8 kHz. Odd rates such as 47999 Hz are refused because their table would be millions of taps. `prepare()` throws `std::invalid_argument`; the plugin should then pass audio through and say why. |
| Resampler | **Hand-written polyphase Kaiser-windowed sinc**, linear phase, one prototype per rate on the grid `F = lcm` of the two rates (48 kHz: ÷3; 44.1 kHz: ×160 ÷441). No dependency. | Linear phase costs latency (≈ 3 ms each way); a minimum-phase filter would roughly halve that but would make option B's split inexact. |
| Latency | One function, `hostLatencySamples(hostRate, hop, processorDelay)`. Exact and constant, independent of block sizes. | The adapter runs sample by sample on a fixed schedule, which is what makes it exact; at 0.3 % of a core that costs nothing worth saving. |

### Filter (same at every rate)

- Passband 0–7 kHz, stopband from 8 kHz, cutoff 7.5 kHz. Nothing aliases into
  or images out of the band below 7 kHz; 7–8 kHz is the transition band
  (−6 dB per pass at 7.5 kHz, so −12 dB for the round trip).
- Length `2 · M · 48 + 1` taps on the grid, i.e. exactly **48 samples at
  16 kHz (3 ms) of delay in each direction** (`kResampleHalfLength`); Kaiser
  β from its estimate for that length (≈ 94 dB).
- Measured from the prototype (zero-padded FFT, `test_resample`):
  passband ripple **±0.00018 dB**, stopband attenuation **≥ 94.0 dB** (94.0–94.3
  depending on the rate); polyphase phases' DC gain within 2e-6 of 1.
- Taps per output: 48 kHz down 289 (per 16 kHz sample), up 97 (per 48 kHz
  sample); 44.1 kHz down 265 × 160 phases, up 97 × 441 phases (one shared
  42 337-tap table, 170 kB per direction, shared by all channels).

### Latency

`hostLatencySamples` = `floor((2·48 + processorDelay + hop − 1) · hostRate / 16000)`
host samples, or `processorDelay + hop − 1` at 16 kHz:

- 2 · 48: the two resamplers;
- `processorDelay`: the 16 kHz processor's own delay (`n_fft − hop` for the
  streaming denoiser, i.e. 48);
- `hop − 1`: regrouping into hops. Each 16 kHz output sample leaves as soon as
  the 16 kHz input sample with the same index has arrived; that needs `hop − 1`
  samples of slack, not `hop`.
- The fraction that `floor` drops is absorbed by sampling the upsampler up
  to one host sample late (on the fine grid), so the number is exact.

With the default model (`n_fft` 128, `hop` 80):

| host rate | latency (samples) | ms |
|---|---|---|
| 16 kHz (bypass) | 127 | 7.94 |
| 22.05 kHz | 307 | 13.92 |
| 32 kHz | 446 | 13.94 |
| 44.1 kHz | 614 | 13.92 |
| 48 kHz | 669 | 13.94 |
| 88.2 kHz | 1229 | 13.93 |
| 96 kHz | 1338 | 13.94 |
| 176.4 kHz | 2458 | 13.93 |
| 192 kHz | 2676 | 13.94 |

`denoiser/stream.py` calls its latency `n_fft` (128) because it counts a whole
hop of buffering; sample by sample it is `n_fft − 1`, which is what the plugin
reports at 16 kHz.

### Measured (`test_host_rate`, Release, Intel Xeon @ 2.3 GHz)

- Impulses at several positions and a broadband signal (40 sines up to 6 kHz),
  pass-through stand-ins (identity with delay 48 and the built-in null
  processor; also hop 160 / delay 96, hop 64 / delay 0, hop 1 / delay 3), at
  every supported rate from 16 to 192 kHz, for blocks of 1, 7, 64, 80, 333
  (in place), 512, 4096 and random sizes 0–700: peaks land exactly at the
  reported latency, the broadband output matches the input delayed by it to
  **−100 to −105 dB** (compared one sample off: −7 to −20 dB when resampling), and the output is
  bit-identical for every block pattern.
- Round trip with a pass-through model, option A, error relative to the
  delayed input, per sine: 48 kHz **≤ −98.6 dB up to 6.5 kHz, −87.8 dB at
  7 kHz**; 44.1 kHz ≤ −99.3 dB up to 6.5 kHz, −87.8 dB at 7 kHz (the test
  allows −80 dB below 7 kHz). Level at 7.5 kHz −12.0 dB; at 8 kHz and above
  **≤ −106 dB** (−106.6 dB at 8.5 kHz is the worst; the test requires
  ≤ −85 dB).
- Option B with a pass-through model: exactly the delayed input (error 0). With
  a model that outputs silence: below 7 kHz ≤ −87.8 dB, 8 kHz and above passes
  at 0.00 dB, matching the delayed input to ≤ −115 dB.
- 16 kHz: output is the hop-by-hop result of the same processor delayed by
  `hop − 1`, bit for bit.
- `reset()` reproduces the first run bit for bit; no heap allocation in
  `process`, `processChannel`, `reset`, `setProcessor` or
  `processClipAligned` (counted by replacing `operator new`); stereo channels
  equal their separate mono runs.
- Cost of the adapter alone, 48 kHz stereo, blocks of 512: **3.2 ms per
  second of audio** (0.3 % of one core) for option A, 4.8 ms for B; 44.1 kHz
  about the same.
- The tests also pass under AddressSanitizer + UBSan (Debug).

## Inference, the per-channel denoiser and the command-line tool

`core/include/denoiser/networks.h` (MLP, SNN), `denoiser.h` (`Model`,
`ChannelDenoiser`), `cli/` (`denoise`). TCN models are refused with a clear
error until the TCN lands (`makeNetwork` has one line to change).

### Using it (plugin, ARA, tools)

```cpp
#include "denoiser/denoiser.h"
// Load once, off the audio thread (allocates; throws denoiser::FormatError for a
// malformed file, std::runtime_error for a network this build can't run):
std::shared_ptr<const denoiser::Model> model = denoiser::Model::fromMemory(bytes, size);
model->describe();      // "mlp 130 -> 1024 -> 1024 -> 65, magnitude; n_fft 128, hop 80, ..."
model->info().realtime; // false: centred frames, offline (ARA) only

// Real time: one ChannelDenoiser per channel, all sharing the model's weights
std::vector<std::unique_ptr<denoiser::ChannelDenoiser>> ch;   // throws std::invalid_argument if !realtime
for (int c = 0; c < channels; ++c) ch.push_back(std::make_unique<denoiser::ChannelDenoiser>(model));
denoiser::HostRateAdapter adapter;
adapter.prepare(ch[0]->hostRateConfig(hostRate, channels));   // HighBand::Drop by default
for (int c = 0; c < channels; ++c) adapter.setProcessor(c, ch[c].get());
setLatencySamples(adapter.latencySamples());
// audio thread: adapter.process(in, out, n); adapter.reset();   (no allocation, locks or exceptions)

// Offline at the model's rate (either framing; ARA analysis after resampling, the CLI):
auto net = model->makeNetwork();
std::vector<float> y = denoiser::denoiseClip(model->info(), *net, x, n);   // same length, no delay
// ... or the plugin's own path at the host rate, latency removed:
denoiser::processClipAligned(adapter, channel, in, out, n);
```

- **Threads.** A `Model` is immutable once loaded: share one
  `shared_ptr<const Model>` between channels, threads and the ARA analysis;
  the weights are not copied per channel (an SNN's transposed copies are made
  once, in the `Model`). A `ChannelDenoiser` or network has state and scratch
  memory: one thread at a time. `ChannelDenoiser` keeps its `Model` alive, so
  a model can be swapped by building new denoisers off the audio thread and
  swapping pointers; release the old ones (and with them possibly the last
  reference to the old `Model`, which frees its weights) off the audio thread.
- **Real-time safety.** `ChannelDenoiser::processHop` / `reset`,
  `FrameNetwork::process` / `reset` and the adapter calls don't allocate, lock
  or throw (`tests/test_networks_alloc.cpp` counts allocations).
- **Pre-roll.** `ChannelDenoiser` mutes the first `n_fft - hop` samples after
  a reset (time before the input; they can be ~±280 with untrained models);
  `setMutePreRoll(false)` reproduces `StreamingDenoiser.process` exactly.
- **Latency.** `processorDelay()` = `n_fft - hop`; at 16 kHz the plugin's
  latency is `n_fft - 1` (127), see "Latency" above.
- Lower level: `makeNetwork(const ModelFile&)`, `MlpNetwork`, `SnnNetwork`
  (`SnnWeights::build` to share weights, a `SpikeObserver` for tests). These
  read the `ModelFile`'s tensors in place, so the file must outlive them.

### The command-line tool

```
denoise [--stream] [--high-band=drop|pass] [--format=same|pcm16|pcm24|pcm32|float32|float64] [--float] [-q]
        model.bin in.wav out.wav
```

- Default: offline, `denoiseClip` per channel (any number of channels); the
  file must be at the model's rate (16 kHz). Centred models work here.
- `--stream`: each channel as the plugin runs it (`ChannelDenoiser` behind
  `HostRateAdapter` at the file's rate: 16 kHz or 22.05–384 kHz), latency
  removed with `processClipAligned`. Centred models are refused with a message.
- Output: same length, aligned, by default in the input's sample format
  (`--float` for 32-bit float); PCM is rounded and clipped, with a warning.
- Exit status 0; 1 for errors (missing, unreadable or malformed file, wrong or
  unsupported rate, a TCN model, a centred model with `--stream`); 2 for bad arguments.
- File names go through `std::filesystem::path` and, on Windows with MSVC,
  `wmain`, so non-ASCII paths should work there too (not tested on Windows);
  the tool reads the model file itself and calls `Model::fromMemory`, not the
  narrow-path `ModelFile::load`.

### Tests (ctest; no Python)

| test | what |
|---|---|
| `networks` | every golden MLP / SNN case: `pred` from the golden `x` (1e-5); SNN spikes vs `snn.spikes*` exactly, spike rates vs the header, both kernels with identical spikes; `reset()` bit-exact; missing tensors and `tcn` give clear errors |
| `denoiser_golden` | from the golden `input`: `denoiseClip` vs `offline` (causal and centred), `ChannelDenoiser` vs `streamed` (1e-5, pre-roll 1e-4; muted pre-roll exactly 0); the 16 kHz host path (below) |
| `networks_alloc` | no heap allocation in the networks, `ChannelDenoiser`, and the adapter at 16 and 48 kHz stereo; `reset()` reproduces a fresh run bit for bit |
| `cli` | runs the built `denoise` on float and 16-bit WAVs made from the golden inputs (offline and `--stream`, stereo) and checks the files; 15 runs with their expected exit status; `--stream` at 48 kHz; the WAV code on its own. `DENOISE_LAUNCHER` runs the tool through e.g. `qemu-aarch64` |

Measured (Linux x86-64, GCC 13 Release; error = max |c++ − golden| / max(1, max |golden|)):

| case | pred | offline | streamed | pre-roll |
|---|---|---|---|---|
| mlp_magnitude_default | 6.0e-08 | 3.9e-08 | 4.1e-08 | 1.6e-06 |
| mlp_magnitude_short | 3.7e-08 | 2.3e-08 | 2.5e-08 | 1.0e-07 |
| mlp_mask_default | 6.0e-08 | 3.5e-08 | 3.5e-08 | 1.7e-07 |
| mlp_complex_default | 1.3e-07 | 4.3e-08 | 5.0e-08 | 5.4e-06 |
| mlp_complex_plain | 9.6e-08 | 9.4e-08 | 8.9e-08 | 2.3e-06 |
| mlp_complex_long4 | 1.0e-07 | 6.7e-08 | 6.7e-08 | 1.8e-06 |
| snn_complex_default | 0 | 9.7e-08 | 8.2e-08 | 8.0e-07 |
| snn_complex_recurrent | 0 | 5.2e-08 | 5.2e-08 | 1.0e-06 |
| snn_complex_steps3 | 0 | 4.5e-08 | 4.5e-08 | 6.3e-07 |
| snn_mask_recurrent_steps2 | 1.2e-07 | 5.2e-08 | 4.5e-08 | 2.8e-06 |
| snn_magnitude_nodecay | 0 | 1.0e-06 | 1.0e-06 | 4.5e-07 |
| offline_mlp_mask_centred | 6.0e-08 | 3.4e-08 | – | – |

SNN `pred` with the spike kernel is bit-identical to the golden file in 4 of
5 cases; the dense kernel is within 1.8e-7. Spikes: 0 mismatches (stored for
8 frames of `snn_complex_default` and 12 steps of `snn_complex_steps3`; in all
frames of all 5 cases the two kernels spike identically and match the
header's spike rates). Tolerances enforced: those of `tools/README.md`.

**16 kHz host path = CLI.** A `ChannelDenoiser` behind `HostRateAdapter` at
16 kHz gives, for every causal golden case and for a 0.8 s clip: with
`processClipAligned`, exactly `denoiseClip`'s output over the whole clip, bit
for bit from the first to the last sample (the adapter feeds
`latencySamples()` zeros after the clip, `denoiseClip` the zeros its last
frames need; the samples kept depend only on input both have); in real-time
use with random blocks of 0–700 samples, exactly 0 for the first
`latencySamples()` (`n_fft − 1`) samples, then `denoiseClip`'s output bit for
bit, shifted by that latency. Both sides run the same compiled library code
(`StreamProcessor::process` and the network), which is why bit equality is
a fair test on every compiler and CPU, arm64 with fused multiply-adds too.

Also run: all tests under AddressSanitizer + UBSan (Debug); with Clang 18
(`-Wshadow -Wconversion`: no warnings in the new files); cross-compiled for
arm64 with Clang (`-ffp-contract=on`) under `qemu-aarch64`, where the new
tests pass too (the CLI test with `DENOISE_LAUNCHER="qemu-aarch64 -L
/usr/aarch64-linux-gnu"`).

### Full-size models against Python: `tools/check_cli.py`

```bash
.venv/bin/python plugin/tools/check_cli.py plugin/build/cli/denoise
```

Untrained models at config.toml's sizes (MLP 1024 × 1024, SNN 256 × 3; the
SNN also with threshold 0.1, which gives ~40 % spikes like a working network
instead of ~1–2 %), exported, run by the tool on a 12 s stereo float WAV
(two different synthetic clips), offline and `--stream`, compared with
`pipeline.enhance`. Measured: worst error **1.2e-7** relative (MLP ≤ 9.7e-8,
SNN ≤ 1.2e-7), offline and streamed alike, with one expected exception: in
the recurrent SNN with threshold 0.1, one clip has a membrane potential
**8.9e-8** from the threshold, a spike flips and that channel differs by
1.7e-2. A threshold device can always do that on some input, between any two
float implementations (PyTorch on two CPUs too); the script computes how close
each run comes to the threshold and reports, but doesn't count, runs closer
than 1e-5. Expect trained SNNs to match Python closely but not bit for bit on
long material.

### Speed (one core, Intel Xeon @ 2.3 GHz, Release, baseline SSE2; `tests/bench_networks model.bin ...`)

Ranges over runs on a shared machine (GCC 13 and Clang 18 similar). Live use
needs 200 frames/s per channel (hop 80).

| model (config.toml sizes, untrained) | network alone | `ChannelDenoiser` (STFT + network) | 48 kHz stereo via the adapter |
|---|---|---|---|
| MLP 130 → 1024 → 1024 → 65 (magnitude) | 4000–5600 frames/s (**20–28×** the need) | 4100–5200 hops/s | 8–10 % of a core |
| MLP 260 → 1024 → 1024 → 130 (complex_mask) | 3600–4900 frames/s (18–24×) | 3400–4400 hops/s | 9–15 % |
| SNN 260 → 256 → 256 → 256 → 130, threshold 1.0 (2 % spikes) | 120 000–170 000 frames/s (dense kernel 40 000–63 000) | 42 000–48 000 hops/s | 1.3 % |
| SNN, threshold 0.1 (40 % spikes) | 55 000–78 000 frames/s (dense 36 000–56 000) | 30 000–34 000 hops/s | 1.6 % |
| SNN, recurrent, threshold 0.1 (45 %) | 40 000–50 000 frames/s (dense 22 000–29 000) | 29 000–32 000 hops/s | 1.7 % |

The command-line tool on a 12 s stereo file: MLP 1.4–1.6 s (7.3–8.4× faster
than real time for both channels together, ~15× per channel), SNN 0.08–0.25 s
(46–158×). Before the matrix-vector loop was tuned the MLP ran at ~1900
frames/s, still ~10× the need: roughly what a compiler that vectorises it
badly (MSVC was not measured) might give.

### PESQ / STOI on a trained checkpoint (to be run by the owner)

There is no trained checkpoint or dataset in the development environment, so
this check was only dry-run: on a fake 6-clip test split of synthetic clips
with an untrained default MLP checkpoint, the tables of `eval_cli.py`
(offline and `--stream`) and of evaluate.py's scoring came out identical.
With your own `model.pt` and VoiceBank-DEMAND as configured in
`config.local.toml`:

```bash
cmake -S plugin -B plugin/build -G Ninja && cmake --build plugin/build
.venv/bin/python plugin/tools/export.py runs/default/model.pt out/                    # -> out/model.bin
.venv/bin/python plugin/tools/eval_cli.py plugin/build/cli/denoise out/model.bin --num-clips 100
.venv/bin/python plugin/tools/eval_cli.py plugin/build/cli/denoise out/model.bin --num-clips 100 --stream
.venv/bin/python evaluate.py --model runs/default/model.pt --num-clips 100 --set data.device=cpu
```

`eval_cli.py` scores the tool's output with evaluate.py's own
`evaluate_model` / `print_results` (same clips, same rows including
"clean->model"); PESQ / STOI / ESTOI should equal evaluate.py's to the
printed 3 decimals (`data.device=cpu`, because a GPU's sums differ in the last
bits). An SNN may differ in the last digit if a spike flips (above). Leave
out `--num-clips 100` for the whole test set (824 clips; the tool is started
twice per clip).
