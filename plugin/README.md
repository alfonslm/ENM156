# Denoiser plugin (VST3 + ARA)

C++ port of the causal speech denoiser in this repo, to run inside a DAW:

- as a **VST3 effect**, in real time, like `denoiser/stream.py`;
- as an **ARA 2 plugin**, offline on whole audio sources, like `pipeline.enhance()`.

Everything for the plugin lives in this folder (plus
`.github/workflows/plugin.yml`), so syncing the fork with upstream never
conflicts. Nothing outside it is changed.

**Status:** skeleton. The plugin builds as a pass-through VST3 with ARA
enabled; the core has a real FFT and a placeholder API. The DSP port,
inference, resampling, the real-time processor and the ARA document
controller come in later issues.

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
| C++ standard | **C++17** | What JUCE 9 requires as a minimum; nothing in the core needs more. |
| First target OS | **Linux x86-64** for development (it is what was built and tested so far) | CI builds and tests on Linux, Windows and macOS. |
| Plugin formats | **VST3** (with ARA) | AU (macOS) can be added to `FORMATS` in `vst/CMakeLists.txt` when someone can test it. VST 2 is not included and not wanted. |

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
    include/denoiser/   public headers: core.h (placeholder API), fft.h (RealFft)
    src/                sources
  cli/                  denoiser_cli, built as `denoise`: links the core
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
