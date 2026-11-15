# Denoiser plugin (VST3 + ARA)

C++ port of the causal speech denoiser in this repo, to run inside a DAW:

- as a **VST3 effect**, in real time, like `denoiser/stream.py`;
- as an **ARA 2 plugin**, offline on whole audio sources, like `pipeline.enhance()`.

Everything for the plugin lives in this folder (plus
`.github/workflows/plugin.yml`), so syncing the fork with upstream never
conflicts. Nothing outside it is changed.

**Status:** the core runs MLP, SNN and TCN models end to end and matches Python
(golden vectors and full-size models; see "Inference, the per-channel
denoiser and the command-line tool" below), with host-rate resampling and a
working `denoise` command-line tool (milestone M1). The VST3 plugin runs the
denoiser in real time (see "The VST3 effect" below): model file chosen in its
window and saved in the session, mix, latency-compensated bypass, high-band
switch, exact latency, no allocation or locks on the audio thread, tested
headlessly through its VST3 interface and checked with Steinberg's validator
and pluginval (strictness 10). Bound to an ARA document (see "ARA: whole
clips" below), the same binary denoises whole audio sources offline, on a
background thread, at the source's rate, with causal and centred
(non-causal) models, and plays them back sample-aligned with no latency
(milestone M3); tested through the real ARA interface by a headless ARA host
built on JUCE's ARA hosting classes, and with the ARA SDK's `ARATestHost`.
Neither mode has been tried in a DAW yet (there is none in the development
environment); checklists for the owner are below.

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
| `DENOISER_BUILD_TESTS` | `ON` | the tests (and, with the plugin, the plugin tests) |
| `DENOISER_PROCESSOR_TEST_COUNT_ALLOCATIONS` | `ON` | plugin builds: count allocations in `denoiser_processor_test` (turn off for sanitizer builds) |
| `DENOISER_DEFAULT_MODEL` | empty | plugin builds, **for validation only**: a model file every new instance loads, so that pluginval / Steinberg's validator (which can't restore a state) test the denoiser itself |

Offline, or to share one checkout between build folders, point a dependency at
a local clone of the same version: `-DFETCHCONTENT_SOURCE_DIR_JUCE=/path/to/JUCE`,
`-DFETCHCONTENT_SOURCE_DIR_ARA_SDK=/path/to/ARA_SDK` (cloned with `--recursive`),
`-DFETCHCONTENT_SOURCE_DIR_KISSFFT=/path/to/kissfft`.

### Checking the plugin

- `ctest` in a plugin build also runs `plugin_vst3` (a small headless host,
  `vst/HostTest.cpp`, loads the built `.vst3` with JUCE's VST3 hosting and
  checks the denoiser through it) and `plugin_processor_*` (the processor
  class without a host); see "The VST3 effect", "Tests". With ARA (the
  default) also `plugin_ara` (a headless ARA host, `vst/AraHostTest.cpp`) and
  `plugin_ara_engine` (the ARA engine without JUCE); see "ARA: whole clips",
  "Tests".
- The ARA SDK's own `ARATestHost` (in `ARA_Examples`) can be pointed at the
  plugin; see "ARA: whole clips", "What was run".
- Steinberg's validator (from the VST3 SDK, MIT, tag `v3.8.0_build_66`) can be
  built with `-DSMTG_ENABLE_VST3_PLUGIN_EXAMPLES=OFF -DSMTG_ENABLE_VSTGUI_SUPPORT=OFF`
  and target `validator` (on Linux its hosting examples also need
  `libgtkmm-3.0-dev` to configure); run it as `validator "<path>/ENM156 Denoiser.vst3"`.
- [pluginval](https://github.com/Tracktion/pluginval) v1.0.4 (needs `ladspa-sdk`
  on Linux to build): `pluginval --strictness-level 10 --skip-gui-tests --validate "<path>/ENM156 Denoiser.vst3"`.
  Don't pass it `--vst3validator` with the 3.8 validator: pluginval hands it
  the `.so` inside the bundle, which that validator rejects ("is not a module
  directory").

## The VST3 effect (real time)

`vst/`: `PluginProcessor.*` (the JUCE `AudioProcessor`), `RealtimeEngine.*`
(the processing state, no JUCE), `PluginEditor.*` (the UI). The plugin runs
the denoiser live, like `denoiser/stream.py`, at the host's rate, mono or
stereo (one independent denoiser per channel).

### Loading a model

Open the plugin's window and press **Load model...**, then pick a model file
written by [`tools/export.py`](tools/README.md) (`model.bin`). The window shows:

- the file's path, and one line describing the model (`Model::describe()`,
  e.g. `mlp 130 -> 1024 -> 1024 -> 65, magnitude; n_fft 128, hop 80, long window 1280; 16000 Hz; real-time`);
- the latency in samples and ms at the host's rate;
- in orange, why audio passes through, if it does.

**Unload** goes back to no model. Loading reads and builds the model on the
thread that asks (the UI thread): 6–8 ms for a config.toml-sized MLP, 1–3 ms
for an SNN, 35–47 ms for a TCN (measured, see "CPU" below). The audio thread
never waits for it.

Audio passes through unchanged (delayed by the latency, see below) and the
window says why when:

| situation | message (shortened) |
|---|---|
| no model chosen | "No model loaded: audio passes through ..." |
| the file is missing (e.g. a session from another computer) | "Model file not found: <path>. ... choose the file again." |
| the file is not a valid model file | "Could not load <name>: <reason>." |
| the model uses centred frames (`realtime: false`) | "... can't run in real time; audio passes through. It can be used in ARA mode." |
| the host rate is unsupported (below 16 kHz, or e.g. 47999 Hz) | "The host sample rate (... Hz) is not supported ..." (latency 0) |

The path is remembered (and saved) even when loading fails, so that a
session opened where the file does exist works.

### Parameters (automatable, in the host's parameter list)

| parameter | values | |
|---|---|---|
| **Mix** | 0–100 % (default 100) | dry/wet. The dry signal is the input delayed by exactly the reported latency, so the two are aligned. 0 % gives the delayed input bit for bit; 100 % the denoiser's output bit for bit. 20 ms linear ramps. |
| **Bypass** | off / on | Also the host's own bypass (`getBypassParameter()`). Latency-compensated: bypassed, the output is the input delayed by the same latency, bit for bit, so toggling it doesn't shift the audio. The denoiser keeps running while bypassed (constant CPU, no restart transient when it is switched back on). 20 ms ramp. |
| **High band** | Drop / Pass through | What happens above 8 kHz, which the model never sees (see "Host sample rate, the band above 8 kHz" below). **Drop** (default): the 16 kHz result only, nothing above ~7.5 kHz. **Pass through**: the band the 16 kHz round trip removes is added back from the delayed input (noise there stays). No effect at a 16 kHz host rate. 20 ms ramp. |

The high band is a smooth parameter, not a re-prepare: each engine runs a
second resampling round trip with no model (the core's own pass-through
processor) and adds `dry − that` scaled by the parameter. That is exactly
option B of the core (`HighBand::PassThrough`, input delayed by the latency
minus its unprocessed round trip) computed outside the adapter; the plugin
matches the core's `HighBand::PassThrough` to 3e-8 (`plugin_processor_core`).
It costs one more resampler pair per channel, ~0.3 % of a core at 48 kHz
stereo, also while the high band is dropped.

### Latency

The plugin reports `hostLatencySamples(hostRate, hop, n_fft − hop)` (see
"Latency" in the host-rate section): 669 samples at 48 kHz, 614 at 44.1 kHz,
1338 at 96 kHz, 127 at 16 kHz for config.toml's STFT (n_fft 128, hop 80),
13.9 ms (7.9 ms at 16 kHz).

| state | latency reported |
|---|---|
| a model runs | that formula for the model's hop and n_fft |
| no model, a missing / broken / offline-only model | the same formula for the **last shape that ran**: config.toml's (n_fft 128, hop 80) in a new instance, else the hop / n_fft of the last model that ran in this instance, also saved in the state. Audio is the input delayed by that latency. So loading a model of that shape (the usual case) doesn't change the host's compensation, and a session whose model file is missing keeps the latency it was mixed with. |
| a model with a different hop / n_fft is loaded | changes; the host is told by the loading thread (`setLatencySamples`, which JUCE's VST3 wrapper turns into `restartComponent(kLatencyChanged)` on the message thread) when the new engine is handed to the audio thread |
| the host rate changes | `prepareToPlay` rebuilds everything and reports it |
| unsupported host rate | 0, audio copied unchanged |
| bound to an ARA document | 0 (ARA playback is aligned; JUCE's wrapper requires 0) |

### What happens at a model change (swap)

A model is loaded and a complete processing state (`Engine`: the model, one
`ChannelDenoiser` per channel, the `HostRateAdapter`, the dry delay line and
scratch buffers, for the current rate and channel count) is built on the
loading thread and handed to the audio thread through one atomic pointer.
At the start of its next block the audio thread takes it and:

- keeps the **old engine running and audible** until the new one has been fed
  for its latency plus its widest analysis window (so its output depends only
  on real input: 669 + 3840 samples at 48 kHz for config.toml's long window),
- then **crossfades linearly over 10 ms** to the new one,
- then retires the old one to a slot that a control thread frees (the next
  control call or a 10 Hz timer).

So there is no gap or click, CPU doubles for ~100 ms, and from the end of the
crossfade the output is exactly that of a fresh instance of the new model
started at the swap (`plugin_processor_core` checks this to 7e-9 and that the
crossfade is no steeper than the signals themselves). A TCN or SNN still
settles over its own context after that (a TCN looks back 127 frames, 0.64
s), as after any reset. `prepareToPlay` and the host's `reset()` start
from silence instead (a full reset of every state).

### State, and moving sessions between computers

The plugin's state (XML) holds the model file's **absolute path**, the
hop / n_fft of the last model that ran, the three parameters, and, for
information only, what the instance was doing (description, message,
latency). Restoring it loads the file again.

The model is **referenced, not embedded**. A session moved to another computer
finds its model only if the file exists at the same absolute path there;
otherwise the plugin passes audio through with the latency it had, and the
window says "Model file not found: <path>". Choose the file again (or copy it
to that path) and save. To move sessions, keep models at the same path on
every machine (e.g. a folder synced to the same place), or re-pick them.

Embedding the model in the state would make sessions self-contained, but every
instance would carry the whole file in the session and in every undo /
snapshot the host keeps: ~5 MB for config.toml's MLP (5.8 MB complex_mask),
~6.5 MB for its TCN, 0.7–1.5 MB for its SNN, per instance (as base64 in some
hosts' project files, +33 %). Ten instances of the MLP make a 50 MB project.
That is why it is not done; an option "embed the model in the session" could
be added later (the state format has a `version` attribute for that).

### Threads and real-time safety (also: hooks for ARA)

- **Control threads** (message thread, a host's state thread, the editor):
  `loadModelFile`, `clearModel`, `setStateInformation`, `prepareToPlay`,
  `releaseResources`. They are serialised by one mutex the audio thread never
  touches, read files, parse, allocate, build engines, report the latency,
  free retired engines.
- **Audio thread** (`processBlock`): takes a pending engine (atomic exchange),
  crossfades, retires the old one (atomic compare-exchange into one of 8
  slots; if all are full, it tries again next block). No allocation, free,
  lock, file access, logging or exception. Denormals flushed
  (`ScopedNoDenormals`). Host blocks of any size are split into chunks of 256,
  so nothing depends on the block size. The host's `reset()` sets a flag the
  audio thread acts on at its next block.
- The only lock on the audio path is the one JUCE's VST3 wrapper takes around
  `processBlock` (`getCallbackLock()`); `prepareToPlay` / `releaseResources`
  take it too, to replace the engine directly.
- For the ARA path: `getModel()` returns the loaded model (also an
  offline-only, centred one that the real-time path refuses) as an
  immutable `shared_ptr<const denoiser::Model>`; `getStatus()` its path,
  description and message; `getMix()`, `getBypass()`, `getHighBand()` and
  `parameters()` the settings. The processor is a `juce::ChangeBroadcaster`
  that sends a change message (on the message thread) whenever the model or
  the status changes. `processBlockForARA` still comes first in `processBlock`.
  (In the end the ARA side keeps its own, document-wide model and does not
  use the instance's: see "ARA: whole clips", "Where the model and the
  settings live". It uses `mixParam_` / `bypassParam_` directly.)

### Tests (ctest, in a plugin build; all headless, ~10 s in Release)

| test | what |
|---|---|
| `plugin_vst3` | `vst/HostTest.cpp`: JUCE's VST3 hosting loads the built `.vst3` (as a DAW would) and sets models and parameters through the plugin's state. Checks: an ARA factory is exposed; no model: latency 669 at 48 kHz and output = input delayed by it, bit for bit, mono and stereo, blocks 1–512; latency = `hostLatencySamples` at 16, 44.1, 48 and 96 kHz for two model shapes; at 16 kHz the output of an MLP, an SNN and a TCN golden model equals the core's `denoiseClip` shifted by the latency (golden input and a 1 s clip, 2 channels; tolerance 1e-5, measured 0); blocks of 16, 17, 64, 256, 333, 1024, 4096, random 1–4096 and a pattern give the same output (1e-5, measured 0); bypass and mix 0 give the input delayed by the latency bit for bit, also through the host's bypass parameter; mix 0.5 = mean of wet and dry (1e-5, measured 0); state saved, a new instance restored from it gives the same output; a missing file, a broken file and a centred model pass audio through with the right latency and message; 8 kHz passes through with latency 0. On Linux it also counts heap allocations and frees through the whole VST3 processing call (JUCE's hosting side, JUCE's wrapper and the processor; the counting `malloc` / `operator new` of the test program are the ones the plugin calls): 0 in steady state, 0 across a model swap made by a state restore, 0 when processing a parameter change. |
| `plugin_processor_alloc` | `vst/ProcessorTest.cpp` compiles the processor sources into a program (no host, ARA off) and counts allocations on the audio thread over 1300 blocks of 0–512 samples at 48, 44.1 and 16 kHz stereo, with parameter changes and a reset, across swaps prepared beforehand to a TCN, a model of another shape, no model, a centred model and an SNN: 0. (Only the audio thread is counted: JUCE's timer thread allocates while posting messages nobody dispatches in a test.) |
| `plugin_processor_core` | the plugin against the core's `HostRateAdapter` with `HighBand::Drop` (0 difference) and `PassThrough` (3e-8) at 48 and 44.1 kHz; status messages; what a swap does (above) |
| `plugin_processor_stress` | 2 s of audio in random blocks on one thread while another loads models (6 kinds incl. missing and centred), unloads, saves / restores state, changes parameters and resets, and a third reads the status. Run under ThreadSanitizer: see below. |

Also run here (not in CI): all 19 tests under **AddressSanitizer + UBSan**
(Debug, GCC 13; this found a test bug, now fixed), the stress test for 10 s
under **ThreadSanitizer** (GCC 13, RelWithDebInfo, JUCE built with TSan too:
1210 model loads during 5877 audio blocks, **no reports**), **Steinberg's
validator** (3.8.0) **47/47 passed** and **pluginval 1.0.4 at strictness 10
SUCCESS**, both with no model and with a model loaded (a build with
`-DDENOISER_DEFAULT_MODEL=<config.toml-sized TCN>`, see Options), pluginval
including its GUI tests (editor open while processing, editor automation)
under Xvfb.

To run the sanitizer builds yourself:

```bash
cmake -S plugin -B plugin/build-tsan -G Ninja -DDENOISER_BUILD_PLUGIN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS=-fsanitize=thread -DCMAKE_CXX_FLAGS=-fsanitize=thread \
  -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=thread \
  -DDENOISER_PROCESSOR_TEST_COUNT_ALLOCATIONS=OFF
cmake --build plugin/build-tsan --target denoiser_processor_test
plugin/build-tsan/vst/denoiser_processor_test_artefacts/RelWithDebInfo/denoiser_processor_test stress plugin/tests/golden 10
```

(ASan + UBSan the same way with `-fsanitize=address,undefined`, then `ctest`.)

### CPU (48 kHz stereo, blocks of 512)

`denoiser_processor_test bench <model.bin>...` times `processBlock` over 10 s
of audio (best of 3). Release, GCC 13, Intel Xeon @ 2.3 GHz shared VM, one
thread; untrained models at config.toml's sizes made by `tools/check_cli.py
--out`; ranges over 3 runs (this machine is noisy):

| model | % of one core | slowest 512-sample block, % of its 10.7 ms |
|---|---|---|
| no model (pass-through) | 0.01 % | 0 % |
| MLP 130 → 1024 → 1024 → 65, magnitude | **8.5–12.9 %** | 14–26 % |
| MLP 260 → 1024 → 1024 → 130, complex_mask | 10–16 % | 18–61 % |
| SNN 260 → 256 × 3 → 130, complex_mask (2 % spikes) | **1.6–1.7 %** | 2–11 % |
| SNN, threshold 0.1 (40 % spikes) / recurrent (45 %) | 2.0–2.2 % | 3–7 % |
| TCN 256 channels, dilations 1–32, magnitude | **7.3–12 %** | 13–34 % |
| TCN, complex_mask | 7.8–13 % | 12–44 % |

The high-band setting makes no measurable difference. The cost is the
network's (see "Speed" above): per channel 200 frames/s, each with ~1.3 M
(MLP) or ~1.6 M (TCN) multiply-adds. The slowest block is what matters for
drop-outs: a block of 512 at 48 kHz carries 2–3 hops per channel, and the
worst seen (61 %, once) is the shared VM; on a desktop expect the averages
above or less. During a model swap both models run for ~100 ms.

### Limitations

- Mono and stereo only (input layout = output layout). Each channel is
  denoised independently, which can shift the stereo image.
- Host rates below 16 kHz or with a ratio needing more than 1000 resampler
  phases pass through (latency 0).
- The model is referenced by path (above). The file chooser shows `*.bin`.
- Loading blocks the UI thread for the load time (up to ~50 ms for a
  config.toml-sized TCN).
- No smoothing of a model's own start-up: a TCN or SNN settles over its
  context after a reset or a swap.
- No preset / program support; one program ("Default").
- AU and standalone are not built.
- `DENOISER_DEFAULT_MODEL` is for validation builds only; normal builds start
  without a model.

### Not verified: running in a DAW

There is no DAW in the development environment, so **"loads and runs in at
least two VST3 hosts" has not been checked**. Everything above ran through
JUCE's VST3 hosting code, Steinberg's validator and pluginval, which are not
DAWs. To finish that item, in REAPER (and then one of Cubase / Studio One /
Bitwig):

1. Build Release, copy `ENM156 Denoiser.vst3` to the VST3 folder (see
   "The VST3 plugin" above), rescan. It should appear as an effect (category
   Fx / Restoration) by ENM156.
2. Put it on a mono and on a stereo track with speech. Without a model: the
   window says "No model loaded", the audio sounds unchanged, and REAPER's
   performance meter / FX window shows a PDC of 669 samples at 48 kHz (614 at
   44.1 kHz).
3. **Load model...** → your exported `model.bin`. The description line
   appears, the orange message disappears, the latency stays the same (for
   an n_fft 128 / hop 80 model). Listen: noise reduced, nothing above ~7.5
   kHz with High band = Drop; with Pass through the top end comes back
   (with its noise).
4. Toggle Bypass (the plugin's and REAPER's own bypass button) and sweep Mix
   during playback: no clicks, no timing shift (compare with a duplicate dry
   track: with Mix 0 they must null completely, see 6).
5. Change the block size (REAPER: Preferences → Audio → Device, e.g. 64,
   512, 2048) and the sample rate (44.1 / 48 / 96 kHz): it keeps working, the
   reported latency follows the table above.
6. **Null test against the command-line tool** (the issue's "null test"):
   make a 16 kHz mono WAV, run `denoise --stream model.bin in.wav cli.wav`
   (or plain `denoise model.bin in.wav cli.wav`: the same samples, see "16 kHz
   host path = CLI"), set the project to 16 kHz, render the track with the
   plugin to a 32-bit float WAV (Mix 100 %; REAPER removes the reported
   latency when rendering), import both, invert the polarity of one and sum:
   the residue should be silence (below −100 dB; `plugin_vst3` measures 0
   for the same comparison through JUCE's VST3 hosting). At 48 kHz compare
   with `denoise --stream` on a 48 kHz file the same way. If the render is
   shifted by exactly the reported latency, the host did not compensate it.
7. Save the project, close, reopen: the model comes back. Rename the model
   file and reopen: audio passes through, the window says "Model file not
   found", the latency is unchanged. Put the file back, reload it.
8. Load a model while playing (and change models repeatedly): no drop-out
   or click, at most a short crossfade.
9. Hosts that bind the plugin through ARA (e.g. as an item / take effect in
   REAPER, or as an ARA extension in Studio One / Cubase) get the ARA path
   instead: see "ARA: whole clips" and its own checklist below.

## ARA: whole clips

`vst/DocumentController.*` (the ARA document controller and playback
renderer), `vst/AraEngine.*` (everything below the ARA interface, no JUCE),
`core/include/denoiser/offline_rate.h` (whole clips at any rate). Bound to an
ARA document, the plugin denoises every audio source of the document as a
whole clip, like `pipeline.enhance()`, on a background thread, and plays the
result back in place of the regions' audio: no latency, sample-aligned with
the source, causal and centred (`realtime: false`) models alike.

### What happens to a source

1. The host creates the audio source and enables sample access. When it ends
   that edit, the analysis thread reads the whole source through the host's
   audio reader (32-bit float, in chunks of 65 536 samples, at most two
   channels kept). From then on the **original** plays.
2. Each channel is resampled to 16 kHz (`clipToModelRate`; skipped at
   16 kHz), denoised as one clip (`denoiseClip`, the CLI's offline path:
   causal or centred framing from the clip's first sample), and resampled
   back to the source's rate (`clipFromModelRate`), aligned: the filters'
   delays are removed, the result has the source's length. With the high
   band passed through, the input minus its own round trip through 16 kHz is
   added (as the real-time plugin does). The whole chain is
   `denoiseClipAtRate`, which is also `denoise --resample` (see the CLI).
3. The denoised audio replaces the original for every region of that source
   (atomically, see below). Progress goes to the host through ARA's
   analysis-progress notifications (started, updated about every 1 %,
   completed) and to the plugin's window.

It is done again when the host reports that the source's samples changed
(`updateAudioSourceContent` with samples affected, or new properties: rate,
length, channels), when the document's model changes (network only; the
source is not read again) and when the high-band choice changes (no network
run: the 16 kHz result is kept and only resampled again). A running analysis
that becomes out of date is cancelled (checked every frame and every read
chunk). Until the new result is there, the previous one keeps playing; after a
content change the original plays (the old result is wrong for the new
samples). Disabling sample access destroys the reader (ARA's rule; a read in
progress stops and starts again when access returns).

| situation | what plays | message (window / state) |
|---|---|---|
| no model chosen | the original | "No model chosen: the original audio plays. ..." |
| the model file is missing / unreadable / not a model | the original | "Model file not found: <path>. ..." / "Could not load ..." |
| a centred model, source shorter than it needs (half its widest window at 16 kHz, 641 samples for the golden one) | the original | "Too short for this model (it needs at least ... samples at 16 kHz ...)" |
| a zero-length source | nothing | "Empty audio source." |
| source rate not supported (below 16 kHz, or e.g. 47 999 Hz) | the original (at its own rate) | "Sample rate 8000 Hz is not supported ..." |
| more than two channels | the first two | "Only the first 2 of N channels are played." |
| longer than the limit (below) or out of memory | silence | "Too long to analyse ..." / "Out of memory ..." |
| the host fails to read | silence | "The host could not read this source's audio ..." |

### Design: classes and threads

- `DenoiserDocumentController` (`ARADocumentControllerSpecialisation`): one
  per host document, shared by all plugin instances bound to it. Keeps one
  `Source` record per audio source (captured properties, the host reader,
  what needs doing, status, its `AudioSlot`s), owns **one analysis thread**,
  stores and restores the archive. Declares no content types and no
  playback transformations (`ARA_TRANSFORMATION_FLAGS
  kARAPlaybackTransformationNoChanges`; `ARAPlaybackRegion` time-stretch is
  not supported); `ARATestHost` reports "does not support content analysis /
  time-stretching / content-based fades / storing audio file chunks".
- `SourceAnalysis` (AraEngine.h, no JUCE): the input at the source's rate,
  its 16 kHz version and the 16 kHz denoised result; makes the
  `SourceAudio` to play at the source's rate or at another rate.
- `SourceAudio`: what one source plays at one sample rate: the original
  (shared, not copied, between versions) and the denoised samples (or none
  yet). Immutable once published.
- `AudioSlot`: where a source's `SourceAudio` (for one playback rate) is
  published. **Hand-over without locks**: the analysis thread stores the new
  pointer, then waits until no audio thread can still read the old one, then
  frees it (a "left-right" scheme with two reader counters); an audio thread
  registers on one counter, loads the pointer, renders, and leaves: two
  atomic increments per region and block, no allocation, no lock, no wait.
  The analysis thread is the only one that ever waits (for at most one
  render block). The slot also has a `working` flag (analysis queued or
  running for it), used by offline renders.
- `DenoiserPlaybackRenderer` (`ARAPlaybackRenderer`): one per bound
  instance. Its regions are turned into a `RenderPlan` (per region: timeline
  range in samples, offset into the stored audio, the slot; the plan owns
  the slots) on the host's document thread whenever the host ends an edit
  or prepares the instance, and handed to the audio thread like the
  real-time engine (one atomic pointer; the old plan is retired to one of 8
  slots and freed on the control side). The audio thread reads only the plan
  and the slots: no ARA model objects, no allocation, no locks, no file or
  host reads (counted: 0 allocations through the whole VST3 call, see
  Tests).
- Who owns the stored audio: the slot owns the current `SourceAudio` (plus
  the analysis's copy of the original, shared); plans own slots; the
  document controller owns sources. Everything is freed off the audio thread.
- Threads: the **host's document thread** (all ARA model calls, archives,
  sample access; the editor's model choice; normally the message thread)
  creates and destroys the host audio readers (ARA allows that only inside
  those calls); the **analysis thread** reads, denoises and publishes; the
  **audio threads** render. One mutex protects the controller's state; the
  audio threads never take it, the analysis thread doesn't hold it while
  working.

### Where the model and the settings live

**The model and the high-band choice are document settings.** One document
controller serves every instance bound to the document, while the real-time
model belongs to one instance; a per-instance model would make the same
source sound different depending on which instance happens to render it. So:

- **Load model...** / **Unload** in any bound instance's window set the
  document's model, used for every source (the window says "Model (all
  clips)"). The instance's own real-time model (its VST3 state) is not used
  in ARA mode.
- **Above 8 kHz** in that window is the document's choice too (it changes
  the stored audio; switching it re-renders from the kept 16 kHz result, no
  network run).
- **Mix** and **Bypass** stay the instance's automatable parameters, applied
  by its renderer as a plain crossfade (both signals are aligned):
  `(1 − g) · original + g · denoised`, 20 ms ramps; mix 0 or bypass give the
  original bit for bit. The host's bypass through `processBlockBypassed`
  plays the original too.
- **Saved in the ARA archive** (`doStoreObjectsToStream`, only when the host
  asks for document data): the magic `ENMD`, a version (1), the model file's
  absolute path (referenced, not embedded, as in VST3 mode) and the high-band
  choice. **The audio is not stored**: on restore the model is loaded again
  and every source is analysed again (the issue's recommendation; a cache is
  a later improvement). An empty archive (from before this version) restores
  as "no model"; a newer version is refused (restore fails). A missing model
  file leaves the original playing with "Model file not found: <path>", and
  the path is kept, so saving again doesn't lose it. The archive ID stays
  `org.enm156.denoiser.aradocumentarchive.1` (the earlier plugin stored
  nothing).
- For information only, an instance's VST3 state also carries an `ARA`
  element with the document's model path, message and per-source states
  (the headless test reads it).

### The stored audio: memory, not a file

Everything is kept **in memory**, as float32: the original and the denoised
audio at the source's rate, plus the 16 kHz input and output (a third each
at 48 kHz), about 2.7 × 4 bytes per sample and channel: **an hour of 48 kHz
stereo takes about 3.7 GB**, a 5-minute stereo take 300 MB. Memory keeps the
audio thread free of file reads (a temporary file would need its own
read-ahead thread and real-time-safe buffering) and costs nothing to set up;
a DAW project with several hour-long sources would want a file-backed store,
which is not done. **Limit: 400 million samples per source** (all channels:
69 minutes of 48 kHz stereo, 2.3 hours of 48 kHz mono); a longer source is not
read and plays silence with a message, and an allocation failure does the
same ("Out of memory"). There is no limit on the number of sources.

### Sample rates

- The source is analysed at its own rate (any rate from 16 to 384 kHz that
  the resampler supports: 22.05, 24, 32, 44.1, 48, 88.2, 96, 176.4, 192,
  352.8, 384 kHz).
- **Playback at another rate than the source's** (a 44.1 kHz file in a
  48 kHz project, which ARA leaves to the plugin): a second version at the
  playback rate is made from the 16 kHz results (the denoised audio and the
  original, both upsampled from 16 kHz), so it is the same denoised signal,
  sample-aligned in time; the original there is band-limited to ~7.5 kHz and
  the high band can't be passed through (dropped). A source whose own rate is
  unsupported plays silence at other rates.
- Timeline → source mapping: a region plays its modification range 1:1 from
  its start on the timeline (`getSampleRange` at the render rate; source
  sample = timeline sample − round(start in playback time · rate) +
  round(start in modification time · rate)). Overlapping regions add up.
  Stereo sources on a mono output play the mean of both channels; mono
  sources on a stereo output play on both.

### Rendering, offline renders, preview

- Realtime: whatever is stored now plays (the original before the analysis
  finishes, silence before the source has been read; reading is fast
  compared with denoising).
- **Offline (non-realtime) renders and bounces wait** for analysis that is
  queued or running for the regions in the block (up to 300 s per block),
  so a bounce right after loading a model contains the denoised audio. Not
  for sources the host hasn't released yet or can't be read.
- Editor renderer (preview): **not provided** beyond JUCE's default (which
  adds nothing). The plugin has no ARA editor view of its own (no waveform,
  no regions to audition in its window), so nothing would call for a
  preview; the playback renderer covers what the host plays.
- Speed (Release, one analysis thread per document, untrained models at
  config.toml's sizes, `denoiser_ara_host_test ... bench <model> 60`): 60 s
  of 48 kHz stereo in 5.2 s with the MLP (11.6× real time), 4.5 s with the
  TCN (13.5×), 1.3 s with the recurrent SNN at 40 % spikes (46×). Sources are
  analysed one after another; channels of one source too.

### Tests (ctest)

| test | what |
|---|---|
| `plugin_ara` | `vst/AraHostTest.cpp`: a **headless ARA host** made of JUCE's ARA hosting classes (`ARAHostDocumentController`, `ARAHostModel::*`) on the ARA SDK's host dispatch library. It loads the built `.vst3` with JUCE's VST3 hosting, creates ARA documents (musical context, region sequence, in-memory audio sources read through its audio access controller, modifications, playback regions), binds instances as playback renderers, gives the document its settings through `restoreObjectsFromArchive`, renders through the VST3 `process` call with a play head, and compares with the core. Covers: a whole source at 16 kHz equals `denoiseClip` (MLP, SNN, TCN, centred MLP; mono and stereo), at 44.1 and 48 kHz equals `denoiseClipAtRate` (MLP, TCN with the high band passed, centred TCN); a region moved to 0.5 s, trimmed (0.25–0.6 s of the source at 0.1 s, changed while prepared), split in two with the second part moved, a second use of the same source elsewhere (silence around and between); **realtime playback before the analysis finishes is the original bit for bit**, then the denoised audio, and the host receives started / updated / completed progress; a content update (sample access off, new samples, on, `updateAudioSourceContent`) gives the new source's denoised audio; a model change gives the new model's output; bypass and mix 0 give the original bit for bit, mix 0.5 the mean; `storeObjectsToArchive` → a new document → `restoreObjectsFromArchive` gives the same output and model path; a missing model file plays the original with "not found" and keeps the path; a 44.1 kHz source played at 48 kHz equals the 16 kHz result upsampled; a too-short source with a centred model, an empty source and an 8 kHz source (silence at 48 kHz, the original at 8 kHz) with their messages; on Linux, **0 heap allocations** over 400 realtime blocks of 1–512 samples with two regions, through JUCE's hosting, the VST3 wrapper and the renderer. Measured error: **0** in every comparison (the tolerance is 1e-5 of max(1, \|ref\|), as in `plugin_vst3`), Release with LTO and Debug, GCC 13. ~0.3 s in Release, ~1.5 s in Debug. |
| `plugin_ara_engine` | `vst/AraEngineTest.cpp`, no JUCE: `SourceAnalysis` equals `denoiseClipAtRate` (MLP / SNN / TCN / two centred models at 16, 44.1, 48 kHz, both high-band choices; versions at another rate); no model, too short, empty, unsupported rates, >2 channels, cancellation; `renderPlan` mapping for moved / trimmed / split / overlapping regions, regions before the source's start or past its end, mono / stereo in and out, mix 0 / 0.25 / 1; 0 allocations in `renderPlan`; `AudioSlot` under load: one thread publishes new audio while two render from it for 1 s (~10 000 publications, ~600 000 blocks), no block mixes two publications. |
| `offline_rate` (core) | `tests/test_offline_rate.cpp`: `modelRateLength`; 16 kHz copies and equals `denoiseClip` bit for bit; alignment at 22.05, 44.1, 48, 96 kHz: a band-limited signal to 16 kHz matches the same signal computed at 16 kHz to −114 dB, the round trip matches the input to −108 dB (one sample off: −6 to −18 dB); `denoiseClipAtRate` equals the parts composed by hand, PassThrough − Drop = input − round trip; short clips for centred models, unsupported rates, empty clips, cancellation. |
| `cli` | also `denoise --resample` at 48 kHz (causal and centred, both high-band choices) equals `denoiseClipAtRate` (1e-5), and its error cases. |

### What was run (and on what)

All on Linux x86-64 (Ubuntu 24.04, GCC 13, 4 cores); CI runs the same
ctest on Linux, Windows and macOS, which has not been seen for these tests
yet (they are written to be portable: no GCC-only code, the allocation count
only on Linux/glibc, the test hook through an environment variable set with
`SetEnvironmentVariableW` on Windows).

- **ctest, Release (GCC 13, link-time optimisation)**: all 23 tests pass
  (`plugin_ara` ~0.6 s, `plugin_ara_engine` ~1.1 s); `plugin_ara` and
  `plugin_ara_engine` also passed 30 times in a row.
- **ctest, Debug** (JUCE assertions and the ARA library's API validation,
  `ARA_VALIDATE_API_CALLS`, on in the plugin and in the host side): all pass,
  no assertion. That validation found the missing information URL.
- **AddressSanitizer + UBSan** (Debug, JUCE, the plugin and the tests all
  instrumented): all 22 tests pass with no report, and `plugin_ara` 10 more
  times (this found two races in the test itself, now fixed, and the case of
  a renderer prepared while the analysis was busy staying silent instead of
  playing the original, now fixed in the plugin).
- **ThreadSanitizer**: TSAN_PLACEHOLDER
- **Steinberg's VST3 validator** 3.8.0 on the Release build: **47 / 47 passed**.
- **pluginval 1.0.4, strictness 10, including its GUI tests under Xvfb**:
  **SUCCESS** (no model; these tools don't bind ARA, so they test the
  real-time side and that ARA support didn't break it).
- **The ARA SDK's `ARATestHost`** (ARA_Examples of the pinned SDK 2.3.0,
  built in Debug against the VST3 SDK 3.8.0 with its ARA validation on;
  outside ctest, it needs the VST3 SDK and a separate build; GCC 13 needed
  `-include limits` for one of its headers): all its test cases
  (property updates, content updates, content reading, modification
  cloning, archiving, split archives, drag and drop, playback rendering,
  editor view, processing algorithms, audio file chunk saving and loading)
  ran against the Debug and the Release plugin with **no assertion and exit
  status 0**. It reports the plugin's factory correctly (no content analysis,
  no time-stretching, no fades, no audio file chunks). It doesn't check
  output values (it prints the first rendered samples; its render happens
  before the analysis, so they are the source's first samples or silence),
  and it gives the plugin no model, so its value is the API conformance of
  the document controller under a second, independent host.
  Run as `ARATestHost -vst3 "<...>/ENM156 Denoiser.vst3/Contents/x86_64-linux/ENM156 Denoiser.so"`.
- Not run: anything on Windows or macOS (CI will run the ctest part), any
  DAW, the ARA-mode window (it compiles and is shared with the tested VST3
  window, but no test opens it bound to ARA).

### The issue's "Done when", one by one

| "Done when" item | status |
|---|---|
| a clip in an ARA host plays back denoised, in sync, including when the region is moved, trimmed or split | **Verified through the headless ARA host** (`plugin_ara`: whole sources at 16 / 44.1 / 48 kHz, a moved, a trimmed, a split region, two uses of one source; sample-aligned, error 0). **Not verified in a DAW.** |
| the samples equal the CLI output for the same source and model (after resampling) | **Verified headlessly, in two steps**: the plugin's output equals the core's `denoiseClip` (16 kHz) / `denoiseClipAtRate` (44.1, 48 kHz) with error 0 (`plugin_ara`), and `denoise --resample` (16 kHz: plain `denoise`) equals `denoiseClipAtRate` (`cli`, 1e-5). **Not verified with a DAW's render.** |
| save, close and reopen a project: it comes back denoised with the same model | **Verified headlessly** as ARA archive store → new document and instance → restore: same output (error 0) and model path; a missing model file plays the original with a message. **Not verified in a DAW** (a DAW's own project saving, and when it restores, is not exercised). |
| editing the source in the host triggers a new analysis | **Verified headlessly**: sample access off, new samples, on, `updateAudioSourceContent` → the new samples' denoised output (error 0); also under concurrent rendering. **Not verified in a DAW** (which edits send which notifications is host-specific). |
| a non-causal checkpoint works in ARA mode and is refused, with a message, in real-time mode | **Verified headlessly** with the golden centred models (`offline_mlp_mask_centred`, `offline_tcn_complex_centred`): in ARA mode equal to `denoiseClip` / `denoiseClipAtRate` (error 0, `plugin_ara`); in real-time mode audio passes through with "... centred ... can't run in real time ... It can be used in ARA mode." (`plugin_vst3`). There is no exported trained non-causal checkpoint here. **Not verified in a DAW.** |

### Limitations and what is not supported

- Time-stretching, content-based fades, content (notes, tempo, chords) for
  the host, audio file chunks, processing algorithms: none (declared so).
  A region whose playback duration differs from its modification duration
  (a host that stretches anyway) plays unstretched from its start and is
  cut at its end.
- Head and tail time are 0: denoising doesn't ring past the region.
- The model is one per document; per-clip models would need a per-
  modification setting (the archive format has a version for it).
- Two channels per source at most.
- Memory as above; no on-disk cache, so reopening a project analyses every
  source again (11–46× real time for config.toml-sized models on one core).
- The analysis uses one thread per document.
- Logic (ARA through Audio Units) is out of scope (no AU target).
- The host's document thread is assumed to be the message thread when the
  model is chosen in the window (true for the hosts named in the issue): the
  host is told about the content change from that thread.
- JUCE 9.0.5 / ARA SDK 2.3 quirks found on the way (none needed a change
  outside this folder): ARA requires a non-empty information URL in the
  factory (the plugin now has `COMPANY_WEBSITE`); JUCE's
  `ARAHostModel::PlaybackRegionRegistry` destructor crashes (erases from the
  map it iterates) if regions are still registered, so the test removes them
  first; JUCE's VST3 hosting unloads the module with its last instance while
  a document's ARA objects still need it, so the test keeps one unbound
  instance alive, as a DAW keeps modules loaded; the ARA library drops an
  analysis that starts and completes between two `notifyModelUpdates` calls
  without reporting it (by design), so a host sees progress only for
  analyses that outlast its polling interval.

### Checklist for the owner: ARA in a DAW (not done: no DAW here)

Nothing in this section has been run in a DAW. Use a Release build installed
as in "Not verified: running in a DAW" above, an exported `model.bin`, and a
speech recording with noise (best: a 16 kHz mono WAV and a 48 kHz stereo
WAV). Try **REAPER** (6.x or 7.x) and **one of Studio One (5.5+) / Cubase or
Nuendo (Pro / Artist 12+)**. Menu names below are from those hosts' manuals
and may differ slightly by version.

1. **Insert as an ARA extension.**
   - REAPER: select the item, open its take FX chain (Item → Take → Take FX
     chain, or Shift+E) and add "VST3: ENM156 Denoiser (ENM156)". REAPER
     binds ARA plug-ins in take FX when ARA is enabled (Preferences →
     Plug-ins → ARA, and in the FX window's menu "Enable ARA for this
     plug-in"). The plug-in's window shows "Model (all clips)" and a list of
     sources: that is ARA mode. If it shows "Latency: ..." instead, it is
     running as a plain real-time effect.
   - Studio One: right-click the audio event → the plug-in's name in the
     event menu (ARA extensions are listed there), or drag the plug-in onto
     the event.
   - Cubase / Nuendo: select the event → Audio → Extensions → ENM156
     Denoiser.
2. **Load a model**: in the plug-in's window, **Load model...** → your
   `model.bin`. The description line appears; the source list shows each
   clip "denoising NN %", then "denoised". The host may show its own
   analysis progress bar. Without a model every clip says "original" and
   sounds unchanged.
3. **Listen**: noise reduced, nothing above ~7.5 kHz with "Drop"; switch
   "Above 8 kHz" to "Pass through": the top end comes back for every clip
   (quickly: no network run). Mix and Bypass act per plug-in instance, with
   no click and no timing shift.
4. **Sync**: put an unprocessed copy of the clip on another track, invert
   its polarity, set Mix to 0 (or bypass): the two must null completely
   (the original is copied bit for bit). With Mix 100 % the difference is
   the removed noise, with no comb filtering or echo (no delay). Move the
   clip, trim its start and end, split it and move one part, duplicate it:
   each part stays in sync (null with Mix 0, and denoised content stays
   under the same words).
5. **Against the command-line tool**: export (render / bounce / "glue") the
   processed clip with Mix 100 % at the project rate to a 32-bit float WAV,
   starting exactly at the clip start. Run
   `denoise --resample --float model.bin source.wav cli.wav` on the same
   source file (`--high-band=pass` if you chose Pass through; for a 16 kHz
   source plain `denoise model.bin source.wav cli.wav` gives the same
   samples). Import `cli.wav` at the clip's position, invert, sum: the
   residue should be below −100 dB (the headless test measures 0). A trimmed
   region equals the corresponding part of `cli.wav`. Do this with the
   project at the file's own rate: at another project rate the plug-in plays
   the 16 kHz result upsampled to the project rate (see "Sample rates"),
   which the tool doesn't produce directly.
6. **Bounce right after loading a model** (offline render): the bounce must
   contain the denoised audio (the renderer waits for the analysis), not the
   original.
7. **Edit the source**: a destructive edit (REAPER: e.g. apply a fade or
   gain as new take / "Apply track FX to items" on the source, Studio One /
   Cubase: a direct offline process on the event): the clip shows
   "denoising" again and then plays the denoised new samples.
8. **Save, close, reopen** the project: it comes back denoised with the same
   model (the source list shows the analysis running again on open; the
   audio is not stored in the project). Rename the model file and reopen:
   the clips play the original and the window says "Model file not found:
   <path>"; put the file back and choose it again.
9. **A non-causal model** (exported from a checkpoint with
   `stft.causal = false`; its description ends in "centred frames: offline
   only"): load it
   in ARA mode, it denoises (listen: no pre-echo, it uses half a window of
   look-ahead); load the same file in a plug-in instance used as a normal
   real-time track insert: audio passes through and the window says the
   model "can't run in real time ... It can be used in ARA mode."
10. Also check: a 44.1 kHz file in a 48 kHz project (plays denoised, in
    sync, dull above ~7.5 kHz even with Pass through), a mono file on a
    stereo track, a very short clip with a non-causal model ("Too short",
    original plays), two instances on different tracks sharing the model,
    and that the project's CPU load during playback is negligible (all work
    happens in the analysis).

## Layout

```
plugin/
  CMakeLists.txt        options, C++17, adds the folders below
  cmake/Dependencies.cmake   KISS FFT always; JUCE + ARA SDK with DENOISER_BUILD_PLUGIN
  core/                 denoiser_core (alias denoiser::core): static library, no JUCE
    include/denoiser/   public headers: fft.h (RealFft), model_file.h, features.h, stream.h,
                        offline.h, offline_rate.h, host_rate.h, resample.h, networks.h (MLP, SNN),
                        denoiser.h (Model, ChannelDenoiser)
    src/                sources
  cli/                  denoiser_cli, built as `denoise`: links the core
    main.cpp            argument handling, offline, --resample and --stream modes
    wav.h / wav.cpp     WAV reader/writer
  tests/                core tests, one executable each, run by ctest
    check.h             CHECK / CHECK_NEAR, no test framework
    golden/             golden test vectors written by tools/ (see tools/README.md)
  tools/                Python: checkpoint export and golden vectors (see tools/README.md)
  vst/                  DenoiserPlugin: JUCE AudioProcessor, VST3 + ARA
    PluginProcessor.*   the processor: parameters, state, model loading, engine hand-over
    RealtimeEngine.*    one prepared processing state (model, denoisers, adapter, mix); no JUCE
    PluginEditor.*      the plugin's window
    DocumentController.*     ARA: document controller (analysis thread, archive) and playback renderer
    AraEngine.*         ARA below the ARA interface: analysis, stored audio (AudioSlot), region rendering; no JUCE
    HostTest.cpp        headless VST3 host: the plugin_vst3 test
    AraHostTest.cpp     headless ARA host: the plugin_ara test (and an analysis benchmark)
    AraEngineTest.cpp   the ARA engine on its own: the plugin_ara_engine test
    ProcessorTest.cpp   the processor without a host: plugin_processor_* tests, CPU benchmark
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

## Incremental TCN (`core/include/denoiser/tcn.h`)

`TcnNetwork` (a `FrameNetwork`; `makeTcnNetwork(model)`) evaluates the TCN
one frame per `process()` call: per block a ring of its last
`(kernel − 1) · dilation` frames of `relu(LayerNorm(block input))`, plus the
current frame, is all the dilated convolution reads. The start-up
`FrameModel.forward` does (`context − 1` zero input frames) runs once when the
weights are loaded (`TcnModel`); `reset()` copies that snapshot back.
Weights (`TcnModel`, immutable) can be shared between instances, e.g. one per
channel: `std::make_shared<const TcnModel>(model)`.

Kernels: weights stored transposed (`[in][out]`, the dilated conv as
`[tap][in][out]`), so every layer is a sum of weight rows scaled by its
inputs, eight rows at a time; rows of inputs that are exactly zero (after a
ReLU, about half) are skipped. Plain loops, vectorised by GCC/Clang/MSVC
without fast-math; no BLAS.

Measured (`bench_tcn`, Release, GCC 13, generic x86-64 flags, one thread,
Intel Xeon @ 2.3 GHz shared VM, 2 MB L2 per core; best of 21 runs, the median
is up to 1.5× slower on this noisy machine), config.toml's default size (n_in
130, 256 channels, kernel 3, dilations 1–32, n_out 65; 1.62 M multiply-adds
and 6.5 MB of weights per frame), random weights:

| | µs per frame | frames/s | real-time factor at 200 frames/s |
|---|---|---|---|
| random weights (~half the ReLU outputs zero) | 150–200 | 5000–6500 | **25–33×** |
| worst case: nothing zero, every row read | 230–340 | 2900–4300 | **15–22×** |
| `reset()` | 2.6 | | (copies 129 kB) |
| the straightforward start-up it replaces (126 frames) | 21 000–43 000 | | |

`-march=x86-64-v3` (AVX2, FMA) gains little (~10 %): the loop streams 6.5 MB
of weights per frame from L3, so it is bound by memory bandwidth, not
arithmetic. Clang 18 is in the same range. Per instance: 129 kB of state plus
~3 kB scratch; per model: 6.5 MB of weights plus the 129 kB snapshot.

Tests: `tests/test_tcn.cpp` (golden `pred` and `tcn.h*` from the first frame,
streamed and offline end to end, reset and start-up equivalence, a
brute-force double-precision reference for other kernels/dilations and clips
shorter than the context, malformed headers), `tests/test_tcn_alloc.cpp`.

## Inference, the per-channel denoiser and the command-line tool

`core/include/denoiser/networks.h` (MLP, SNN), `denoiser.h` (`Model`,
`ChannelDenoiser`), `cli/` (`denoise`). `Model` builds all three network
types; the TCN itself is described in the section above.

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
// ... or offline at any supported rate, aligned (ARA mode, denoise --resample;
// #include "denoiser/offline_rate.h"; also clipToModelRate / clipFromModelRate):
std::vector<float> z = denoiser::denoiseClipAtRate(model->info(), *net, rate, x, n, denoiser::HighBand::Drop);
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
denoise [--stream | --resample] [--high-band=drop|pass] [--format=same|pcm16|pcm24|pcm32|float32|float64] [--float] [-q]
        model.bin in.wav out.wav
```

- Default: offline, `denoiseClip` per channel (any number of channels); the
  file must be at the model's rate (16 kHz). Centred models work here.
- `--resample`: offline at the file's own rate (16 kHz or 22.05–384 kHz),
  exactly as the plugin's ARA mode does it (`denoiseClipAtRate`: each channel
  to 16 kHz, `denoiseClip`, back, aligned; `--high-band` as there). Centred
  models work here too. Use it to compare an ARA render with the tool.
- `--stream`: each channel as the plugin runs it (`ChannelDenoiser` behind
  `HostRateAdapter` at the file's rate: 16 kHz or 22.05–384 kHz), latency
  removed with `processClipAligned`. Centred models are refused with a message.
- Output: same length, aligned, by default in the input's sample format
  (`--float` for 32-bit float); PCM is rounded and clipped, with a warning.
- Exit status 0; 1 for errors (missing, unreadable or malformed file, wrong or
  unsupported rate, a centred model with `--stream`); 2 for bad arguments.
- File names go through `std::filesystem::path` and, on Windows with MSVC,
  `wmain`, so non-ASCII paths should work there too (not tested on Windows);
  the tool reads the model file itself and calls `Model::fromMemory`, not the
  narrow-path `ModelFile::load`.

### Tests (ctest; no Python)

| test | what |
|---|---|
| `networks` | every golden MLP / SNN case: `pred` from the golden `x` (1e-5); SNN spikes vs `snn.spikes*` exactly, spike rates vs the header, both kernels with identical spikes; `reset()` bit-exact; missing tensors give clear errors; the factory builds a TCN |
| `denoiser_golden` | from the golden `input`: `denoiseClip` vs `offline` (causal and centred), `ChannelDenoiser` vs `streamed` (1e-5, pre-roll 1e-4; muted pre-roll exactly 0); the 16 kHz host path (below) |
| `networks_alloc` | no heap allocation in the networks, `ChannelDenoiser`, and the adapter at 16 and 48 kHz stereo; `reset()` reproduces a fresh run bit for bit |
| `cli` | runs the built `denoise` on float and 16-bit WAVs made from the golden inputs (offline and `--stream`, stereo) and checks the files; 16 runs with their expected exit status; `--stream` at 48 kHz; `--resample` at 48 kHz equals `denoiseClipAtRate` (causal and centred, both high-band choices); the WAV code on its own. `DENOISE_LAUNCHER` runs the tool through e.g. `qemu-aarch64` |
| `offline_rate` | whole clips at other rates (`offline_rate.h`, for ARA): see "ARA: whole clips", "Tests" |

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
