# plugin/tools: from Python checkpoints to C++

The plugin can't read `model.pt` (a PyTorch pickle). These scripts turn a
checkpoint into a file C++ reads in a few lines, and write the golden test
vectors the C++ port is tested against. They only import `denoiser/`; nothing
there is changed.

| file | what it does |
|---|---|
| `export.py` | checkpoint → `model.bin` (+ `vectors.bin` with `--vectors`), with a round-trip check |
| `make_golden.py` | writes the golden set to `plugin/tests/golden/` (17 cases, ~1.4 MB) |
| `check_golden.py` | recomputes every golden case from its own files, with PyTorch and with `reference.py` |
| `check_roundtrip.py` | export → read back → rebuild, for every network type × output kind × framing |
| `reference.py` | the whole denoiser in NumPy, written from this spec alone: a blueprint for the C++ port |
| `tensorfile.py` | the container (read/write) |
| `modelfile.py` | the model file: export, read, rebuild the PyTorch model from it |
| `vectors.py` | the vectors file, the synthetic test clip |

## Running

Use the repo's Python with torch (`.venv/bin/python` in the root README's setup):

```bash
# a trained checkpoint -> out/model.bin; checks that it reads back to the same weights
# and settings and that the model rebuilt from the file alone gives identical output
.venv/bin/python plugin/tools/export.py runs/default/model.pt out/
# ... and test vectors of that model on a synthetic clip (or --wav clip.wav, 16 kHz mono, first second)
.venv/bin/python plugin/tools/export.py runs/default/model.pt out/ --vectors
# refuse a centred (stft.causal = false) model instead of marking it offline-only
.venv/bin/python plugin/tools/export.py runs/default/model.pt out/ --realtime

# the golden set (deterministic; commit the result)
.venv/bin/python plugin/tools/make_golden.py
.venv/bin/python plugin/tools/check_golden.py      # exit code 1 on any mismatch
.venv/bin/python plugin/tools/check_roundtrip.py
```

Checkpoints don't store `data.sample_rate`: `export.py` takes it from
`config.toml` / `config.local.toml` (16000), or `--set data.sample_rate=...`.
`export.py` refuses `model.filterbank` checkpoints before anything else.

## Container (both file kinds)

All integers little-endian. Version 1:

| offset | size | content |
|---|---|---|
| 0 | 8 | magic: the ASCII bytes `ENMTENSR` |
| 8 | 4 | uint32: container version, `1` |
| 12 | 4 | uint32: header length `H` in bytes |
| 16 | `H` | header: one UTF-8 JSON object, padded at the end with spaces (0x20) so that `16 + H` is a multiple of 64 |
| `16 + H` | ... | data section; `data_start = 16 + H` is 64-byte aligned |

The header always has:

- `"kind"`: `"denoiser-model"` or `"denoiser-vectors"`, and `"kind_version"`: `1`
- `"tensors"`: a list, in storage order, of `{"name", "shape", "dtype", "offset"}`:
  - `dtype` is always `"f32"`: IEEE-754 float32, little-endian
  - `shape` is a list of ints, row-major (C order, last index fastest); `[]` would be a scalar (none is written today)
  - the tensor's bytes are `[data_start + offset, data_start + offset + 4 * prod(shape))`; `offset` is a multiple of 64
  - zero bytes fill the gaps between tensors; nothing follows the last tensor

So a reader: check the magic and version, read `H`, parse the JSON, then
point a `const float*` at `data_start + offset` for each tensor (the data is
aligned, so a whole file read into a 64-byte aligned buffer, or `mmap`ed, can
be used in place on a little-endian machine). Unknown header keys must be
ignored, so later versions can add some.

## Model file (`"kind": "denoiser-model"`)

Header keys besides the container's:

- `"realtime"`: `true` for causal models. **`false` for centred models
  (`stft.causal = false`)**: they look half a window ahead, so the real-time
  (VST3) path must refuse them; the offline (ARA) path can run them.
- `"derived"`: everything the C++ code needs, already worked out:

| key | meaning |
|---|---|
| `sample_rate` | Hz (16000) |
| `n_fft`, `hop` | short window and hop, in samples |
| `long_factor`, `long_n_fft` | long window = `n_fft * long_factor` samples; `0` = no long window |
| `n_freq` | `n_fft / 2 + 1` bins per frame (`F` below) |
| `n_in`, `n_out` | sizes of a model input and output frame |
| `input_layout`, `output_layout` | the parts of a frame, in order, as `[[part, size], ...]` (below) |
| `output` | `"magnitude"`, `"mask"` or `"complex_mask"` |
| `out_activation` | `"relu"` (magnitude), `"sigmoid"` (mask) or `"none"` (complex_mask) |
| `compress`, `decompress` | complex_mask only (else `null`): magnitude exponent, and `1 / compress` as Python computes it (in double; it is used as float32) |
| `mask_bound` | complex_mask only (else `null`): soft limit of the mask's size; `0` = none |
| `context_frames` | input frames one output frame depends on: 1 (MLP, SNN) or `1 + (kernel - 1) * sum(dilations)` (TCN) |
| `framing` | `"causal"` or `"centred"` |
| `stream_delay` | `n_fft - hop` for causal models (else `null`): see "Streaming" |

- `"net"`: the network, by `"type"`:
  - `mlp`: `"layers": [{"in", "out", "activation"}]`; `"relu"` for all but the last, whose activation is `out_activation`
  - `tcn`: `"channels"`, `"kernel"`, `"blocks": [{"dilation", "trim"}]` (`trim = (kernel - 1) * dilation`), `"layer_norm_eps"` (1e-5), `"context"`
  - `snn`: `"hidden"` (neurons per spiking layer), `"recurrent"` (bool), `"threshold"`, `"steps_per_frame"`
- `"config"`: the `stft`, `model`, `tcn` and `snn` settings as the checkpoint
  stores them (after `load_checkpoint` filled in defaults of old checkpoints).
  For information and for rebuilding in Python; C++ needs only `derived` and
  `net`. (It contains settings that don't apply, e.g. `model.hidden` of an SNN.)

### Tensors, in the order the forward pass uses them

Matrices are `[out, in]`: `y = W @ x + b` with `W` row-major. Every name is
always present for its network type unless marked optional.

**MLP**: for `i = 0 .. len(layers) - 1`:
`linear{i}.weight [out, in]`, `linear{i}.bias [out]`;
`h = act_i(W h + b)`.

**TCN** (`C` channels, `K` = kernel):

```
input.weight [C, n_in], input.bias [C]                 # 1x1 conv = matrix
for each block b:
  block{b}.norm.weight [C], block{b}.norm.bias [C]      # LayerNorm over the C channels of one frame
  block{b}.conv.weight [C, C, K], block{b}.conv.bias [C] # [out, in, tap]
  block{b}.mix.weight  [C, C],    block{b}.mix.bias [C]  # 1x1 conv = matrix
final_norm.weight [C], final_norm.bias [C]
output.weight [n_out, C], output.bias [n_out]
```

Per frame `t` (`h_b(t)` = input of block `b` at frame `t`):

```
h_0(t)   = input.weight @ x(t) + input.bias
a_b(t)   = relu(LayerNorm_b(h_b(t)))                               # biased variance, eps 1e-5
c_b(t)   = conv.bias + sum_{j=0}^{K-1} conv.weight[:, :, j] @ a_b(t - (K-1-j) * d_b)
h_b+1(t) = h_b(t) + mix.weight @ relu(c_b(t)) + mix.bias
y(t)     = out_activation(output.weight @ relu(LayerNorm_final(h_last(t))) + output.bias)
```

So tap `K-1` is the current frame and tap 0 the oldest. Each block needs a
ring of its last `trim + 1` inputs (or of their `a_b`, plus the newest `h_b`
for the residual).

**Start-up / reset.** Offline, `FrameModel.forward` puts `context - 1` zero
**input frames** `x` in front of the clip, not zero activations (`input.bias`
and the LayerNorms make those non-zero). To match from the first frame:
on reset, fill the rings with anything finite and push `context - 1` zero
input frames through the network, discarding the outputs. That flushes every
ring with exactly the right values (`reference.py` does this; tested with
rings starting at random values). Cheaper alternative: the network's response
to constant zero input is constant per block, so each ring can be filled with
its block's constant input once per model.

**SNN** (layers `i = 0 .. L-1` with `h_i` neurons):

```
layer{i}.weight [h_i, in_i], layer{i}.bias [h_i]   # in_0 = n_in, in_i = h_{i-1}
layer{i}.decay [h_i]                               # already sigmoid(raw_decay), use as is
layer{i}.recurrent [h_i, h_i]                      # only if net.recurrent; no bias
readout.weight [n_out, h_{L-1}]                    # no bias
readout.decay [n_out]                              # already through the sigmoid
output.bias [n_out]                                # added to the readout potential, at the output only
```

State: `v_i` (membrane potentials), `s_i` (spikes of the last time step) per
layer, `u` (readout potentials); all zero at reset. Per frame `x`:

```
first = layer0.weight @ x + layer0.bias            # once per frame, same in every step
repeat steps_per_frame times:
    below = None
    for i in 0 .. L-1:
        cur = first                                if i == 0
              layer{i}.weight @ below + layer{i}.bias  otherwise
        if recurrent: cur += layer{i}.recurrent @ s_i   # own spikes of the previous time step
        v_i = cur + layer{i}.decay * v_i
        s_i = (v_i - threshold > 0) ? 1 : 0            # strictly above
        v_i = v_i - s_i * threshold                    # reset by subtraction
        below = s_i
    u = readout.weight @ below + readout.decay * u
y = out_activation(u + output.bias)                # after the frame's last step
```

The previous step's spikes `s_i` carry across frames too.

### Features and synthesis (`denoiser/features.py`, `pipeline.py`, `stream.py`)

Notation: `F = n_freq`, `L = long_n_fft`. All float32 as in PyTorch.

**Window**: `torch.hann_window(N)`, the *periodic* Hann,
`w[i] = 0.5 - 0.5 cos(2 pi i / N)`, `i = 0 .. N-1` (so `w[0] = 0`). PyTorch
computes it in float32 (roughly `cosf(i * (float)(2 pi / N)) * -0.5f + 0.5f`);
in double and rounded it differs by up to 1.5e-7. The golden cases with
`spec` also store `window` and `long_window` exactly.

**Causal frames** (`framing = "causal"`): frame `t` covers input samples
`[(t+1) hop - n_fft, (t+1) hop)`, and its long window
`[(t+1) hop - L, (t+1) hop)`; samples before 0 are zeros. Live: keep the last
`max(n_fft, L)` samples (zeros at reset), push `hop` new ones per frame.

**Centred frames** (`framing = "centred"`, offline only): `torch.stft`'s
centring: reflect-pad the clip by `n_fft / 2` on both sides (mirror without
repeating the edge sample, like `numpy.pad(mode="reflect")`), frame `t` =
padded `[t hop, t hop + n_fft)`; the long window pads by `L / 2` and takes
`[t hop, t hop + L)`. `T = floor(N / hop) + 1` frames for `N` samples.
Reflect padding needs `N > L / 2` (and `N > n_fft / 2`).

**Short spectrum**: `spec = rfft(frame * w)`, `F` complex bins.

**Long window** (`long_factor > 0`): `P[j] = |rfft(long_frame * w_long)[j]|^2`
for `j = 0 .. L/2` (`|z|` first, then squared), then for each short bin
`k = 0 .. F-1`, with `p = long_factor / 2` (integer division):

```
lo = max(k * long_factor - p, 0),  hi = min(k * long_factor - p + long_factor, L/2 + 1)
mean = sum(P[lo:hi]) / (hi - lo)                 # avg_pool1d, count_include_pad=False
long_mag[k] = sqrt(mean * long_factor) / long_factor
```

The edge windows have fewer bins (`k = 0` covers `long_factor - p` bins).

**Model input** (`input_layout`), concatenated in this order:

| part | value |
|---|---|
| `log1p_mag` (magnitude/mask) | `log1p(|spec|)` |
| `compressed_real`, `compressed_imag` (complex_mask) | `r = |spec| ^ compress`, `theta = atan2(im, re)`: all `F` values `r cos(theta)`, then all `F` values `r sin(theta)` (not interleaved) |
| `log1p_mag` (complex_mask with `mag_input`) | `log1p(|spec|)` of the *uncompressed* spectrum |
| `log1p_long_mag` (long window) | `log1p(long_mag)` |

**Output → spectrum** (`output_layout`):

| output | denoised bin |
|---|---|
| magnitude (`log1p_mag`) | magnitude `max(expm1(pred), 0)`, phase of `spec` |
| mask (`gain`, 0..1) | magnitude `|spec| * pred`, phase of `spec` |
| complex_mask (`mask_real`, `mask_imag`) | below |

complex_mask, per bin `k` with `m = (pred[k], pred[F+k])`:

```
if mask_bound != 0:
    size  = sqrt(mr*mr + mi*mi + 1e-8)
    scale = (mask_bound * tanh(size / mask_bound)) / size
    m *= scale
z  = m * c          # complex multiply with the compressed noisy bin c = (x[k], x[F+k]) from the input frame
out = |z| ^ decompress at angle atan2(z)
```

"Magnitude at the phase of" is always `torch.polar`: `mag * cos(theta)`,
`mag * sin(theta)` with `theta = atan2(im, re)` of the source bin, not
`z * (mag / |z|)` (NaN at `|z| = 0`; there `theta = 0`).

**Synthesis**: `frame = irfft(out, n_fft) * w`. A real inverse FFT ignores the
imaginary parts of the DC and Nyquist bins (complex_mask makes them non-zero:
drop them, as `torch.fft.irfft` does). Overlap-add with hop, and divide by the
overlap-added `w^2`:

- causal, live (`stream.py`): `pending += frame`, `envelope += w^2`; output
  `pending[0:hop] / max(envelope[0:hop], 1e-8)`, then shift both left by `hop`
  (zeros in). One `hop` in, one `hop` out.
- centred, offline (`torch.istft`): overlap-add all `T` frames into
  `(T-1) hop + n_fft` samples, divide by the envelope (no clamp; torch
  refuses if it is < 1e-11), keep samples `[n_fft/2, n_fft/2 + N)`.

### Streaming

`process()` output is the offline output delayed by `stream_delay = n_fft - hop`
samples: `streamed[i + n_fft - hop] == offline[i]` (Python: within 3e-7).
Frame `k` of the stream (the `k`-th call after reset) is row `k` of the
offline `x`/`pred`. `StreamingDenoiser.latency` says `n_fft`: that is the
delay seen by a host, which also has to wait for a whole hop to arrive
before processing it.

**The pre-roll.** The first `n_fft - hop` output samples after a reset belong
to time before the input started (offline trims them). They are the model's
output for silence divided by the tiny start of the window envelope, so they
can be large: up to ~280 in the golden cases (untrained models; a trained model
gives much less, but not necessarily 0). A plugin should output silence for
them; they are in the golden `streamed` so a port can still be compared
sample for sample, but with a looser tolerance (below).

## Vectors file (`"kind": "denoiser-vectors"`) and the golden set

`plugin/tests/golden/` holds, per case `<name>`:

- `<name>.model.bin`: the exported model (as above)
- `<name>.vectors.bin`: its vectors on the same synthetic clip (1600 samples = 0.1 s
  at 16 kHz, peak ~0.4: a formant-shaped harmonic sound in bursts plus noise)
- `index.json`: every case with what it exercises, its sizes and its tensors'
  shapes. A C++ test can also just hard-code the case names.

Header keys of a vectors file: `case`, `model_file`, `n_samples` (`N`),
`n_frames` (`T`), `stream` (`null` for centred models, else `delay`, `chunks`,
`input_zeros`, `max_abs_diff_vs_offline`), and for SNNs `snn`
(`spike_rate_per_layer`, `min_abs_v_minus_threshold`, `spike_frames`).

| tensor | shape | what |
|---|---|---|
| `input` | `[N]` | the input waveform |
| `window`, `long_window` | `[n_fft]`, `[L]` | `torch.hann_window` (only in cases with `spec`) |
| `spec` | `[T, 2F]` | short noisy spectrum: real parts then imaginary parts (some cases) |
| `long_mag` | `[T, F]` | pooled long-window magnitudes, before `log1p` (cases with a long window) |
| `x` | `[T, n_in]` | model input frames (`pipeline.run_model`) |
| `pred` | `[T, n_out]` | model output frames |
| `out_spec` | `[T, 2F]` | denoised spectrum before the inverse FFT: real then imaginary (cases with `spec`) |
| `offline` | `[N]` | `pipeline.enhance` on the whole clip |
| `streamed` | `[chunks * hop]` | raw `StreamingDenoiser.process` output, hop by hop, for `input` followed by `input_zeros` zeros (causal only) |
| `tcn.h0` .. `tcn.h{B}` | `[T, C]` | TCN: channels after the input layer (`h0`) and after each block, per clip frame |
| `snn.spikes{i}` | `[spike_frames * steps, h_i]` | SNN: spikes of layer `i` at every time step of the first frames |

Frame counts: causal `T = floor((N + n_fft - hop) / hop) + 1`; the stream
makes `chunks = ceil((N + n_fft - hop) / hop)` frames, which is `T` or `T - 1`
(then the last offline frame only covers samples after the clip).
Centred: `T = floor(N / hop) + 1`.

| case | exercises |
|---|---|
| `mlp_magnitude_default` | the repo's default STFT: n_fft 128, hop 80, long window 1280 (not a power of two); relu, expm1; `spec`, `out_spec`, windows |
| `mlp_magnitude_short` | no long window, hop 64 (stream makes `T - 1` frames) |
| `mlp_mask_default` | mask (sigmoid), gain on the noisy magnitude |
| `mlp_complex_default` | complex_mask with the repo defaults: compress 0.3, `mag_input`, mask_bound 1.0; `spec`, `out_spec` |
| `mlp_complex_plain` | complex_mask without `mag_input` and bound (old checkpoints); n_fft 256, hop 128 (50% overlap) |
| `mlp_complex_long4` | compress 0.5, mask_bound 1.5 (no baked-in constants); n_fft 64, long window 256 |
| `tcn_mask_short` | TCN kernel 2, context 16 |
| `tcn_magnitude_default` | TCN with the default dilations 1..32: context 127 frames > the clip's 21 frames, so start-up padding is in every frame; `tcn.h*` |
| `tcn_complex_long3` | TCN complex_mask; odd long_factor 3 (pooling padding 1, long window 384) |
| `tcn_complex_plain` | TCN complex_mask without `mag_input` and bound |
| `snn_complex_default` | SNN with the default STFT; `snn.spikes*` of the first 8 frames |
| `snn_complex_recurrent` | recurrent weights |
| `snn_complex_steps3` | 3 time steps per frame; `snn.spikes*` of the first 4 frames (12 steps) |
| `snn_mask_recurrent_steps2` | one layer, recurrent, 2 steps per frame, mask |
| `snn_magnitude_nodecay` | decay 0 (stored as ~1e-4), magnitude |
| `offline_mlp_mask_centred` | centred frames (realtime = false): reflect padding, istft synthesis, with the 1280 long window; `spec`, `out_spec` |
| `offline_tcn_complex_centred` | centred frames with a TCN and complex_mask |

All models are untrained, from fixed seeds, with small hidden sizes (MLP
24/16, TCN 12 channels, SNN 24/16 with threshold 0.1 so that 30-50% of the
neurons spike per step). In every SNN case the membrane potentials stay at
least 3e-4 away from the threshold (`min_abs_v_minus_threshold`;
`make_golden.py` fails below 1e-5), far more than float32 rounding moves
them, so a port should match its spikes exactly.

## Tolerances

Errors below are `max |port - golden| / max(1, max |golden|)` over a tensor.

What the Python side itself achieves:

- `streamed` vs `offline` (shifted by `n_fft - hop`): at most 2.4e-7
- `reference.py` (NumPy, float32, one hop at a time, incremental TCN) vs golden:
  `x` ≤ 3.5e-6 (the `pow` of complex_mask inputs; others ≤ 2e-7),
  `long_mag`/`spec`/`out_spec` ≤ 9e-7, `pred` ≤ 4e-7, `tcn.h*` ≤ 4e-7,
  `offline` ≤ 5.4e-7, `streamed` after the pre-roll ≤ 4.8e-7,
  pre-roll ≤ 1.3e-5, spikes: 0 mismatches

What a C++ port should be held to:

| check | tolerance |
|---|---|
| `x`, `long_mag`, `spec`, `out_spec` | 1e-5 |
| `pred` from the golden `x` (each network on its own) | 1e-5 |
| `pred`, `offline`, `streamed[n_fft - hop:]` end to end | 1e-5 |
| `streamed[0 : n_fft - hop]` (pre-roll) | 1e-4, or skip it |
| `snn.spikes*` (from the golden `x`) | exactly equal |
| `tcn.h*` (from the golden `x`) | 1e-5 |

`check_golden.py` holds `reference.py` to exactly these numbers.

**Reproducibility.** `make_golden.py` runs single-threaded; run twice on one
machine it writes identical bytes. On another CPU or PyTorch version the
vectorised `cos`/`exp`/BLAS kernels can differ in the last bits, so
regenerated files may differ by ~1e-7 relative (`check_golden.py` accepts
1e-6 for its PyTorch recomputation). `index.json` records the versions used.
