"""
Test vectors of one model on one clip, computed with the Python reference
(denoiser.pipeline / denoiser.stream), written as a tensorfile of kind
"denoiser-vectors". Layout: plugin/tools/README.md, "Vectors file".
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from denoiser.features import analysis, window  # noqa: E402
from denoiser.pipeline import long_mag, output_spec, run_model  # noqa: E402
from denoiser.snn.nets import SpikingModel  # noqa: E402
from denoiser.stream import StreamingDenoiser  # noqa: E402

import tensorfile  # noqa: E402

KIND = "denoiser-vectors"
KIND_VERSION = 1


def synthetic_clip(sr, seconds, seed):
    """A deterministic speech-like noisy clip, float32 (samples,): a voiced
    sound (harmonics of a gliding pitch, shaped by three formants) in
    syllable-like bursts, plus white and low-frequency noise. Peaks around
    0.4, RMS around -20 dBFS: every bin gets non-trivial values and nothing clips."""
    rng = np.random.default_rng(seed)
    n = int(round(sr * seconds))
    t = np.arange(n) / sr
    f0 = 140 + 35 * np.sin(2 * np.pi * 1.7 * t + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    formants = [(rng.uniform(450, 750), 120), (rng.uniform(1100, 1800), 180), (rng.uniform(2300, 3000), 250)]
    voiced = np.zeros(n)
    for k in range(1, int(sr / 2 / 105)):
        freq = k * f0
        amp = sum(np.exp(-0.5 * ((freq - fc) / bw) ** 2) for fc, bw in formants) + 0.05 / k
        voiced += np.where(freq < sr / 2, amp, 0) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    envelope = 0.15 + 0.85 * np.sin(np.pi * 4.5 * t + rng.uniform(0, np.pi)) ** 2  # syllables
    speech = 0.3 * voiced / np.abs(voiced).max() * envelope
    hum = np.convolve(rng.standard_normal(n + 63), np.hanning(64) / 32, mode="valid")[:n]
    noise = 0.02 * rng.standard_normal(n) + 0.03 * hum
    return (speech + noise).astype(np.float32)


def stream_raw(model, cfg, wav):
    """process() output for wav followed by zeros up to a whole number of hops
    that flushes the last input sample out. -> (output, chunks)."""
    n_fft, hop = cfg["stft"]["n_fft"], cfg["stft"]["hop"]
    chunks = -(-(len(wav) + n_fft - hop) // hop)
    padded = torch.cat([wav, torch.zeros(chunks * hop - len(wav))])
    stream = StreamingDenoiser(model, cfg, torch.device("cpu"))
    return torch.cat([stream.process(c) for c in padded.split(hop)]), chunks


def frames_c(spec):
    """complex (F, T) -> real (T, 2F): real parts, then imaginary parts."""
    return torch.cat([spec.real, spec.imag], dim=0).T


def tcn_hidden(model, x):
    """The TCN's channels for each clip frame: after the input layer, then
    after each block. -> [(name, (T, C))]."""
    n = x.shape[0]
    h = model.inp(F.pad(x, (0, 0, model.context - 1, 0)).T[None])
    out = [("tcn.h0", h[0, :, -n:].T)]
    for b, block in enumerate(model.blocks):
        h = block(h)
        out.append((f"tcn.h{b + 1}", h[0, :, -n:].T))
    return out


def snn_trace(model, x):
    """Runs the spiking network frame by frame like LIFNet.step, also keeping
    each layer's spikes of every time step and how close the membrane
    potentials came to the threshold. -> (pred (T, n_out), [spikes (T*steps, h)], min |v - threshold|)."""
    state = model.init_state((), x.device)
    v, s, u = list(state["v"]), list(state["s"]), state["u"]
    spikes = [[] for _ in model.hidden]
    margin, preds = math.inf, []
    for frame in x:
        first = model.inputs[0](frame)
        for _ in range(model.steps_per_frame):
            below = None
            for i, lif in enumerate(model.neurons):
                current = first if i == 0 else model.inputs[i](below)
                if model.recurrent is not None:
                    current = current + model.recurrent[i](s[i])
                before = torch.addcmul(current, lif.decay, v[i])
                margin = min(margin, (before - lif.threshold).abs().min().item())
                s[i], v[i] = lif.step(current, v[i])
                spikes[i].append(s[i])
                below = s[i]
            y, u = model.readout.step(model.out(below), u)
        preds.append(model.out_act(y + model.bias))
    return torch.stack(preds), [torch.stack(sp) for sp in spikes], margin


def compute(model, cfg, wav, spectra=False, tcn_debug=False, snn_debug_frames=0):
    """Everything a vectors file holds. wav: float32 numpy (samples,).
    -> (header info dict, [(name, array)])."""
    stft = cfg["stft"]
    model.eval()
    wav_t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    info = {"n_samples": len(wav)}
    with torch.no_grad():
        offline, pred, x = run_model(model, wav_t, cfg)
        spec = analysis(wav_t, stft["n_fft"], stft)
        info["n_frames"] = x.shape[0]
        tensors = [("input", wav_t)]
        if spectra:
            tensors.append(("window", window(stft["n_fft"], "cpu")))
            if stft["long_factor"]:
                tensors.append(("long_window", window(stft["n_fft"] * stft["long_factor"], "cpu")))
            tensors.append(("spec", frames_c(spec)))
        if stft["long_factor"]:
            tensors.append(("long_mag", long_mag(wav_t, cfg)))
        tensors += [("x", x), ("pred", pred)]
        if spectra:
            tensors.append(("out_spec", frames_c(output_spec(pred, x, spec, cfg))))
        tensors.append(("offline", offline))
        if stft["causal"]:
            streamed, chunks = stream_raw(model, cfg, wav_t)
            delay = stft["n_fft"] - stft["hop"]
            info["stream"] = {"delay": delay, "chunks": chunks, "input_zeros": chunks * stft["hop"] - len(wav),
                              "max_abs_diff_vs_offline":
                                  float((streamed[delay:delay + len(wav)] - offline).abs().max())}
            tensors.append(("streamed", streamed))
        else:
            info["stream"] = None
        if tcn_debug:
            tensors += tcn_hidden(model, x)
        if isinstance(model, SpikingModel):
            trace_pred, spikes, margin = snn_trace(model, x)
            steps = model.steps_per_frame
            info["snn"] = {
                "spike_rate_per_layer": [round(float(sp.mean()), 4) for sp in spikes],
                "min_abs_v_minus_threshold": float(margin),
                "step_vs_forward_max_abs_diff": float((trace_pred - pred).abs().max()),
                "spike_frames": snn_debug_frames,
            }
            for i, sp in enumerate(spikes):
                if snn_debug_frames:
                    tensors.append((f"snn.spikes{i}", sp[:snn_debug_frames * steps]))
    return info, [(name, t.detach().numpy() if torch.is_tensor(t) else t) for name, t in tensors]


def write_vectors(path, model, cfg, wav, case, model_file, **debug):
    info, tensors = compute(model, cfg, wav, **debug)
    header = {"kind": KIND, "kind_version": KIND_VERSION, "case": case, "model_file": model_file, **info}
    tensorfile.write(path, header, tensors)
    return {**info, "tensors": {name: list(np.shape(a)) for name, a in tensors}}
