"""
An independent NumPy implementation of the whole denoiser, written from the
model file alone (plugin/tools/README.md), without PyTorch or denoiser/:
a blueprint for the C++ port, and the proof that the exported file says
everything a port needs. check_golden.py holds it to the golden vectors.

It works one hop at a time like the plugin will (fixed-size state, the TCN
incremental with one ring buffer per block), and also offline with centred
frames for non-causal (realtime = false) models. Arithmetic is float32 like
the Python reference, except the FFTs (float64 in NumPy, then rounded).

  ref = Reference(*modelfile.read(path))
  out = ref.process(chunk)        # hop samples in, hop samples out
  y = ref.offline(wav)            # whole clip, either framing
"""
import numpy as np

f32 = np.float32


def hann(n):
    """torch.hann_window(n): the periodic Hann window, 0.5 - 0.5 cos(2 pi i / n)."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(f32)


def polar(mag, angle):
    """torch.polar: mag * (cos angle + i sin angle), angle = atan2(imag, real) of the source.
    Not z * (mag / |z|): that is NaN where |z| = 0 and differs in the last bits."""
    return (mag * np.cos(angle)).astype(f32) + 1j * (mag * np.sin(angle)).astype(f32)


def relu(x):
    return np.maximum(x, f32(0))


def activation(x, name):
    if name == "relu":
        return relu(x)
    if name == "sigmoid":
        return (f32(1) / (f32(1) + np.exp(-x))).astype(f32)
    return x


def layer_norm(x, weight, bias, eps):
    """Over the channels of one frame; biased variance, as torch.nn.LayerNorm."""
    mean = x.mean(dtype=f32)
    var = ((x - mean) ** 2).mean(dtype=f32)
    return ((x - mean) / np.sqrt(var + f32(eps)) * weight + bias).astype(f32)


class Reference:
    def __init__(self, header, tensors):
        d, net = header["derived"], header["net"]
        self.d, self.net, self.t = d, net, tensors
        self.n_fft, self.hop, self.lf, self.F = d["n_fft"], d["hop"], d["long_factor"], d["n_freq"]
        self.long_n_fft = d["long_n_fft"]
        self.window = hann(self.n_fft)
        self.spike_log = None  # set to [[] per layer] to collect the SNN's spikes of every time step
        self.long_window = hann(self.long_n_fft) if self.lf else None
        self.reset()

    # ---- state -------------------------------------------------------------
    def reset(self):
        """Everything back to the start of a recording."""
        self.history = np.zeros(max(self.n_fft, self.long_n_fft), f32)
        self.pending = np.zeros(self.n_fft, f32)
        self.envelope = np.zeros(self.n_fft, f32)
        net = self.net
        if net["type"] == "snn":
            self.v = [np.zeros(h, f32) for h in net["hidden"]]
            self.s = [np.zeros(h, f32) for h in net["hidden"]]
            self.u = np.zeros(self.d["n_out"], f32)
        elif net["type"] == "tcn":
            # one ring per block of its last (kernel - 1) * dilation + 1 inputs; any finite
            # start value works, because the context - 1 zero input frames below (what
            # FrameModel.forward pads a clip with) flush every ring with the right values
            c = net["channels"]
            self.rings = [np.zeros((b["trim"] + 1, c), f32) for b in net["blocks"]]
            self.normed = [np.zeros((b["trim"] + 1, c), f32) for b in net["blocks"]]
            zero = np.zeros(self.d["n_in"], f32)
            for _ in range(net["context"] - 1):
                self.network(zero)

    # ---- features ----------------------------------------------------------
    def pooled_long(self, long_spec):
        """features.pool_long_mag for one frame: power of the long_factor times finer
        bins, averaged over long_factor of them (window starting long_factor // 2
        bins before k * long_factor, bins outside the spectrum not counted),
        times long_factor, square root, divided by long_factor."""
        lf, pad = self.lf, self.lf // 2
        power = (np.abs(long_spec) ** 2).astype(f32)
        out = np.empty(self.F, f32)
        for k in range(self.F):
            lo, hi = max(k * lf - pad, 0), min(k * lf - pad + lf, len(power))
            mean = power[lo:hi].sum(dtype=f32) / f32(hi - lo)
            out[k] = np.sqrt(mean * f32(lf)) / f32(lf)
        return out

    def model_input(self, spec, long):
        """pipeline.input_frames for one frame: spec (F,) complex64, long (F,) or None."""
        d = self.d
        log_mag = np.log1p(np.abs(spec)).astype(f32)
        if d["output"] == "complex_mask":
            c = polar(np.abs(spec) ** f32(d["compress"]), np.angle(spec))
            parts = [c.real, c.imag]
            if any(p == "log1p_mag" for p, _ in d["input_layout"]):
                parts.append(log_mag)
        else:
            parts = [log_mag]
        if long is not None:
            parts.append(np.log1p(long).astype(f32))
        return np.concatenate(parts).astype(f32)

    def output_spectrum(self, pred, x, spec):
        """pipeline.output_spec for one frame."""
        d, F = self.d, self.F
        if d["output"] == "magnitude":
            return polar(relu(np.expm1(pred)), np.angle(spec))
        if d["output"] == "mask":
            return polar(np.abs(spec) * pred, np.angle(spec))
        mr, mi = pred[:F], pred[F:]
        if d["mask_bound"]:
            bound = f32(d["mask_bound"])
            size = np.sqrt(mr * mr + mi * mi + f32(1e-8))
            scale = bound * np.tanh(size / bound) / size
            mr, mi = mr * scale, mi * scale
        xr, xi = x[:F], x[F:2 * F]  # the compressed noisy frame
        z = (mr * xr - mi * xi) + 1j * (mr * xi + mi * xr)
        return polar(np.abs(z).astype(f32) ** f32(d["decompress"]), np.angle(z))

    # ---- networks ----------------------------------------------------------
    def network(self, x):
        """One input frame (n_in,) -> one output frame (n_out,), updating the state."""
        t, net = self.t, self.net
        if net["type"] == "mlp":
            h = x
            for i, layer in enumerate(net["layers"]):
                h = activation(t[f"linear{i}.weight"] @ h + t[f"linear{i}.bias"], layer["activation"])
            return h
        if net["type"] == "tcn":
            eps, k = net["layer_norm_eps"], net["kernel"]
            h = t["input.weight"] @ x + t["input.bias"]
            for b, block in enumerate(net["blocks"]):
                p, d = f"block{b}.", block["dilation"]
                ring, normed = self.rings[b], self.normed[b]
                ring[:-1], normed[:-1] = ring[1:].copy(), normed[1:].copy()  # oldest out
                ring[-1] = h
                normed[-1] = relu(layer_norm(h, t[p + "norm.weight"], t[p + "norm.bias"], eps))
                w = t[p + "conv.weight"]  # [out, in, kernel]; tap j sees the frame (kernel-1-j)*d back
                conv = t[p + "conv.bias"].copy()
                for j in range(k):
                    conv += w[:, :, j] @ normed[j * d]
                h = ring[-1] + t[p + "mix.weight"] @ relu(conv) + t[p + "mix.bias"]
            self.tcn_out = h  # the last block's output, for debugging
            h = relu(layer_norm(h, t["final_norm.weight"], t["final_norm.bias"], eps))
            return activation(t["output.weight"] @ h + t["output.bias"], self.d["out_activation"])
        # snn
        threshold = f32(net["threshold"])
        first = t["layer0.weight"] @ x + t["layer0.bias"]  # the same in every step of the frame
        for _ in range(net["steps_per_frame"]):
            below = None
            for i in range(len(net["hidden"])):
                current = first if i == 0 else t[f"layer{i}.weight"] @ below + t[f"layer{i}.bias"]
                if net["recurrent"]:
                    current = current + t[f"layer{i}.recurrent"] @ self.s[i]  # own spikes, step before
                v = current + t[f"layer{i}.decay"] * self.v[i]
                s = (v - threshold > 0).astype(f32)
                self.v[i], self.s[i] = v - s * threshold, s
                if self.spike_log is not None:
                    self.spike_log[i].append(s)
                below = s
            self.u = t["readout.weight"] @ below + t["readout.decay"] * self.u
        return activation(self.u + t["output.bias"], self.d["out_activation"])

    # ---- one hop (causal models) --------------------------------------------
    def frame(self, short, long):
        """Short window samples (n_fft,), long window samples or None -> (x, pred, out_spec)."""
        spec = np.fft.rfft(short * self.window).astype(np.complex64)
        lmag = self.pooled_long(np.fft.rfft(long * self.long_window).astype(np.complex64)) if self.lf else None
        x = self.model_input(spec, lmag)
        pred = self.network(x).astype(f32)
        return spec, lmag, x, pred, self.output_spectrum(pred, x, spec)

    def process(self, chunk):
        """stream.StreamingDenoiser.process: hop samples in, hop samples out."""
        if not self.d["framing"] == "causal":
            raise ValueError("realtime = false: this model can't run live")
        hop = self.hop
        self.history = np.concatenate([self.history[hop:], np.asarray(chunk, f32)])
        *_, out = self.last = self.frame(self.history[-self.n_fft:],
                                         self.history[-self.long_n_fft:] if self.lf else None)
        self.pending += (np.fft.irfft(out, n=self.n_fft) * self.window).astype(f32)  # imag of DC/Nyquist dropped
        self.envelope += self.window ** 2
        done = self.pending[:hop] / np.maximum(self.envelope[:hop], f32(1e-8))
        self.pending = np.concatenate([self.pending[hop:], np.zeros(hop, f32)])
        self.envelope = np.concatenate([self.envelope[hop:], np.zeros(hop, f32)])
        return done.astype(f32)

    # ---- whole clip ----------------------------------------------------------
    def offline(self, wav):
        """pipeline.enhance on a whole clip. Causal: the stream, shifted by n_fft - hop.
        Centred: reflect-padded frames and torch.istft's overlap-add.
        -> (output, x (T, n_in), pred (T, n_out), extras)."""
        wav = np.asarray(wav, f32)
        n, n_fft, hop = len(wav), self.n_fft, self.hop
        self.reset()
        if self.d["framing"] == "causal":
            delay = n_fft - hop
            chunks = (n + delay) // hop + 1  # as many frames as offline makes (>= enough to flush the clip)
            padded = np.concatenate([wav, np.zeros(chunks * hop - n, f32)])
            out, rows = [], []
            for c in range(chunks):
                out.append(self.process(padded[c * hop:(c + 1) * hop]))
                rows.append(self.last)
            streamed = np.concatenate(out)
            return streamed[delay:delay + n], rows, streamed
        # centred: frame t is centred on sample t * hop; mirrored half windows at both ends
        short = np.pad(wav, n_fft // 2, mode="reflect")
        long = np.pad(wav, self.long_n_fft // 2, mode="reflect") if self.lf else None
        n_frames = n // hop + 1
        rows = [self.frame(short[t * hop:t * hop + n_fft],
                           long[t * hop:t * hop + self.long_n_fft] if self.lf else None) for t in range(n_frames)]
        length = (n_frames - 1) * hop + n_fft
        y, env = np.zeros(length, f32), np.zeros(length, f32)
        for t, row in enumerate(rows):
            y[t * hop:t * hop + n_fft] += (np.fft.irfft(row[-1], n=n_fft) * self.window).astype(f32)
            env[t * hop:t * hop + n_fft] += self.window ** 2
        start = n_fft // 2
        return (y[start:start + n] / env[start:start + n]).astype(f32), rows, None
