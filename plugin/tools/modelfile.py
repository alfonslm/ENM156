"""
The exported model file: a tensorfile (see tensorfile.py) whose header says
how to rebuild and run the network without knowing anything about PyTorch,
and whose tensors are named and laid out in the order the forward pass uses
them. Specification: plugin/tools/README.md, "Model file".

  header, tensors = to_export(model, cfg)    # from a model built by denoiser.model
  write(path, model, cfg)
  header, tensors = read(path)                # back, as numpy arrays
  model, cfg = rebuild(path)                  # back, as the PyTorch model
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from denoiser.config import MODEL_SECTIONS  # noqa: E402
from denoiser.model import make_model, model_type  # noqa: E402
from denoiser.pipeline import io_sizes, is_complex, mag_input, n_freq, out_activation  # noqa: E402

import tensorfile  # noqa: E402

KIND = "denoiser-model"
KIND_VERSION = 1


class Unsupported(SystemExit):
    """A model the plugin can't run."""


def check_supported(cfg):
    if cfg["model"].get("filterbank"):
        raise Unsupported(f"model.filterbank is set ({cfg['model']['filterbank']!r}): the plugin only "
                          "supports STFT models; filterbank models can't be exported")


def layouts(cfg):
    """What each value of an input and an output frame is, as [[part, size], ...] in order."""
    f = n_freq(cfg)
    if is_complex(cfg):
        inp = [["compressed_real", f], ["compressed_imag", f]]
        if mag_input(cfg):
            inp.append(["log1p_mag", f])
        out = [["mask_real", f], ["mask_imag", f]]
    else:
        inp = [["log1p_mag", f]]
        out = [["log1p_mag", f]] if cfg["model"]["output"] == "magnitude" else [["gain", f]]
    if cfg["stft"]["long_factor"]:
        inp.append(["log1p_long_mag", f])
    return inp, out


def derived(cfg, model):
    stft = cfg["stft"]
    n_in, n_out = io_sizes(cfg)
    inp, out = layouts(cfg)
    complex_mask = is_complex(cfg)
    return {
        "sample_rate": cfg["data"]["sample_rate"],
        "n_fft": stft["n_fft"],
        "hop": stft["hop"],
        "long_factor": stft["long_factor"],
        "long_n_fft": stft["n_fft"] * stft["long_factor"],
        "n_freq": n_freq(cfg),
        "n_in": n_in,
        "n_out": n_out,
        "input_layout": inp,
        "output_layout": out,
        "output": cfg["model"]["output"],
        "out_activation": out_activation(cfg) or "none",
        # complex_mask only (null otherwise): the magnitude exponent, its inverse
        # as Python computes it (1 / compress in double, then used as float32),
        # and the soft limit of the mask (0 = none)
        "compress": cfg["model"]["compress"] if complex_mask else None,
        "decompress": 1 / cfg["model"]["compress"] if complex_mask else None,
        "mask_bound": cfg["model"]["mask_bound"] if complex_mask else None,
        "context_frames": model.context,
        "framing": "causal" if stft["causal"] else "centred",
        # samples of process() output that come before the output of the first
        # input sample: streamed[i + stream_delay] == offline[i]
        "stream_delay": stft["n_fft"] - stft["hop"] if stft["causal"] else None,
    }


def _mat(t):
    """Conv1d weight with kernel 1 [out, in, 1] -> [out, in]; anything else as is."""
    t = t.detach().float().cpu()
    return t[..., 0] if t.dim() == 3 and t.shape[-1] == 1 else t


def net_and_tensors(model, cfg):
    """-> (description of the network for the header, [(name, tensor)] in forward order)."""
    kind = model_type(cfg)
    tensors = []
    if kind == "mlp":
        linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
        layers = []
        for i, lin in enumerate(linears):
            last = i == len(linears) - 1
            tensors += [(f"linear{i}.weight", _mat(lin.weight)), (f"linear{i}.bias", _mat(lin.bias))]
            layers.append({"in": lin.in_features, "out": lin.out_features,
                           "activation": (out_activation(cfg) or "none") if last else "relu"})
        return {"type": "mlp", "layers": layers}, tensors
    if kind == "tcn":
        tensors += [("input.weight", _mat(model.inp.weight)), ("input.bias", _mat(model.inp.bias))]
        blocks = []
        for b, block in enumerate(model.blocks):
            tensors += [(f"block{b}.norm.weight", _mat(block.norm.weight)),
                        (f"block{b}.norm.bias", _mat(block.norm.bias)),
                        (f"block{b}.conv.weight", _mat(block.conv.weight)),
                        (f"block{b}.conv.bias", _mat(block.conv.bias)),
                        (f"block{b}.mix.weight", _mat(block.mix.weight)),
                        (f"block{b}.mix.bias", _mat(block.mix.bias))]
            blocks.append({"dilation": block.conv.dilation[0], "trim": block.trim})
        tensors += [("final_norm.weight", _mat(model.norm.weight)), ("final_norm.bias", _mat(model.norm.bias)),
                    ("output.weight", _mat(model.out.weight)), ("output.bias", _mat(model.out.bias))]
        return {"type": "tcn", "channels": model.inp.out_channels, "kernel": model.blocks[0].conv.kernel_size[0],
                "blocks": blocks, "layer_norm_eps": model.norm.eps, "context": model.context}, tensors
    # snn
    for i, lif in enumerate(model.neurons):
        tensors += [(f"layer{i}.weight", _mat(model.inputs[i].weight)), (f"layer{i}.bias", _mat(model.inputs[i].bias)),
                    (f"layer{i}.decay", lif.decay.detach().float())]
        if model.recurrent is not None:
            tensors.append((f"layer{i}.recurrent", _mat(model.recurrent[i].weight)))
    tensors += [("readout.weight", _mat(model.out.weight)), ("readout.decay", model.readout.decay.detach().float()),
                ("output.bias", _mat(model.bias))]
    thresholds = {lif.threshold for lif in model.neurons}
    return {"type": "snn", "hidden": list(model.hidden), "recurrent": model.recurrent is not None,
            "threshold": thresholds.pop(), "steps_per_frame": model.steps_per_frame}, tensors


def to_export(model, cfg):
    """(header without "tensors", [(name, float32 array)])."""
    check_supported(cfg)
    net, tensors = net_and_tensors(model, cfg)
    header = {
        "kind": KIND,
        "kind_version": KIND_VERSION,
        # false for centred (stft.causal = false) models: they look half a window
        # ahead, so a real-time host must refuse them; offline (ARA) can run them
        "realtime": bool(cfg["stft"]["causal"]),
        "derived": derived(cfg, model),
        "net": net,
        # the settings the checkpoint stores, as load_checkpoint completed them
        "config": {s: copy.deepcopy(cfg[s]) for s in MODEL_SECTIONS},
    }
    return header, [(name, t.numpy()) for name, t in tensors]


def write(path, model, cfg):
    header, tensors = to_export(model, cfg)
    tensorfile.write(path, header, tensors)
    return header


def read(path):
    header, tensors = tensorfile.read(path)
    if header.get("kind") != KIND:
        raise ValueError(f"{path}: not a model file (kind {header.get('kind')!r})")
    if header["kind_version"] != KIND_VERSION:
        raise ValueError(f"{path}: model file version {header['kind_version']}, this reader knows {KIND_VERSION}")
    return header, tensors


def _raw_decay(decay):
    """float32 raw values whose float32 sigmoid is exactly `decay`, so a
    rebuilt LIF/Leaky layer decays exactly like the exported one."""
    decay = torch.as_tensor(decay, dtype=torch.float32)
    d = decay.double()
    raw = torch.log(d / (1 - d)).float()
    for _ in range(64):
        got = torch.sigmoid(raw)
        if torch.equal(got, decay):
            return raw
        step = torch.where(got < decay, torch.full_like(raw, math.inf), torch.full_like(raw, -math.inf))
        raw = torch.where(got == decay, raw, torch.nextafter(raw, step))
    raise ValueError("no float32 raw decay gives these decays exactly")


def rebuild(path):
    """The PyTorch model and the settings, from an exported file alone.
    -> (model in eval mode, cfg with data.sample_rate and the model sections)."""
    header, tensors = read(path)
    cfg = copy.deepcopy(header["config"])
    cfg["data"] = {"sample_rate": header["derived"]["sample_rate"]}
    model = make_model(cfg)
    t = {name: torch.from_numpy(a) for name, a in tensors.items()}
    state = {}
    kind = header["net"]["type"]
    if kind == "mlp":
        linears = [i for i, m in enumerate(model.net) if isinstance(m, torch.nn.Linear)]
        for i, idx in enumerate(linears):
            state[f"net.{idx}.weight"], state[f"net.{idx}.bias"] = t[f"linear{i}.weight"], t[f"linear{i}.bias"]
    elif kind == "tcn":
        state["inp.weight"], state["inp.bias"] = t["input.weight"][..., None], t["input.bias"]
        for b in range(len(model.blocks)):
            for part in ("norm.weight", "norm.bias", "conv.weight", "conv.bias", "mix.bias"):
                state[f"blocks.{b}.{part}"] = t[f"block{b}.{part}"]
            state[f"blocks.{b}.mix.weight"] = t[f"block{b}.mix.weight"][..., None]
        state["norm.weight"], state["norm.bias"] = t["final_norm.weight"], t["final_norm.bias"]
        state["out.weight"], state["out.bias"] = t["output.weight"][..., None], t["output.bias"]
    else:
        for i in range(len(model.neurons)):
            state[f"inputs.{i}.weight"], state[f"inputs.{i}.bias"] = t[f"layer{i}.weight"], t[f"layer{i}.bias"]
            state[f"neurons.{i}.raw_decay"] = _raw_decay(t[f"layer{i}.decay"])
            if model.recurrent is not None:
                state[f"recurrent.{i}.weight"] = t[f"layer{i}.recurrent"]
        state["out.weight"], state["readout.raw_decay"] = t["readout.weight"], _raw_decay(t["readout.decay"])
        state["bias"] = t["output.bias"]
    model.load_state_dict(state)  # strict: every parameter must be in the file
    return model.eval(), cfg
