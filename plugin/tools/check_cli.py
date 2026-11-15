"""
Checks the C++ command-line tool against Python at full size: untrained
models with config.toml's default sizes (MLP 1024 x 1024, SNN 256 x 3),
exported with modelfile.write, run by the built `denoise` on a synthetic
clip (float WAV), compared with denoiser.pipeline.enhance on the same clip.
No checkpoint or dataset needed.

  python plugin/tools/check_cli.py plugin/build/cli/denoise [--seconds 12] [--out DIR]

Prints, per model and mode (offline, --stream), the max abs error and the
error relative to max(1, max |python|) (the measure of tools/README.md), the
SNN's spike rate and how close its membrane potentials come to the
threshold, and the tool's real-time factor. Exit status 1 if any relative
error is above 1e-4, except for SNN runs that come within 1e-5 of the
threshold: there a spike can flip between any two float implementations,
and such a run is reported but not counted.

Only reads denoiser/; changes nothing there.
"""
import argparse
import copy
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from denoiser.config import load_config, merge  # noqa: E402
from denoiser.model import describe_model, make_model  # noqa: E402
from denoiser.pipeline import enhance, features  # noqa: E402

import modelfile  # noqa: E402
import vectors  # noqa: E402

TOLERANCE = 1e-4
SPIKE_FLIP_MARGIN = 1e-5  # an SNN run that comes this close to its threshold may flip a spike

# (name, settings changed from config.toml): the network sizes stay at config.toml's defaults
CASES = [
    ("mlp_magnitude", {"model": {"type": "mlp", "output": "magnitude"}}),
    ("mlp_complex_mask", {"model": {"type": "mlp", "output": "complex_mask"}}),
    ("snn_complex_mask", {"model": {"type": "snn", "output": "complex_mask"}}),
    ("snn_mask", {"model": {"type": "snn", "output": "mask"}}),
    ("tcn_magnitude", {"model": {"type": "tcn", "output": "magnitude"}}),
    ("tcn_complex_mask", {"model": {"type": "tcn", "output": "complex_mask"}}),
    # untrained, the default threshold (1.0) gives only ~1-2% spikes; 0.1 gives ~40%, closer to a trained
    # network, and so many more chances for a spike to flip between C++ and PyTorch
    ("snn_complex_mask_threshold0.1", {"model": {"type": "snn", "output": "complex_mask"}, "snn": {"threshold": 0.1}}),
    ("snn_mask_recurrent_threshold0.1", {"model": {"type": "snn", "output": "mask"},
                                         "snn": {"threshold": 0.1, "recurrent": True}}),
]


def write_wav_float(path, sr, channels):
    """32-bit float WAV, interleaved; channels: list of equal-length float32 arrays."""
    data = np.stack(channels, axis=1).astype("<f4").tobytes()
    n_ch = len(channels)
    fmt = struct.pack("<HHIIHHH", 3, n_ch, sr, sr * 4 * n_ch, 4 * n_ch, 32, 0)
    fact = struct.pack("<I", len(channels[0]))
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"fact" + struct.pack("<I", 4) + fact
            + b"data" + struct.pack("<I", len(data)) + data)
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def read_wav_float(path):
    """-> (sr, [channel arrays]) for a 32-bit float WAV as the tool writes it with --float."""
    b = Path(path).read_bytes()
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", path
    pos, sr, n_ch, data = 12, None, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        body = b[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            code, n_ch, sr = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            assert code == 3 and bits == 32, "expected a float WAV"
        elif cid == b"data":
            data = np.frombuffer(body, dtype="<f4")
        pos += 8 + size + (size & 1)
    return sr, [data[c::n_ch].astype(np.float32) for c in range(n_ch)]


def snn_margin(path, x):
    """The SNN of model file `path` on input frames x (T, n_in), in NumPy float32
    as tools/README.md specifies it -> min |v - threshold| over every neuron and
    time step: how close the run comes to a spike that rounding could flip."""
    header, t = modelfile.read(path)
    net = header["net"]
    threshold = np.float32(net["threshold"])
    v = [np.zeros(h, np.float32) for h in net["hidden"]]
    s = [np.zeros(h, np.float32) for h in net["hidden"]]
    closest = np.inf
    for frame in x:
        first = t["layer0.weight"] @ frame + t["layer0.bias"]
        for _ in range(net["steps_per_frame"]):
            below = None
            for i in range(len(net["hidden"])):
                cur = first if i == 0 else t[f"layer{i}.weight"] @ below + t[f"layer{i}.bias"]
                if net["recurrent"]:
                    cur = cur + t[f"layer{i}.recurrent"] @ s[i]
                vi = cur + t[f"layer{i}.decay"] * v[i]
                closest = min(closest, float(np.abs(vi - threshold).min()))
                s[i] = (vi - threshold > 0).astype(np.float32)
                v[i], below = vi - s[i] * threshold, s[i]
    return closest


def run_tool(tool, args):
    p = subprocess.run([str(tool), *map(str, args)], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"{tool} {' '.join(map(str, args))} failed ({p.returncode}):\n{p.stderr}")
    return p.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tool", type=Path, help="the built denoise executable")
    ap.add_argument("--seconds", type=float, default=12.0, help="length of the test clip (default 12)")
    ap.add_argument("--out", type=Path, help="folder for the model and WAV files (default: a temporary one)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = args.out or Path(tempfile.mkdtemp(prefix="check_cli_"))
    out.mkdir(parents=True, exist_ok=True)

    base = load_config()
    sr = base["data"]["sample_rate"]
    # two different clips, as the two channels of one stereo file
    clips = [vectors.synthetic_clip(sr, args.seconds, seed=s) for s in (11, 12)]
    wav_in = out / "in.wav"
    write_wav_float(wav_in, sr, clips)
    print(f"clip: {args.seconds} s at {sr} Hz, stereo (two different synthetic clips), in {out}")

    worst, flips = 0.0, []
    for seed, (name, settings) in enumerate(CASES):
        cfg = merge(copy.deepcopy(base), copy.deepcopy(settings))
        torch.manual_seed(100 + seed)
        model = make_model(cfg).eval()
        model_path = out / f"{name}.model.bin"
        header = modelfile.write(model_path, model, cfg)
        print(f"\n{name}: {describe_model(cfg)}")
        with torch.no_grad():
            refs = [enhance(model, torch.from_numpy(c), cfg).numpy() for c in clips]
        margins = [None, None]
        if cfg["model"]["type"] == "snn":
            enhance(model, torch.from_numpy(clips[0]), cfg)
            print(f"  python spike rate (clip 1): {float(model.spike_rate.detach()):.3f} per neuron per step")
            with torch.no_grad():
                margins = [snn_margin(model_path, features(torch.from_numpy(c)[None], cfg)[1][0].numpy())
                           for c in clips]
            print("  closest membrane potential to the threshold: "
                  + ", ".join(f"channel {c + 1} {m:.1e}" for c, m in enumerate(margins)))
        modes = [("offline", [])] + ([("--stream", ["--stream"])] if header["realtime"] else [])
        for label, extra in modes:
            wav_out = out / f"{name}.{label.strip('-')}.wav"
            report = run_tool(args.tool, [*extra, "--float", model_path, wav_in, wav_out])
            out_sr, ys = read_wav_float(wav_out)
            assert out_sr == sr and len(ys) == 2
            for c, (y, ref) in enumerate(zip(ys, refs)):
                assert len(y) == len(ref), (len(y), len(ref))
                diff = float(np.abs(y.astype(np.float64) - ref).max())
                rel = diff / max(1.0, float(np.abs(ref).max()))
                note = ""
                if rel > TOLERANCE and margins[c] is not None and margins[c] < SPIKE_FLIP_MARGIN:
                    # a threshold device: a potential within rounding of the threshold can spike in one
                    # implementation and not the other (any two float implementations, not only C++ vs Python)
                    flips.append(f"{name} {label} channel {c + 1}")
                    note = (f"  <- a membrane potential came within {margins[c]:.1e} of the threshold: "
                            "a spike flipped (expected, not counted)")
                else:
                    worst = max(worst, rel)
                print(f"  {label:9s} channel {c + 1}: max |c++ - python| {diff:.2e}, relative {rel:.2e}{note}")
            timing = [line for line in report.splitlines() if line.startswith("time:")]
            if timing:
                print(f"  {label:9s} {timing[0]}")
    print(f"\nworst relative error {worst:.2e} (tolerance {TOLERANCE:g})")
    if flips:
        print(f"spike flips (potential within {SPIKE_FLIP_MARGIN:g} of the threshold), not counted: {', '.join(flips)}")
    if worst > TOLERANCE:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
