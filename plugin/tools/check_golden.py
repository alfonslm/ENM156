"""
Checks every golden case in plugin/tests/golden/ (or DIR) against its own
files, two ways:

  torch: the model rebuilt from <case>.model.bin alone (modelfile.rebuild),
         run through denoiser/ on the stored input, must give the stored
         vectors (bit for bit on the machine that wrote them; within 1e-6 elsewhere)
  numpy: reference.py, written from the file format only, one hop at a time
         (incremental TCN, SNN state), must match within the tolerances a
         C++ port is held to (plugin/tools/README.md, "Tolerances")

  python plugin/tools/check_golden.py [DIR]

Prints the largest differences per case; exits 1 if any check fails.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import modelfile  # noqa: E402
import tensorfile  # noqa: E402
import vectors  # noqa: E402
from make_golden import GOLDEN  # noqa: E402
from reference import Reference  # noqa: E402

TORCH_TOL = 1e-6  # relative to max(1, largest stored value); the pre-roll gets 100 times that
# numpy reference vs golden, relative to max(1, largest stored value)
NUMPY_TOL = {"spec": 1e-5, "long_mag": 1e-5, "x": 1e-5, "pred": 1e-5, "out_spec": 1e-5,
             "offline": 1e-5, "streamed": 1e-5, "streamed_preroll": 1e-4, "tcn": 1e-5}


def rel_err(a, b):
    return float(np.abs(np.asarray(a, np.float64) - b).max() / max(1.0, float(np.abs(b).max())))


def as_frames(rows, i):
    return np.stack([r[i] for r in rows])


def complex_frames(rows, i):
    z = as_frames(rows, i)
    return np.concatenate([z.real, z.imag], axis=1)


def check_torch(case, model_path, header, stored):
    model, cfg = modelfile.rebuild(model_path)
    debug = {"spectra": "spec" in stored, "tcn_debug": "tcn.h0" in stored,
             "snn_debug_frames": (header.get("snn") or {}).get("spike_frames", 0)}
    info, tensors = vectors.compute(model, cfg, stored["input"], **debug)
    errs = {}
    for name, a in tensors:
        if name not in stored or a.shape != stored[name].shape:
            raise SystemExit(f"{case}: tensor {name} missing or of another shape")
        if name == "streamed":  # the pre-roll is ill-conditioned (see check_numpy): relative, looser
            delay = header["stream"]["delay"]
            errs[name] = rel_err(a[delay:], stored[name][delay:])
            errs["streamed_preroll"] = rel_err(a[:delay], stored[name][:delay]) / 100
        else:
            errs[name] = rel_err(a, stored[name])
    if set(stored) != {n for n, _ in tensors}:
        raise SystemExit(f"{case}: stored tensors {sorted(stored)} != computed")
    return errs


def check_numpy(model_path, stored):
    ref = Reference(*modelfile.read(model_path))
    out, rows, streamed = ref.offline(stored["input"])
    if len(rows) != stored["x"].shape[0]:
        raise SystemExit(f"numpy reference made {len(rows)} frames, golden has {stored['x'].shape[0]}")
    got = {"x": as_frames(rows, 2), "pred": as_frames(rows, 3), "offline": out}
    if "spec" in stored:
        got["spec"], got["out_spec"] = complex_frames(rows, 0), complex_frames(rows, 4)
    if "long_mag" in stored:
        got["long_mag"] = as_frames(rows, 1)
    if streamed is not None:
        # the first n_fft - hop samples come before the clip starts: overlap-add divides
        # them by the tiny start of the window envelope, so they are large and ill-conditioned
        # (offline trims them off). Held to a looser, relative tolerance; see README
        delay = ref.n_fft - ref.hop
        got["streamed"] = streamed[delay:]
        got["streamed_preroll"] = streamed[:delay]
    if "tcn.h0" in stored:  # the incremental TCN's channels, frame by frame, from the golden x
        ref.reset()
        hidden = []
        for x in stored["x"]:
            ref.network(x)
            hidden.append([ring[-1].copy() for ring in ref.rings] + [ref.tcn_out])  # each block's newest input, last output
        for b in range(len(ref.rings) + 1):
            got[f"tcn.h{b}"] = np.stack([h[b] for h in hidden])
    want = dict(stored)
    if "streamed" in got:
        delay = len(got["streamed_preroll"])
        want["streamed"], want["streamed_preroll"] = stored["streamed"][delay:], stored["streamed"][:delay]
    errs = {name: rel_err(a[:len(want[name])], want[name]) for name, a in got.items()}
    if "snn.spikes0" in stored:  # spikes of the first frames: exactly equal or not
        steps = ref.net["steps_per_frame"]
        ref.reset()
        ref.spike_log = [[] for _ in ref.net["hidden"]]
        for x in stored["x"][:stored["snn.spikes0"].shape[0] // steps]:  # golden x: only the network under test
            ref.network(x)
        errs["spike_mismatches"] = int(sum((np.stack(log) != stored[f"snn.spikes{i}"]).sum()
                                           for i, log in enumerate(ref.spike_log)))
    return errs


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    torch.set_num_threads(1)
    index = json.loads((root / "index.json").read_text())
    failed = False
    for case in index["cases"]:
        name = case["name"]
        header, stored = tensorfile.read(root / case["vectors_file"])
        model_path = root / case["model_file"]
        t_errs = check_torch(name, model_path, header, stored)
        n_errs = check_numpy(model_path, stored)
        bad = [k for k, e in t_errs.items() if e > TORCH_TOL]
        bad += [k for k, e in n_errs.items() if k != "spike_mismatches" and e > NUMPY_TOL[k.split(".")[0]]]
        bad += ["spikes"] if n_errs.get("spike_mismatches") else []
        failed |= bool(bad)
        print(f"{'ok  ' if not bad else 'FAIL'} {name}: torch max {max(t_errs.values()):.1e}; numpy "
              + ", ".join(f"{k} {v:.1e}" if isinstance(v, float) else f"{k} {v}" for k, v in n_errs.items())
              + (f"  <- {bad}" if bad else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
