"""
Writes the golden test vectors the C++ port is tested against to
plugin/tests/golden/ (or OUT_DIR): per case an exported model file
(<case>.model.bin) and its vectors (<case>.vectors.bin), and index.json
listing every case, what it exercises, and its tensors and shapes.
Untrained models from fixed seeds and a synthetic speech-like clip, so no
dataset or checkpoint is needed. Deterministic: rerunning it on the same
machine and library versions writes the same bytes.

  python plugin/tools/make_golden.py [OUT_DIR]
  python plugin/tools/check_golden.py      # recomputes every case from its files

Layout and tolerances: plugin/tools/README.md.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from denoiser.config import merge  # noqa: E402
from denoiser.model import describe_model, make_model  # noqa: E402

import modelfile  # noqa: E402
import vectors  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden"
SAMPLE_RATE = 16000
SECONDS = 0.1  # 1600 samples: longer than the 1280-sample long window, short enough to keep the folder small
DEFAULT_STFT = {"n_fft": 128, "hop": 80, "long_factor": 10, "causal": True}  # as config.toml
MLP = {"hidden": [24, 16]}  # unequal sizes: a transposed weight can't go unnoticed
TCN_SMALL = {"channels": 12, "kernel": 3, "dilations": [1, 2, 4, 8]}  # context 31 frames
TCN_DEFAULT_CONTEXT = {"channels": 12, "kernel": 3, "dilations": [1, 2, 4, 8, 16, 32]}  # context 127, as config.toml
SNN = {"hidden": [24, 16], "threshold": 0.1}  # low threshold: an untrained network should spike

# Every setting a model file depends on, spelled out (the values of config.toml when this was
# written) instead of read from config.toml / config.local.toml, so the golden files don't change
# when someone's defaults do
BASE = {
    "data": {"sample_rate": SAMPLE_RATE},
    "stft": DEFAULT_STFT,
    "model": {"type": "mlp", "hidden": [1024, 1024], "output": "magnitude", "compress": 0.3, "mag_weight": 0.7,
              "mag_input": True, "mask_bound": 1.0, "filterbank": ""},
    "tcn": {"channels": 256, "kernel": 3, "dilations": [1, 2, 4, 8, 16, 32]},
    "snn": {"hidden": [256, 256, 256], "recurrent": False, "decay": 0.9, "readout_decay": 0.5, "learn_decay": True,
            "threshold": 1.0, "surrogate_slope": 25, "steps_per_frame": 1, "rate_weight": 0.0},
}

# name, settings changed from BASE (data.sample_rate fixed at 16 kHz), what it exercises,
# extra debug tensors. Every case also gets x, pred, offline and (causal ones) streamed.
CASES = [
    ("mlp_magnitude_default", {"stft": DEFAULT_STFT, "model": {"type": "mlp", "output": "magnitude", **MLP}},
     "the repo's default STFT (n_fft 128, hop 80, long window 1280 = not a power of two); relu output, expm1",
     {"spectra": True}),
    ("mlp_magnitude_short", {"stft": {"n_fft": 128, "hop": 64, "long_factor": 0},
                             "model": {"type": "mlp", "output": "magnitude", **MLP}},
     "no long window, hop 64", {}),
    ("mlp_mask_default", {"stft": DEFAULT_STFT, "model": {"type": "mlp", "output": "mask", **MLP}},
     "mask output (sigmoid), noisy magnitude times gain", {}),
    ("mlp_complex_default", {"stft": DEFAULT_STFT, "model": {"type": "mlp", "output": "complex_mask", **MLP}},
     "complex_mask with the repo defaults: compress 0.3, mag_input on, mask_bound 1.0", {"spectra": True}),
    ("mlp_complex_plain", {"stft": {"n_fft": 256, "hop": 128, "long_factor": 0},
                           "model": {"type": "mlp", "output": "complex_mask", "mag_input": False, "mask_bound": 0,
                                     **MLP}},
     "complex_mask as in old checkpoints: real/imag input only, no mask bound; n_fft 256 with hop 128 (50% overlap)", {}),
    ("mlp_complex_long4", {"stft": {"n_fft": 64, "hop": 32, "long_factor": 4},
                           "model": {"type": "mlp", "output": "complex_mask", "compress": 0.5, "mask_bound": 1.5,
                                     **MLP}},
     "complex_mask with compress 0.5 and mask_bound 1.5 (no constants baked in); n_fft 64, long window 256",
     {"spectra": True}),
    ("tcn_mask_short", {"stft": {"n_fft": 128, "hop": 64, "long_factor": 0},
                        "model": {"type": "tcn", "output": "mask"}, "tcn": {**TCN_SMALL, "kernel": 2}},
     "TCN with kernel 2 (context 16 frames)", {}),
    ("tcn_magnitude_default", {"stft": DEFAULT_STFT, "model": {"type": "tcn", "output": "magnitude"},
                               "tcn": TCN_DEFAULT_CONTEXT},
     "TCN with the default dilations 1..32: context 127 frames, longer than the whole clip (21 frames), "
     "so every frame depends on the zero start-up frames; per-block channels in tcn.h*",
     {"tcn_debug": True}),
    ("tcn_complex_long3", {"stft": {"n_fft": 128, "hop": 64, "long_factor": 3},
                           "model": {"type": "tcn", "output": "complex_mask"}, "tcn": TCN_SMALL},
     "TCN complex_mask with mag_input and mask_bound; odd long_factor 3 (pooling padding 1, long window 384)",
     {}),
    ("tcn_complex_plain", {"stft": {"n_fft": 128, "hop": 64, "long_factor": 0},
                           "model": {"type": "tcn", "output": "complex_mask", "mag_input": False, "mask_bound": 0},
                           "tcn": TCN_SMALL},
     "TCN complex_mask without mag_input and mask bound", {}),
    ("snn_complex_default", {"stft": DEFAULT_STFT, "model": {"type": "snn", "output": "complex_mask"}, "snn": SNN},
     "SNN with the default STFT and complex_mask; spikes of each layer for the first 8 frames in snn.spikes*",
     {"snn_debug_frames": 8}),
    ("snn_complex_recurrent", {"stft": {"n_fft": 128, "hop": 64, "long_factor": 0},
                               "model": {"type": "snn", "output": "complex_mask"}, "snn": {**SNN, "recurrent": True}},
     "SNN with recurrent weights", {}),
    ("snn_complex_steps3", {"stft": {"n_fft": 128, "hop": 64, "long_factor": 0},
                            "model": {"type": "snn", "output": "complex_mask"}, "snn": {**SNN, "steps_per_frame": 3}},
     "SNN with 3 time steps per frame; spikes of the first 4 frames (12 steps)", {"snn_debug_frames": 4}),
    ("snn_mask_recurrent_steps2", {"stft": {"n_fft": 128, "hop": 80, "long_factor": 0},
                                   "model": {"type": "snn", "output": "mask"},
                                   "snn": {**SNN, "hidden": [48], "recurrent": True, "steps_per_frame": 2}},
     "SNN with one layer, recurrent and 2 steps per frame, mask output", {}),
    ("snn_magnitude_nodecay", {"stft": DEFAULT_STFT, "model": {"type": "snn", "output": "magnitude"},
                               "snn": {**SNN, "learn_decay": False, "decay": 0.0}},
     "SNN with decay 0 (stored as sigmoid(logit(1e-4)) ~ 1e-4), magnitude output", {}),
    ("offline_mlp_mask_centred", {"stft": {**DEFAULT_STFT, "causal": False},
                                  "model": {"type": "mlp", "output": "mask", **MLP}},
     "centred frames (stft.causal = false, realtime = false): reflect padding, torch.istft synthesis; "
     "offline only, no streamed output", {"spectra": True}),
    ("offline_tcn_complex_centred", {"stft": {"n_fft": 128, "hop": 64, "long_factor": 0, "causal": False},
                                     "model": {"type": "tcn", "output": "complex_mask"}, "tcn": TCN_SMALL},
     "centred frames with a TCN and complex_mask; offline only", {}),
]


def case_config(settings):
    return merge(copy.deepcopy(BASE), copy.deepcopy(settings))


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # same reduction order on every run
    wav = vectors.synthetic_clip(SAMPLE_RATE, SECONDS, seed=0)
    index = {
        "about": "Golden vectors for the C++ port; written by plugin/tools/make_golden.py, format in "
                 "plugin/tools/README.md. Untrained models (fixed seeds), synthetic clip.",
        "torch_version": torch.__version__.split("+")[0],
        "numpy_version": np.__version__,
        "sample_rate": SAMPLE_RATE,
        "cases": [],
    }
    for seed, (name, settings, exercises, debug) in enumerate(CASES):
        cfg = case_config(settings)
        torch.manual_seed(1000 + seed)
        model = make_model(cfg).eval()
        model_file, vectors_file = f"{name}.model.bin", f"{name}.vectors.bin"
        header = modelfile.write(out / model_file, model, cfg)
        info = vectors.write_vectors(out / vectors_file, model, cfg, wav, case=name, model_file=model_file, **debug)
        index["cases"].append({
            "name": name,
            "exercises": exercises,
            "description": describe_model(cfg),
            "model_file": model_file,
            "vectors_file": vectors_file,
            "realtime": header["realtime"],
            "net": header["net"]["type"],
            "output": header["derived"]["output"],
            "n_fft": header["derived"]["n_fft"],
            "hop": header["derived"]["hop"],
            "long_factor": header["derived"]["long_factor"],
            "n_in": header["derived"]["n_in"],
            "n_out": header["derived"]["n_out"],
            "context_frames": header["derived"]["context_frames"],
            **info,
        })
        snn = info.get("snn")
        # what makes a case useful as a reference; settings changes that break it should fail here
        if info["stream"] and info["stream"]["max_abs_diff_vs_offline"] > 1e-5:
            raise SystemExit(f"{name}: streamed output differs from offline")
        if snn and not (all(0.05 < r < 0.95 for r in snn["spike_rate_per_layer"])
                        and snn["min_abs_v_minus_threshold"] > 1e-5 and snn["step_vs_forward_max_abs_diff"] < 1e-5):
            raise SystemExit(f"{name}: the SNN barely spikes, or a membrane potential comes within 1e-5 of the "
                             f"threshold (a port couldn't match its spikes exactly): {snn}")
        print(f"{name}: {(out / model_file).stat().st_size + (out / vectors_file).stat().st_size} bytes"
              + (f", stream vs offline {info['stream']['max_abs_diff_vs_offline']:.1e}" if info["stream"] else "")
              + (f", spike rates {snn['spike_rate_per_layer']}, closest to threshold "
                 f"{snn['min_abs_v_minus_threshold']:.1e}" if snn else ""))
    written = {c["model_file"] for c in index["cases"]} | {c["vectors_file"] for c in index["cases"]}
    for stale in [*out.glob("*.model.bin"), *out.glob("*.vectors.bin")]:
        if stale.name not in written:  # a case that was renamed or removed
            stale.unlink()
            print(f"removed {stale.name}")
    with open(out / "index.json", "w") as f:
        json.dump(index, f, indent=1)
        f.write("\n")
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"{len(CASES)} cases, {total} bytes in {out}")


if __name__ == "__main__":
    main()
