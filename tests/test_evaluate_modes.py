"""
Checks which of its two modes evaluate.py runs: no --model scores the saved
samples (<train.output_dir>/samples/*/, or --samples-dir), --model runs that
checkpoint over the test clips, and both at once is an error. Uses an
untrained model, synthetic clips and a parquet file made here; no dataset is
needed. The cache evaluate.py keeps is written to a temporary folder.

  .venv/bin/python tests/test_evaluate_modes.py
"""
import contextlib
import io
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate  # noqa: E402
from denoiser.config import load_config, merge  # noqa: E402
from denoiser.model import make_model, save_checkpoint  # noqa: E402

SR = 16000


def speech_like(seed, seconds=1.0):
    """A clean clip: harmonics of a gliding pitch, switched on and off like syllables."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voiced = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = (np.sin(2 * np.pi * 3 * t) > 0) * (0.5 + 0.5 * rng.random())
    clean = 0.1 * voiced * envelope
    return clean.astype("float32"), (clean + 0.02 * rng.standard_normal(len(t))).astype("float32")


def wav_bytes(wav):
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def run(argv, cache):
    """evaluate.main() with these arguments: (exit code, stdout, stderr)."""
    out, err, code = io.StringIO(), io.StringIO(), 0
    old_argv, old_cache = sys.argv, evaluate.CACHE_PATH
    sys.argv, evaluate.CACHE_PATH = ["evaluate.py", *argv], str(cache)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # PESQ may not score a synthetic clip: nan, and numpy warns
            evaluate.main()
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv, evaluate.CACHE_PATH = old_argv, old_cache
    return code, out.getvalue(), err.getvalue()


def main():
    torch.manual_seed(0)
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache = tmp / "eval_cache.json"
        clips = [speech_like(i) for i in range(2)]

        # saved samples, as save_samples writes them, in <output_dir>/samples and elsewhere
        for root in (tmp / "run" / "samples", tmp / "other"):
            for i, (clean, noisy) in enumerate(clips):
                d = root / f"clip{i}"
                d.mkdir(parents=True)
                for name, wav in (("clean", clean), ("noisy", noisy), ("denoised", clean)):
                    sf.write(str(d / f"{name}.wav"), wav, SR)

        # a test split with the same clips, and an untrained model
        data = tmp / "data"
        data.mkdir()
        column = pa.struct([("bytes", pa.binary())])
        table = pa.table({"noisy": pa.array([{"bytes": wav_bytes(n)} for _, n in clips], column),
                          "clean": pa.array([{"bytes": wav_bytes(c)} for c, _ in clips], column)})
        pq.write_table(table, str(data / "test-00000.parquet"))
        cfg = merge(load_config(), {"model": {"type": "mlp", "hidden": [16]}})
        save_checkpoint(make_model(cfg), cfg, tmp / "model.pt")
        run_dir, cpu = f"train.output_dir={tmp / 'run'}", "data.device=cpu"

        code, out, _ = run(["--set", run_dir], cache)
        checks["no arguments: scores <train.output_dir>/samples"] = (
            code == 0 and f"Evaluating samples in {tmp / 'run' / 'samples'}" in out and "denoised" in out)

        code, out, _ = run(["--samples-dir", str(tmp / "other"), "--set", run_dir], cache)
        checks["--samples-dir: scores that folder"] = (
            code == 0 and f"Evaluating samples in {tmp / 'other'}" in out and "denoised" in out)

        code, out, _ = run(["--model", str(tmp / "model.pt"), "--num-clips", "2",
                            "--set", f"data.test_dir={data}", "--set", cpu], cache)
        checks["--model: runs the model over the test clips"] = (
            code == 0 and f"Evaluating {tmp / 'model.pt'} on up to 2 test clips" in out
            and "clean->model" in out and "cost:" in out and cache.exists())

        code, out, err = run(["--model", str(tmp / "model.pt"), "--samples-dir", str(tmp / "other")], cache)
        checks["--model with --samples-dir: an error"] = (
            code == 2 and "--model and --samples-dir" in err and "Evaluating" not in out)

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
