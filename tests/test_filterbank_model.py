"""
Checks a denoiser on a learned filterbank (model.filterbank) where it isn't
covered elsewhere: that train.py refuses one up front, whether set in the
settings or in the checkpoint it starts from, instead of failing on a shape
mismatch. Uses an untrained filterbank and model; no dataset is needed.

  .venv/bin/python tests/test_filterbank_model.py
"""
import contextlib
import io
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train  # noqa: E402
from denoiser.config import load_config  # noqa: E402
from denoiser.filterbank import Filterbank, save_filterbank  # noqa: E402
from denoiser.model import make_model, save_checkpoint  # noqa: E402
from denoiser.pipeline import segment_frames  # noqa: E402

FB = {"filters": 64, "kernel": 32, "stride": 16, "relu": True}


def run_train(argv):
    """train.main() with these arguments: its SystemExit message, or None."""
    old = sys.argv, sys.stdout, sys.stderr  # log_output replaces stdout and stderr
    sys.argv = ["train.py", *argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train.main()
    except SystemExit as e:
        return str(e)
    finally:
        sys.argv, sys.stdout, sys.stderr = old
    return None


def main():
    torch.manual_seed(0)
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_filterbank(Filterbank(**FB), FB, tmp / "filterbank.pt")
        fb_settings = ["--set", f"model.filterbank={tmp / 'filterbank.pt'}", "--set", "model.output=mask"]
        cfg = load_config([s for s in fb_settings if s != "--set"])
        model = make_model(cfg)

        # what train.py would run into: its STFT frames don't fit the filterbank model
        wav = 0.1 * torch.randn(2, 4000)
        x, _ = segment_frames(wav, wav, torch.zeros(2, dtype=torch.long), cfg)
        try:
            model(x)
            checks["the mismatch train.py would hit"] = False
        except RuntimeError:
            checks["the mismatch train.py would hit"] = True

        run_dir = tmp / "run"
        msg = run_train([*fb_settings, "--set", f"train.output_dir={run_dir}"])
        checks["train.py, filterbank in the settings: refused up front"] = (
            msg is not None and "train_pesq.py" in msg and not run_dir.exists())

        save_checkpoint(model, cfg, tmp / "model.pt")
        msg = run_train(["--set", f"train.init_from={tmp / 'model.pt'}", "--set", "data.device=cpu",
                         "--set", f"train.output_dir={run_dir}"])
        checks["train.py, filterbank in train.init_from: refused"] = (
            msg is not None and "train_pesq.py" in msg and str(tmp / "model.pt") in msg)

        # without a filterbank, train.py goes on past the check (to the missing dataset here)
        msg = run_train(["--set", "data.device=cpu", "--set", f"data.voicebank_dir={tmp / 'no_data'}",
                         "--set", f"train.output_dir={run_dir}"])
        checks["train.py without a filterbank: not refused for it"] = msg is not None and "parquet" in msg

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
