"""
Checks that load_config refuses a setting config.toml doesn't define, from
--set or config.local.toml, naming the nearest one, and still takes every
real setting with values of any type. No dataset is needed.

  .venv/bin/python tests/test_config.py
"""
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser import config  # noqa: E402
from denoiser.config import load_config, merge  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def refused(*args):
    """The SystemExit message of load_config(*args), or None if it loads."""
    try:
        load_config(*args)
    except SystemExit as e:
        return str(e)
    return None


def main():
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        # the tests run with no config.local.toml of the user's, then with one made here
        config.LOCAL_CONFIG = Path(tmp) / "config.local.toml"

        cfg = load_config(["train.epochs=5", "pesq.init_from=", "stft.causal=false", "train.lr=1e-4",
                           "model.hidden=[64, 32]", "train.output_dir=runs/x y", "model.filterbank=a.pt"])
        checks["--set: values of every type"] = (
            cfg["train"]["epochs"] == 5 and cfg["pesq"]["init_from"] == "" and cfg["stft"]["causal"] is False
            and cfg["train"]["lr"] == 1e-4 and cfg["model"]["hidden"] == [64, 32]
            and cfg["train"]["output_dir"] == "runs/x y" and cfg["model"]["filterbank"] == "a.pt")

        msg = refused(["train.epoch=5"])
        checks["--set misspelt key: refused, nearest named"] = (
            msg is not None and "train.epoch=5" in msg and "did you mean train.epochs?" in msg)
        msg = refused(["trian.epochs=5"])
        checks["--set misspelt section: refused, nearest named"] = (
            msg is not None and "[trian]" in msg and "did you mean [train]?" in msg)
        msg = refused(["train.n_fft=256"])
        checks["--set key of another section: named"] = msg is not None and "did you mean stft.n_fft?" in msg
        checks["--set nothing like it: refused"] = refused(["train.zzzzzz=1"]) is not None

        # the example local config, as a user copies it, with its commented lines switched on
        example = (ROOT / "config.local.example.toml").read_text()
        config.LOCAL_CONFIG.write_text(example)
        checks["config.local.example.toml as it is"] = (
            load_config()["data"]["voicebank_dir"] == "/path/to/VoiceBank-DEMAND-16k/data")
        config.LOCAL_CONFIG.write_text(re.sub(r"(?m)^# (?=\w+ = )", "", example))
        checks["config.local.example.toml with every line on"] = load_config()["train"]["batch_size"] == 256

        config.LOCAL_CONFIG.write_text("[train]\nbatch_sise = 256\n")
        msg = refused()
        checks["config.local.toml misspelt key: refused, nearest named"] = (
            msg is not None and "config.local.toml" in msg and "did you mean train.batch_size?" in msg)
        config.LOCAL_CONFIG.write_text("epochs = 5\n")
        checks["config.local.toml key outside a section: refused"] = refused() is not None
        config.LOCAL_CONFIG.write_text("[train]\nepochs = 5\n")
        checks["config.local.toml then --set"] = load_config(["train.epochs=7"])["train"]["epochs"] == 7
        config.LOCAL_CONFIG.write_text("train = 5\n")
        checks["config.local.toml section as a value: refused"] = refused() is not None

        # merge() itself stays unchecked: tests and old checkpoints (load_checkpoint) merge in
        # settings directly, and old checkpoints can hold keys config.toml no longer has
        config.LOCAL_CONFIG = Path(tmp) / "missing.toml"
        cfg = merge(load_config(), {"model": {"old_setting": 1}, "stft": {"hop": 64}})
        checks["merge() unchecked"] = cfg["model"]["old_setting"] == 1 and cfg["stft"]["hop"] == 64

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
