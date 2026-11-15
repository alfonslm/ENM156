"""
Settings come from config.toml (shared defaults), then config.local.toml
(personal, git-ignored), then --set section.key=value on the command line.
Only settings config.toml defines can be set.
"""
import difflib
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG = ROOT / "config.local.toml"
MODEL_SECTIONS = ("stft", "model", "tcn", "snn")  # what a checkpoint stores: needed to rebuild the model


def merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def parse_set(item):
    """'train.epochs=5' -> {'train': {'epochs': 5}}. Values are read as TOML
    (numbers, true/false, lists), anything else as a plain string."""
    key, _, value = item.partition("=")
    section, _, name = key.strip().partition(".")
    if not name:
        raise SystemExit(f"--set needs section.key=value, got {item!r}")
    try:
        value = tomllib.loads(f"v = {value}")["v"]
    except tomllib.TOMLDecodeError:
        value = value.strip()
    return {section: {name: value}}


def check_known(cfg, overrides, source):
    """Refuses a section or key of `overrides` that config.toml (cfg) doesn't
    define: merged in, a misspelt one (train.epoch) would be a new setting
    nothing reads, and the run would go ahead with the default. Names the
    nearest existing setting. Values aren't checked: any type can be meant."""
    names = [f"{s}.{k}" for s, keys in cfg.items() for k in keys]
    for section, values in overrides.items():
        if section not in cfg:
            near = difflib.get_close_matches(section, list(cfg), n=1)
            hint = f"; did you mean [{near[0]}]?" if near else f"; config.toml has {', '.join(cfg)}"
            raise SystemExit(f"{source}: config.toml has no section [{section}]{hint}")
        if not isinstance(values, dict):
            raise SystemExit(f"{source}: {section} is a section of config.toml, not a setting; "
                             f"set one of its keys, e.g. {section}.{next(iter(cfg[section]))}")
        for key in values:
            if key not in cfg[section]:
                # the same key in another section first (train.n_fft -> stft.n_fft)
                near = ([n for n in names if n.partition(".")[2] == key]
                        or difflib.get_close_matches(f"{section}.{key}", names, n=1))
                hint = f"; did you mean {near[0]}?" if near else f"; see [{section}] in config.toml"
                raise SystemExit(f"{source}: config.toml has no setting {section}.{key}{hint}")


def load_config(overrides=()):
    with open(ROOT / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    if LOCAL_CONFIG.exists():
        with open(LOCAL_CONFIG, "rb") as f:
            local = tomllib.load(f)
        check_known(cfg, local, LOCAL_CONFIG.name)
        merge(cfg, local)
    for item in overrides:
        update = parse_set(item)
        check_known(cfg, update, f"--set {item}")
        merge(cfg, update)
    return cfg


def add_config_args(ap):
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="Override a setting for this run, e.g. --set train.epochs=5 (repeatable).")


def print_config(cfg, sections):
    for section in sections:
        print(f"  [{section}] " + json.dumps(cfg[section])[1:-1])


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class _Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def log_output(path, init_from=""):
    """Copy everything printed from here on (stdout and stderr) to path,
    replacing an older log there, like model.pt is replaced. A run that
    starts from a checkpoint (init_from) begins its log with the log.txt
    next to that checkpoint, so the log covers the model's whole training;
    if that log can't be read, the new log starts on its own. Each run's part
    of the log begins with a ===== line saying when it started."""
    previous, problem = "", ""
    if init_from:
        try:  # read before opening path: it can be the same file
            previous = (resolve_path(init_from).parent / "log.txt").read_text()
        except (OSError, UnicodeDecodeError) as e:
            problem = f"{type(e).__name__}: {e}"
    file = open(path, "w", buffering=1)  # line-buffered: a crash still leaves the log
    now = f"{datetime.now():%Y-%m-%d %H:%M}"
    if previous:
        file.write(previous.rstrip("\n") + f"\n\n===== continued from {init_from} at {now} =====\n\n")
    else:
        file.write(f"===== started at {now} =====\n\n")
    sys.stdout, sys.stderr = _Tee(sys.stdout, file), _Tee(sys.stderr, file)
    if problem:
        print(f"Couldn't add the log of {init_from}'s run ({problem}); this log starts here")


def get_device(cfg, key="device"):
    """The torch device data.<key> names. A GPU is capped at data.vram_fraction
    of its memory, so going over it is an out-of-memory error instead of the
    driver moving memory out to RAM, which stalls the GPU and the desktop."""
    name = cfg["data"][key] or cfg["data"]["device"]
    device = torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and cfg["data"]["vram_fraction"] < 1:
        torch.cuda.set_per_process_memory_fraction(cfg["data"]["vram_fraction"], device.index or 0)
    return device


def describe_device(device):
    return f"{device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
