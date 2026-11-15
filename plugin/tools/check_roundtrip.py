"""
Round trip of the exporter for every network type and output kind, causal
and centred: checkpoint (save_checkpoint) -> export.py -> model file ->
read back -> identical weights, settings and header, and the model rebuilt
from the file alone gives bit-identical output to the original. Also checks
that export.py refuses filterbank checkpoints, and centred ones with --realtime.

  python plugin/tools/check_roundtrip.py
"""
import copy
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from denoiser.model import load_checkpoint, make_model, save_checkpoint  # noqa: E402

import modelfile  # noqa: E402
from export import round_trip  # noqa: E402
from make_golden import BASE  # noqa: E402

HERE = Path(__file__).resolve().parent
SMALL = {"model": {"hidden": [24, 16]}, "tcn": {"channels": 8, "dilations": [1, 2, 4]},
         "snn": {"hidden": [16, 12], "threshold": 0.1}}
VARIANTS = {  # extra settings per output kind, so every input layout is exported
    "magnitude": [{}],
    "mask": [{"stft": {"long_factor": 0}}],
    "complex_mask": [{}, {"model": {"mag_input": False, "mask_bound": 0}, "stft": {"long_factor": 0}}],
}


def cli(*args):
    return subprocess.run([sys.executable, str(HERE / "export.py"), *map(str, args)], capture_output=True, text=True)


def main():
    torch.set_num_threads(1)
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        n = 0
        for kind, output, causal in itertools.product(("mlp", "tcn", "snn"), VARIANTS, (True, False)):
            for variant in VARIANTS[output]:
                extra = {"snn": {"recurrent": True, "steps_per_frame": 2}} if kind == "snn" and not causal else {}
                cfg = copy.deepcopy(BASE)
                for part in (SMALL, {"model": {"type": kind, "output": output}, "stft": {"causal": causal}},
                             variant, extra):
                    for section, values in part.items():
                        cfg[section].update(copy.deepcopy(values))
                torch.manual_seed(n)
                ckpt = tmp / f"{n}.pt"
                save_checkpoint(make_model(cfg), cfg, ckpt)
                # as export.py does it: through load_checkpoint
                model, loaded_cfg = load_checkpoint(ckpt, copy.deepcopy(BASE), torch.device("cpu"))
                path = tmp / f"{n}.bin"
                header = modelfile.write(path, model.eval(), loaded_cfg)
                try:
                    result = round_trip(path, model, loaded_cfg, header)
                    ok = header["realtime"] == causal
                except SystemExit as e:
                    result, ok = str(e), False
                failed |= not ok
                print(f"{'ok  ' if ok else 'FAIL'} {kind} {output} {'causal' if causal else 'centred'} "
                      f"{variant or ''}{extra or ''}: {result}")
                n += 1

        # the command line, end to end, on the last checkpoint (centred)
        out = cli(ckpt, tmp / "cli", "--vectors")
        ok = out.returncode == 0 and (tmp / "cli" / "model.bin").exists() and (tmp / "cli" / "vectors.bin").exists()
        print(f"{'ok  ' if ok else 'FAIL'} export.py CLI with --vectors on a centred model"
              + ("" if ok else f": {out.stdout}{out.stderr}"))
        failed |= not ok
        out = cli(ckpt, tmp / "cli2", "--realtime")
        ok = out.returncode != 0 and "can't run live" in out.stderr
        print(f"{'ok  ' if ok else 'FAIL'} --realtime refuses a centred model: {out.stderr.strip()}")
        failed |= not ok

        # a filterbank checkpoint is refused before its filterbank file is even looked for
        cfg = copy.deepcopy(BASE)
        cfg["model"].update({"output": "mask", "filterbank": "runs/filterbank/missing.pt"})
        fb_ckpt = tmp / "fb.pt"
        torch.save({"config": {s: cfg[s] for s in ("stft", "model", "tcn", "snn")},
                    "state_dict": make_model({**copy.deepcopy(BASE), "model": {**cfg["model"], "filterbank": ""}})
                   .state_dict()}, fb_ckpt)
        out = cli(fb_ckpt, tmp / "fb")
        ok = out.returncode != 0 and "filterbank" in out.stderr and not (tmp / "fb" / "model.bin").exists()
        print(f"{'ok  ' if ok else 'FAIL'} filterbank checkpoint refused: {out.stderr.strip()}")
        failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
