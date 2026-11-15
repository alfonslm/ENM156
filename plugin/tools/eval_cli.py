"""
PESQ / STOI of the C++ command-line tool on the test clips, scored with
evaluate.py's own functions, so the table can be put next to
`evaluate.py --model` for the same checkpoint (they should agree to the
printed precision).

  python plugin/tools/export.py runs/default/model.pt out/            # -> out/model.bin
  python plugin/tools/eval_cli.py plugin/build/cli/denoise out/model.bin --num-clips 100 [--stream]
  python evaluate.py --model runs/default/model.pt --num-clips 100 --set data.device=cpu

Every noisy clip (and, like evaluate.py, every clean clip: the "clean->model"
row) is written as a 32-bit float WAV, denoised by the tool and read back.
Clips, splits and --every work as in evaluate.py (same config.toml /
config.local.toml, same --set). --stream runs the tool's real-time path.

Only reads denoiser/ and evaluate.py; changes nothing there.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from denoiser.config import add_config_args, load_config  # noqa: E402
from denoiser.data import load_pairs, split_files  # noqa: E402
from evaluate import evaluate_model, print_results  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tool", type=Path, help="the built denoise executable")
    ap.add_argument("model", type=Path, help="model.bin written by export.py")
    ap.add_argument("--num-clips", type=int, default=824)
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--stream", action="store_true", help="run the tool with --stream (the plugin's path)")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.set)
    sr = cfg["data"]["sample_rate"]
    work = Path(tempfile.mkdtemp(prefix="eval_cli_"))
    wav_in, wav_out = work / "in.wav", work / "out.wav"

    def denoise(wav):
        sf.write(str(wav_in), np.asarray(wav, np.float32), sr, subtype="FLOAT")
        cmd = [str(args.tool), "-q", "--float", *(["--stream"] if args.stream else []),
               str(args.model), str(wav_in), str(wav_out)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"{' '.join(cmd)} failed ({p.returncode}):\n{p.stderr}")
        out, out_sr = sf.read(str(wav_out), dtype="float32")
        assert out_sr == sr and len(out) == len(wav)
        return out

    files = split_files(cfg, args.split)
    print(f"Evaluating {args.tool} {'--stream ' if args.stream else ''}{args.model} on up to {args.num_clips} "
          f"{args.split} clips" + (f", each {args.every}th" if args.every > 1 else "") + "...")
    results, _ = evaluate_model(denoise, load_pairs(files, args.num_clips, sr, every=args.every))
    print_results(results)


if __name__ == "__main__":
    main()
