"""
Checkpoint (model.pt, as denoiser.model.save_checkpoint writes it) -> a model
file the C++ plugin reads (format: plugin/tools/README.md).

  python plugin/tools/export.py runs/default/model.pt out/
      writes out/model.bin and checks that it reads back to the same weights
      and settings, and that the model rebuilt from it alone gives the same output
  python plugin/tools/export.py runs/default/model.pt out/ --vectors [--wav clip.wav]
      also writes out/vectors.bin: test vectors of this model on a short
      synthetic clip (or the first second of clip.wav), as in the golden set
  --realtime   refuse centred (stft.causal = false) models instead of marking them offline-only
  --set data.sample_rate=16000   the sample rate isn't stored in checkpoints:
               it comes from config.toml / config.local.toml unless set here

Only reads denoiser/; changes nothing there.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from denoiser.config import add_config_args, load_config  # noqa: E402
from denoiser.model import describe_model, load_checkpoint  # noqa: E402

import modelfile  # noqa: E402
import vectors  # noqa: E402


def check_checkpoint(path):
    """Refuse a filterbank checkpoint before load_checkpoint tries to load its filterbank."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        modelfile.check_supported({"model": {"filterbank": checkpoint["config"]["model"].get("filterbank", "")}})


def round_trip(path, model, cfg, header):
    """Raises if the file doesn't give back exactly the model's weights,
    settings and output. -> description of what was checked."""
    read_header, tensors = modelfile.read(path)
    _, expected = modelfile.to_export(model, cfg)
    if json.loads(json.dumps(header)) != {k: v for k, v in read_header.items() if k != "tensors"}:
        raise SystemExit("round trip: the header read back differs")
    if [n for n, _ in expected] != list(tensors):
        raise SystemExit("round trip: tensor names or order differ")
    for name, array in expected:
        if array.shape != tensors[name].shape or not np.array_equal(array, tensors[name]):
            raise SystemExit(f"round trip: tensor {name} differs")
    rebuilt, rebuilt_cfg = modelfile.rebuild(path)
    for (name, a), (name2, b) in zip(model.state_dict().items(), rebuilt.state_dict().items()):
        if name.endswith("raw_decay"):  # stored as the decay itself: compare what is used
            a, b = torch.sigmoid(a), torch.sigmoid(b)
        if name != name2 or not torch.equal(a.float(), b.float()):
            raise SystemExit(f"round trip: rebuilt parameter {name} differs")
    wav = torch.from_numpy(vectors.synthetic_clip(cfg["data"]["sample_rate"], 0.5, seed=1))
    from denoiser.pipeline import enhance
    with torch.no_grad():
        a = enhance(model.eval(), wav, cfg)
        b = enhance(rebuilt, wav, rebuilt_cfg)
    if not torch.equal(a, b):
        raise SystemExit(f"round trip: rebuilt model's output differs by {(a - b).abs().max():.1e}")
    return f"{len(tensors)} tensors, header and the rebuilt model's output identical"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--vectors", action="store_true", help="also write vectors.bin")
    ap.add_argument("--wav", type=Path, help="with --vectors: a 16 kHz mono clip instead of the synthetic one")
    ap.add_argument("--realtime", action="store_true", help="refuse models that can't run live")
    add_config_args(ap)
    args = ap.parse_args()
    torch.set_num_threads(1)

    check_checkpoint(args.checkpoint)
    model, cfg = load_checkpoint(args.checkpoint, load_config(args.set), torch.device("cpu"))
    model.eval()
    if args.realtime and not cfg["stft"]["causal"]:
        raise SystemExit("stft.causal = false: this model looks ahead and can't run live (fine offline/ARA)")
    print(f"{args.checkpoint}: {describe_model(cfg)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "model.bin"
    header = modelfile.write(path, model, cfg)
    print(f"wrote {path} ({path.stat().st_size} bytes)" +
          ("" if header["realtime"] else "; centred frames: marked offline-only (realtime = false)"))
    print(f"round trip: {round_trip(path, model, cfg, header)}")
    if args.vectors:
        if args.wav:
            import soundfile as sf
            wav, sr = sf.read(str(args.wav), dtype="float32", always_2d=True)
            if sr != cfg["data"]["sample_rate"]:
                raise SystemExit(f"{args.wav} is {sr} Hz, the model wants {cfg['data']['sample_rate']} Hz")
            wav = wav[:sr, 0]
        else:
            wav = vectors.synthetic_clip(cfg["data"]["sample_rate"], 0.5, seed=0)
        vpath = args.out_dir / "vectors.bin"
        info = vectors.write_vectors(vpath, model, cfg, wav, case="export", model_file="model.bin")
        print(f"wrote {vpath} ({vpath.stat().st_size} bytes): {', '.join(info['tensors'])}")


if __name__ == "__main__":
    main()
