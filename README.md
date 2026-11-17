# Ball possession, frame by frame

The CSV has one row per frame of the tactical video. The `label` column contains `H`, `A` or `D`.
See [TECHNICAL.md](TECHNICAL.md) for the method.

## Run

```powershell
# copy the organisers' starter tree into official/ (revision f8f2629)
# put the data bundle in official/data/ as the organisers describe (official/DATA.md)
uv run --no-project python pipeline/last-touch/build.py
```

Output: `pipeline/last-touch/.local/last_touch.csv`.

Requirements:

- Windows, Git, [uv](https://docs.astral.sh/uv/), ffmpeg with CUDA decoding on `PATH`, a CUDA GPU
  with 8 GB.
- About 10 GB of disk and a network connection. The build downloads the public detector weights,
  the action spotter weights and seven Bundesliga matches (about 2.5 GB).
- About four hours on an RTX 4060 laptop. The three detector stages take nearly all of it.

Useful options:

```powershell
uv run --no-project python pipeline/last-touch/build.py --list          # the stages and their outputs
uv run --no-project python pipeline/last-touch/build.py --from stubs    # rerun the CPU stages after the sensors
```

Check the CSV format with the organisers' tool.

```powershell
uv run --no-project --with numpy --with pandas python official/evaluation/validate_submission.py pipeline/last-touch/.local/last_touch.csv
```

## Layout

Each folder under `pipeline/` is one stage or a library used by a stage. `last-touch/build.py`
lists the stages in order; `from-raw/build.py` holds the stage functions.

| Folder | What it does |
|---|---|
| `possession-engine`, `football-ball-detector` | Loading the provided tracking; fetching the detector weights |
| `hc-video`, `hc-ball` | Ball detection on both videos, the physics tracker, two-view height |
| `touches-video` | Player detection, jersey-colour teams, touch proposals |
| `dfl-owner`, `dfl-possession-transfer`, `free-dfl` | Bundesliga data download and parsing; dead/live and team models trained on it |
| `free-owner`, `free-status`, `free-team` | Unsupervised dead/live model; team model on machine-made targets; physical cues |
| `free-grammar`, `free-brief`, `free-rules` | Ownership grammar, brief contacts, inputs of the rule stages |
| `fm-rebuild` | The base: decoder, grammar and eleven rule stages replayed from fixed recipe files |
| `free-handset`, `free-pretrained`, `free-stack`, `free-stable`, `free-merge`, `free-stoppages`, `free-timing` | The sequence decoder and its supporting code |
| `free-carrier`, `agreed-touches` | Player-level carrier model; the touch stage |
| `last-touch` | The build, the settings (`settings.json`) and the final steps |
| `calibration/` (top level) | Synchronisation between the two videos |

See `pipeline/last-touch/settings.json` and [TECHNICAL.md](TECHNICAL.md) for the settings.
