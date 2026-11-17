"""Shared build runner and input stages for the final pipeline.

Use pipeline/last-touch/build.py to produce the submission CSV.

Needs the match data in official/data/{tactical,broadcast}, uv, ffmpeg on PATH and a CUDA GPU.
Stage times and output paths go to .local/manifest.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "pipeline"
LOCAL = HERE / ".local"
LOGS = LOCAL / "logs"
MANIFEST = LOCAL / "manifest.json"
UV_NP = ["uv", "run", "--no-project", "--with", "numpy"]


def venv_py(lane):
    return str(EXP / lane / ".local" / "venv" / "Scripts" / "python.exe")


def run(cmd, cwd, log, idle=None, retries=0, env=None):
    """Run one command, appending its output to the stage log.

    With `idle` (seconds) the whole process tree is killed when the log has not grown for that
    long, and the command is retried up to `retries` times. The GPU detectors print progress every
    2,000 frames; when the machine runs out of commit memory their frame-reader thread dies with a
    MemoryError and the main loop then waits forever, so a silent detector is a dead one."""
    cmd = [str(c) for c in cmd]
    for attempt in range(retries + 1):
        line = f"[{time.strftime('%H:%M:%S')}] (cd {cwd}) {' '.join(cmd)}"
        print(line, flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            p = subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
                                 env=None if env is None else {**os.environ, **env})
            size, seen = os.path.getsize(log), time.time()
            while (code := p.poll()) is None:
                time.sleep(2)
                if idle is None:
                    continue
                now = os.path.getsize(log)
                if now != size:
                    size, seen = now, time.time()
                elif time.time() - seen > idle:
                    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
                    p.wait()
                    code = "no output for %ds, killed" % idle
                    break
        if code == 0:
            return
        print(f"  attempt {attempt + 1} failed ({code})", flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"  attempt {attempt + 1} failed ({code})\n")
        time.sleep(10)
    raise SystemExit(f"failed ({code}): {' '.join(cmd)} - see {log}")


# ---------------------------------------------------------------------------------------------
# stages: (name, device, function(log), key outputs relative to ROOT)
# ---------------------------------------------------------------------------------------------

def s_tracking(log):
    run(["uv", "run", "python", "-c", "import data; data.tracking()"], EXP / "possession-engine", log)


def s_weights(log):
    run(["uv", "run", "--frozen", "--no-sync", "--project", "pipeline/possession-engine",
         "python", "-B", "pipeline/football-ball-detector/download.py"], ROOT, log)


def make_venv(lane, log, packages):
    """packages: list of pip-argument lists, installed in order."""
    py = venv_py(lane)
    if not os.path.exists(py):
        run(["uv", "venv", f".local/venv", "--python", "3.12.12"], EXP / lane, log)
    for args in packages:
        run(["uv", "pip", "install", "--python", py, *args], EXP / lane, log)


TORCH = ["torch==2.11.0+cu128", "torchvision==0.26.0+cu128", "--index-url", "https://download.pytorch.org/whl/cu128"]
VISION = ["ultralytics==8.4.191", "opencv-python-headless==5.0.0.93", "opencv-python==5.0.0.93", "numpy==2.5.2",
          "pandas==3.1.0", "lightgbm==4.7.0", "scipy==1.18.1", "pyarrow==26.0.0", "numba==0.68.1", "scikit-learn==1.9.1"]


def npz_len(path, key="frames"):
    """Length of one array in an .npz, read from its .npy header (no numpy needed here)."""
    import ast
    import zipfile
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        major = f.read(8)[6]                      # magic (6 bytes), then the format version
        hlen = int.from_bytes(f.read(2 if major == 1 else 4), "little")
        header = ast.literal_eval(f.read(hlen).decode("latin1"))
    return header["shape"][0]


def chunked(lane, script, source, start, end, chunk_dir, tag=None, args=None, chunk=10000, idle=120, home=None):
    """The chunks a lane's run_detect.py (or players.py all) would make, one detector call each,
    with a watchdog, the same file names and arguments, so the lane's own merge step afterwards
    finds them all. A detector whose FFmpeg reader stops early saves a short chunk without
    failing; such a chunk is deleted and redone."""
    def go(log):
        py, cwd = venv_py(lane), EXP / lane          # lane: whose venv; home: whose .local holds the chunks
        for a in range(start, end, chunk):
            b = min(end, a + chunk)
            part = EXP / (home or lane) / ".local" / chunk_dir / f"{tag or source}_{a}_{b}.npz"
            part.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(5):
                if part.exists() and npz_len(part) == b - a:
                    break
                if part.exists():
                    with open(log, "a") as fh:
                        fh.write(f"  short chunk {part.name}: {npz_len(part)} of {b - a} frames, redoing\n")
                    part.unlink()
                cmd = args(a, b, part) if args else [source, a, b, part]
                run([py, str(cwd / script), *cmd], cwd, log, idle=idle, retries=20)
            else:
                raise SystemExit(f"chunk {part} keeps coming out short")
    return go


HCV_DETECT = (("tactical", 11000, 90800, ".local/det_tac_h1.npz"),
              ("tactical", 110600, 190400, ".local/det_tac_h2.npz"),
              ("broadcast", 6980, 84800, ".local/det_bro_h1.npz"),
              ("broadcast", 86540, 163500, ".local/det_bro_h2.npz"))


def s_hcvideo_detect(log):
    make_venv("hc-video", log, [TORCH, VISION])
    py, cwd = venv_py("hc-video"), EXP / "hc-video"
    for source, a, b, out in HCV_DETECT:
        chunked("hc-video", "detect.py", source, a, b, "chunks")(log)
        run([py, "run_detect.py", source, a, b, out], cwd, log)


def s_dfl_ingest(log):
    run(["uv", "run", "--no-project", "--with-requirements", "pipeline/dfl-owner/requirements.txt",
         "python", "-u", "pipeline/dfl-owner/ingest.py"], ROOT, log)


def s_free_dfl(log):
    req = ["uv", "run", "--no-project", "--with-requirements", "pipeline/free-dfl/requirements.txt", "python", "-u"]
    run([*req, "pipeline/free-dfl/checks.py"], ROOT, log)
    for cmd in ("prepare", "validate", "transfer"):
        run([*req, "pipeline/free-dfl/run.py", cmd], ROOT, log)
    local = EXP / "free-dfl" / ".local"
    shutil.copyfile(local / "best.csv", local / "dfl_only.csv")
    shutil.copyfile(local / "evidence.npz", local / "dfl_only_evidence.npz")


def uvw(*packages):
    """`uv run --no-project --with ...` as the lanes' READMEs write it."""
    cmd = ["uv", "run", "--no-project"]
    for p in packages:
        cmd += ["--with", p]
    return cmd + ["python"]


SCI = ("numpy", "pandas", "scipy", "numba")
SCI_GBM = SCI + ("lightgbm",)


def s_hcvideo_free(log):
    py, cwd = venv_py("hc-video"), EXP / "hc-video"
    run([py, "clean.py"], cwd, log)
    run([py, "clean.py", "broadcast"], cwd, log)
    run([py, "lift.py"], cwd, log)
    run([py, str(HERE / "hcvideo_free.py")], cwd, log)     # feat_*.npz and evidence.npz without team_q


def s_touches_players(log):
    py = venv_py("hc-video")
    ranges = {"tactical": [(11185, 90652), (110808, 190198)], "broadcast": [(6980, 84800), (86540, 163500)]}
    for source, sub in (("tactical", "players"), ("broadcast", "players_broadcast")):
        for lo, hi in ranges[source]:
            chunked("hc-video", str(EXP / "touches-video" / "players.py"), source, lo, hi, sub, tag=source[:3],
                    args=lambda a, b, part, s=source: [a, b, part, s], home="touches-video")(log)
        # players.py's own `all` then finds every chunk and has nothing left to do
        run([py, str(EXP / "touches-video" / "players.py"), "all", *([] if source == "tactical" else ["broadcast"])], ROOT, log)


def s_touches_cpu(log):
    tv = uvw("numpy", "pandas", "scipy", "numba", "lightgbm", "pyarrow", "opencv-python-headless")
    for args in (["pixel.py"], ["pixel.py", "broadcast"], ["features.py"], ["events.py"]):
        run([*tv, f"pipeline/touches-video/{args[0]}", *args[1:]], ROOT, log)


HCB_DETECT = (("tactical", 11185, 90652, "zoom_tac_h1"), ("tactical", 110808, 190198, "zoom_tac_h2"),
              ("broadcast", 6980, 84800, "zoom_bro_h1"), ("broadcast", 86540, 163500, "zoom_bro_h2"))


def s_hcball_detect(log):
    make_venv("hc-ball", log, [TORCH, VISION])
    py, cwd = venv_py("hc-ball"), EXP / "hc-ball"
    for source, a, b, tag in HCB_DETECT:
        chunked("hc-ball", "detect.py", source, a, b, "chunks", tag=tag)(log)
        run([py, "run_detect.py", source, a, b, f".local/{tag}.npz"], cwd, log)


def s_hcball_cpu(log):
    py, cwd = venv_py("hc-ball"), EXP / "hc-ball"
    for args in (["track.py", "tactical"], ["track.py", "broadcast"], ["contact.py"], ["ball3d.py"]):
        run([py, *args], cwd, log)


def s_free_owner(log):
    fo = [*uvw("numpy", "pandas", "scipy", "numba", "lightgbm", "scikit-learn"), "-W", "ignore"]
    cwd = EXP / "free-owner"
    for args in (["features.py"], ["events.py"], ["status.py"], ["team.py", "seed"], ["contact_broadcast.py"]):
        run([*fo, *args], cwd, log)
    run([*fo, str(HERE / "owner_evidence.py")], cwd, log)   # evidence.npz: the three label-free keys


LANE_TDEED = False      # --lane-tdeed: run free-pretrained/tdeed_run.py exactly as committed (non-deterministic)
FP_PACKAGES = ["timm==1.0.31", "numpy==2.5.4", "pandas==3.1.1", "scipy==1.18.1", "numba==0.68.1", "tqdm==4.70.1",
               "opencv-python-headless==5.0.0.93", "scikit-learn==1.9.1"]   # the versions the submitted file was built with
FP_WEIGHTS = ("teamball_baseline.pt", "ball_challenge1.pt")


def s_free_pretrained(log):
    lane, cwd = "free-pretrained", EXP / "free-pretrained"
    make_venv(lane, log, [TORCH, FP_PACKAGES])
    local = cwd / ".local"
    (local / "weights").mkdir(parents=True, exist_ok=True)
    for name in FP_WEIGHTS:
        dst = local / "weights" / name
        if not dst.exists():
            gid = {"teamball_baseline.pt": "1Dp2tqDIrNwLmfuUJPPhpz1db7DZjxIzd",
                   "ball_challenge1.pt": "1vvASNW8fuk1c-E-55QSVar1q_i5EFTOT"}[name]
            run(["uv", "pip", "install", "--python", venv_py(lane), "gdown"], cwd, log)
            run([venv_py(lane), "-c", f"import gdown; gdown.download(id='{gid}', output=r'{dst}')"], cwd, log)
    ext = local / "ext" / "sn-teamspotting"
    if not ext.exists():
        run(["git", "clone", "--quiet", "https://github.com/SoccerNet/sn-teamspotting", str(ext)], ROOT, log)
        run(["git", "-C", str(ext), "checkout", "--quiet", "091fed2"], ROOT, log)
    py = venv_py(lane)
    for video, a, b in (("broadcast", 6900, 84900), ("broadcast", 86400, 163600),
                        ("tactical", 10984, 90852), ("tactical", 110608, 190398)):
        out = local / f"tdeed_{video}_{a}_{b}.npz"
        expect = (b - a + 1) // 2
        for attempt in range(5):
            if out.exists() and npz_len(out, "frame") == expect:
                break
            if out.exists():
                out.unlink()          # its FFmpeg reader stopped early: redo the half
            # tdeed_det.py = tdeed_run.py with cuDNN's autotuner off and deterministic kernels on:
            # with the autotuner the spotter does not repeat bit for bit between runs (README)
            script = ["tdeed_run.py"] if LANE_TDEED else [str(HERE / "tdeed_det.py")]
            run([py, "-I", *script, video, a, b], cwd, log, idle=300, retries=10)
        else:
            raise SystemExit(f"{out} keeps coming out short")
    run([py, "-I", "tdeed_evidence.py"], cwd, log)


def s_free_team(log):
    ft = [*uvw("numpy", "pandas", "scipy", "numba", "lightgbm", "scikit-learn"), "-W", "ignore"]
    cwd = EXP / "free-team"
    run([*ft, "-c", "import run; run.snapshot()"], cwd, log)
    run([*ft, "local_features.py"], cwd, log)
    run([*ft, "cotrain.py", "local", "q_team1.npy", "q_local.npy"], cwd, log)


def s_rules(log):
    run([*uvw(*SCI_GBM), str(HERE / "rules_inputs.py")], ROOT, log)


def s_brief_grammar(log):
    cwd = EXP / "free-brief"
    run([*uvw("numpy", "pandas", "scipy"), "-W", "ignore", "contact_bro.py"], cwd, log)
    run([*uvw(*SCI), "-W", "ignore", "detect.py"], cwd, log)
    gl = EXP / "free-grammar" / ".local"
    run([*uvw(*SCI), "pipeline/free-grammar/prepare.py"], ROOT, log)
    run([*uvw("numpy", "scipy", "numba"), "pipeline/free-grammar/learn_prior.py"], ROOT, log)
    stack = EXP / "free-stack" / ".local"
    stack.mkdir(parents=True, exist_ok=True)
    for src in (gl / "touches.npz", gl / "em_prior.json", cwd / ".local" / "detect.npz"):
        shutil.copyfile(src, stack / src.name)     # what free-stack/run.py --prepare copies


def s_stoppages_audio(log):
    run([*uvw("numpy", "scipy"), "pipeline/free-stoppages/audio.py"], ROOT, log)


L = "pipeline/"
STAGES = [
    ("tracking", "CPU", s_tracking, [L + "possession-engine/.local/tracking.npz"]),
    ("weights", "CPU", s_weights, [L + "football-ball-detector/.local/football_best.pt"]),
    ("hc-video-detect", "GPU", s_hcvideo_detect,
     [f"{L}hc-video/.local/det_{v}_h{h}.npz" for v in ("tac", "bro") for h in (1, 2)]),
    ("hc-video-free", "CPU", s_hcvideo_free,
     [L + f"hc-video/.local/{f}" for f in ("ball_video.npz", "ball_broadcast.npz", "ball_3d.npz", "feat_provided.npz",
                                          "feat_video.npz", "feat_broadcast.npz", "evidence.npz",
                                          "homography_tactical.npy", "homography_broadcast.npy")]),
    ("touches-players", "GPU", s_touches_players, []),
    ("touches-cpu", "CPU", s_touches_cpu,
     [L + f"touches-video/.local/{f}" for f in ("boxteam.npz", "boxteam_broadcast.npz", "pixel.npz",
                                               "pixel_broadcast.npz", "frames.npz", "events.parquet")]),
    ("hc-ball-detect", "GPU", s_hcball_detect, [L + f"hc-ball/.local/{t[3]}.npz" for t in HCB_DETECT]),
    ("hc-ball-cpu", "CPU", s_hcball_cpu,
     [L + f"hc-ball/.local/{f}" for f in ("track_tactical.npz", "track_broadcast.npz", "contact_tactical.npz", "ball3d.npz")]),
    ("free-owner", "CPU", s_free_owner,
     [L + f"free-owner/.local/{f}" for f in ("features.npz", "events.npz", "p_dead.npy", "llr_dead.npy", "q_team1.npy",
                                            "contact_broadcast.npz", "evidence.npz")]),
    ("dfl-ingest", "CPU", s_dfl_ingest, [f"{L}dfl-owner/.local/data/{k}.npz" for k in
     ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")]),
    ("free-dfl", "CPU", s_free_dfl, [L + "free-dfl/.local/dfl_only_evidence.npz", L + "free-dfl/.local/dfl_only.csv",
                                     L + "free-dfl/.local/bar.npz", L + "free-dfl/.local/decoder.json"]),
    ("free-pretrained", "GPU", s_free_pretrained,
     [L + f"free-pretrained/.local/{f}" for f in ("tdeed_broadcast_6900_84900.npz", "tdeed_broadcast_86400_163600.npz",
                                                 "tdeed_tactical_10984_90852.npz", "tdeed_tactical_110608_190398.npz",
                                                 "tdeed_events.npz", "tdeed_channels.npz")]),
    ("free-team", "CPU", s_free_team,
     [L + "free-team/.local/local.npz", L + "free-team/.local/q_local.npy"]),
    ("rules", "CPU", s_rules, [L + "free-rules/.local/inputs.npz", L + "free-stack/.local/inputs.npz"]),
    ("brief-grammar", "CPU", s_brief_grammar,
     [L + f for f in ("free-brief/.local/contact_broadcast.npz", "free-brief/.local/detect.npz",
                      "free-grammar/.local/touches.npz", "free-grammar/.local/em_prior.json")]),
    ("stoppages-audio", "CPU", s_stoppages_audio, [L + "free-stoppages/.local/audio.npz"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="stop")
    ap.add_argument("--only", help="a stage, or a comma-separated list")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--lane-tdeed", action="store_true",
                    help="run T-DEED with the lane's own cudnn.benchmark = True (does not repeat bit for bit)")
    a = ap.parse_args()
    global LANE_TDEED
    LANE_TDEED = a.lane_tdeed
    names = [s[0] for s in STAGES]
    if a.list:
        for name, dev, _, outs in STAGES:
            print(f"{name:24s} {dev}  {', '.join(outs)}")
        return
    if a.only:      # one stage, or several separated by commas (run in the build's order)
        wanted = a.only.split(",")
        assert all(w in names for w in wanted), wanted
        selected = [st for st in STAGES if st[0] in wanted]
    else:
        i0 = names.index(a.start) if a.start else 0
        i1 = names.index(a.stop) + 1 if a.stop else len(STAGES)
        selected = STAGES[i0:i1]
    LOGS.mkdir(parents=True, exist_ok=True)
    build(selected)


def build(stages):
    for name, dev, fn, outs in stages:
        log = LOGS / f"{name}.log"
        t0 = time.time()
        print(f"=== {name} ({dev})", flush=True)
        for o in outs:      # some lanes' scripts assume their .local/ already exists (a clean checkout has none)
            (ROOT / o).parent.mkdir(parents=True, exist_ok=True)
        fn(log)
        dt = time.time() - t0
        entry = {"device": dev, "seconds": round(dt, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "outputs": outs}
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}   # re-read: stages may run side by side
        manifest[name] = entry
        manifest = {s[0]: manifest[s[0]] for s in STAGES if s[0] in manifest}
        MANIFEST.write_text(json.dumps(manifest, indent=1))
        print(f"=== {name} done in {dt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
