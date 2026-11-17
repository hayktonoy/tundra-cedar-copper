"""Finite public DFL ingestion. No models, challenge inputs, or external writes."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / "independent-possession-data"
LOCAL = BASE / ".local"
RAW = LOCAL / "raw"
DATASET = LOCAL / "dataset"
KEYS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
RAW_CAP, ARRAY_CAP, RSS_CAP = 3_000_000_000, 1_000_000_000, 600_000_000
REUSED_IDS = (51643520, 51643478)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_bytes(folder, suffix=None):
    return sum(p.stat().st_size for p in folder.rglob("*")
               if p.is_file() and (suffix is None or p.suffix == suffix)) if folder.exists() else 0


def array_bytes():
    return file_bytes(DATASET, ".npy")+file_bytes(LOCAL / "preparation_attempts", ".npy")


class Guard:
    def __init__(self, deadline):
        if os.name != "nt":
            raise RuntimeError("Windows affinity/RSS guard required")
        self.started = time.monotonic()
        self.deadline = dt.datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        self.end = self.started + (self.deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()
        self.last_log, self.peak, self.samples = 0., 0, 0
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        self.handle = kernel.GetCurrentProcess()
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        available, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(self.handle, ctypes.byref(available), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        mask, remaining = 0, available.value
        for _ in range(min(4, remaining.bit_count())):
            bit = remaining & -remaining
            mask |= bit
            remaining ^= bit
        if not mask or not kernel.SetProcessAffinityMask(self.handle, mask):
            raise ctypes.WinError(ctypes.get_last_error())
        self.affinity = mask
        class Memory(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in
                ("peak", "working", "qpp", "qp", "qnp", "qn", "page", "peakpage")]
        self.Memory = Memory
        self.get_memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        self.get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(Memory), ctypes.c_ulong]
        self.get_memory.restype = ctypes.c_int
        self.check("start", force=True, affinity_mask=mask, affinity_cpu_count=mask.bit_count(),
                   free_c_bytes=shutil.disk_usage("C:/").free, pid=os.getpid())

    def check(self, stage, force=False, **fields):
        now = time.monotonic()
        memory = self.Memory()
        memory.cb = ctypes.sizeof(memory)
        if not self.get_memory(self.handle, ctypes.byref(memory), memory.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        self.samples += 1
        self.peak = max(self.peak, memory.working)
        if memory.working > RSS_CAP:
            raise RuntimeError(f"Sampled RSS cap exceeded: {memory.working}")
        if now >= self.end-25:
            raise TimeoutError("Aggregate deadline reserve reached")
        if force or now-self.last_log >= 15:
            row = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), stage=stage,
                       elapsed_s=round(now-self.started, 3), rss_bytes=memory.working, **fields)
            with (LOCAL / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
            self.last_log = now

    def stats(self):
        return dict(elapsed_s=round(time.monotonic()-self.started, 3),
                    sampled_process_tree_peak_rss_bytes=self.peak, rss_samples=self.samples,
                    affinity_mask=self.affinity, affinity_cpu_count=self.affinity.bit_count(),
                    child_processes=0, cpu_time_s=round(time.process_time(), 3),
                    new_raw_bytes=file_bytes(RAW), derived_npy_bytes=array_bytes())


def download_md5(path, guard):
    checksum = hashlib.md5()
    count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024*1024):
            checksum.update(chunk)
            count += len(chunk)
            guard.check("verify_download", file=path.name, bytes=count)
    return count, checksum.hexdigest()


class TLSOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise RuntimeError("Non-TLS redirect rejected")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_get(url, headers):
    request = urllib.request.Request(url, headers=dict(
        {"User-Agent": "dfl-independent-preparation/1.0", "Accept-Encoding": "identity"}, **headers))
    return urllib.request.build_opener(TLSOnly()).open(request, timeout=20)


def fetch(entry, guard, offline):
    name, size, ident = entry["name"], entry["size"], entry["id"]
    if Path(name).name != name or entry["download_url"] != f"https://ndownloader.figshare.com/files/{ident}":
        raise RuntimeError("Unexpected publisher path/URL")
    if entry["computed_md5"] != entry["supplied_md5"]:
        raise RuntimeError("Publisher MD5 fields disagree")
    reused = ident in REUSED_IDS
    target = (OLD / ".local/ingestion_feasibility/raw" / name) if reused else RAW / name
    if target.exists():
        downloaded_size, md5 = download_md5(target, guard)
    else:
        if reused or offline:
            raise RuntimeError(f"Required verified file absent: {target}")
        if file_bytes(RAW)+size > RAW_CAP:
            raise RuntimeError("New raw download cap would be exceeded")
        if shutil.disk_usage("C:/").free < size+1_100_000_000:
            raise RuntimeError("Insufficient C drive capacity before download")
        partial = target.with_suffix(".xml.part")
        state_path = LOCAL / "transfers" / f"{ident}.json"
        if partial.exists() or state_path.exists():
            raise RuntimeError("Preserved incomplete transfer exists; no implicit cross-run resume")
        state = dict(file_id=ident, url=entry["download_url"], size=size, status="incomplete",
                     retry_policy="At most one in-run retry per range; no implicit cross-invocation resume",
                     ranges=[])
        count = 4 if size > 3_000_000 else 1
        for index in range(count):
            state["ranges"].append(dict(start=size*index//count, end=size*(index+1)//count-1,
                                         bytes=0, attempts=[]))
        save(state_path, state)
        with partial.open("xb") as stream:
            stream.truncate(size)
        stop = threading.Event()
        def worker(item):
            for attempt_number in range(2):
                start, end = item["start"]+item["bytes"], item["end"]
                attempt = dict(number=attempt_number+1, start=start, end=end)
                item["attempts"].append(attempt)
                try:
                    headers = {"Range": f"bytes={start}-{end}"} if count == 4 or start else {}
                    with open_get(entry["download_url"], headers) as response:
                        expected_status = 206 if headers else 200
                        attempt.update(http_status=response.status, final_tls_url=response.geturl(),
                                       content_range=response.headers.get("Content-Range"),
                                       content_length=response.headers.get("Content-Length"))
                        if response.status != expected_status or urllib.parse.urlsplit(response.geturl()).scheme != "https":
                            raise ValueError("Unexpected HTTP status or non-TLS response")
                        if headers and response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise ValueError("Incorrect HTTP Content-Range")
                        length = response.headers.get("Content-Length")
                        if length is not None and int(length) != end-start+1:
                            raise ValueError("Incorrect HTTP Content-Length")
                        with partial.open("r+b", buffering=0) as output:
                            output.seek(start)
                            remaining = end-start+1
                            while remaining:
                                if stop.is_set() or time.monotonic() >= guard.end-45:
                                    raise TimeoutError("Transfer deadline/stop reached")
                                # read1 performs at most one underlying buffered read;
                                # the independent watchdog enforces the absolute wall
                                # even for a TLS peer that dribbles bytes indefinitely.
                                chunk = response.read1(min(1024*1024, remaining))
                                if not chunk:
                                    raise OSError("Premature response EOF")
                                output.write(chunk)
                                item["bytes"] += len(chunk)
                                remaining -= len(chunk)
                            if response.read(1):
                                raise ValueError("HTTP body exceeds exact requested size")
                    attempt["status"] = "complete"
                    return
                except Exception as error:
                    attempt.update(status="error", error_type=type(error).__name__, error=str(error))
                    if isinstance(error, (ValueError, TimeoutError)) or stop.is_set() or attempt_number == 1:
                        raise
                    if item["bytes"] == item["end"]-item["start"]+1:
                        raise
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=count)
        futures = [pool.submit(worker, item) for item in state["ranges"]]
        try:
            last_save = 0.
            while not all(future.done() for future in futures):
                for future in futures:
                    if future.done():
                        future.result()
                guard.check("download", file_id=ident, bytes=sum(r["bytes"] for r in state["ranges"]),
                            expected_bytes=size, concurrent_network_requests=count)
                if time.monotonic()-last_save > 15:
                    save(state_path, state)
                    last_save = time.monotonic()
                concurrent.futures.wait(futures, timeout=1)
            for future in futures:
                future.result()
            state["status"] = "transfer_complete"
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
            save(state_path, state)
        downloaded_size, md5 = download_md5(partial, guard)
        if downloaded_size != size or md5 != entry["computed_md5"]:
            state["status"] = "integrity_error"
            save(state_path, state)
            raise RuntimeError("Downloaded integrity mismatch; preserved, no retry")
        partial.rename(target)
        state["status"] = "verified"
        save(state_path, state)
    if downloaded_size != size or md5 != entry["computed_md5"]:
        raise RuntimeError("Existing raw size/MD5 mismatch")
    guard.check("source_verified", force=True, file_id=ident, reused_read_only=reused, size=downloaded_size)
    return dict(id=ident, name=name, path=str(target), publisher_size=size,
                source_url=entry["download_url"], reused_read_only=reused, size=downloaded_size)


def selected_sources():
    source = OLD / ".local/figshare_article_28196177.json"
    payload = source.read_bytes()
    metadata = json.loads(payload)
    pairs = {key: {} for key in KEYS}
    for entry in metadata["files"]:
        key = next((key for key in KEYS if f"DFL-MAT-{key}.xml" in entry["name"]), None)
        kind = "positions" if "positions_raw_observed" in entry["name"] else "matchinformation" if "matchinformation" in entry["name"] else None
        if key and kind:
            if kind in pairs[key]:
                raise RuntimeError("Duplicate publisher source entry")
            pairs[key][kind] = entry
    if any(set(pair) != {"positions", "matchinformation"} for pair in pairs.values()):
        raise RuntimeError("Missing authorized match pair")
    if sum(entry["size"] for pair in pairs.values() for entry in pair.values() if entry["id"] not in REUSED_IDS) > RAW_CAP:
        raise RuntimeError("Publisher new raw allocation exceeds cap")
    return metadata, pairs


def xml_events(path):
    """Yield records while removing both Frame and FrameSet children immediately."""
    stack = []
    for event, element in ET.iterparse(path, events=("start", "end")):
        name = element.tag.rsplit("}", 1)[-1]
        if event == "start":
            stack.append(element)
            yield event, name, element.attrib
        else:
            yield event, name, element.attrib
            element.clear()
            if len(stack) > 1:
                stack[-2].remove(element)
            stack.pop()


def roster_metadata(path, key, guard):
    general, environment, teams, people, officials, team_id = {}, {}, {}, {}, {}, None
    for event, name, attrs in xml_events(path):
        if event == "start":
            if name == "General":
                general = dict(attrs)
            elif name == "Environment":
                environment = dict(attrs)
            elif name == "Team":
                team_id = attrs["TeamId"]
                if team_id in teams:
                    raise RuntimeError("Duplicate roster TeamId")
                teams[team_id] = dict(attrs)
            elif name == "Player":
                ident = attrs["PersonId"]
                if ident in people or team_id is None:
                    raise RuntimeError("Duplicate/unparented roster PersonId")
                people[ident] = dict(attrs, TeamId=team_id)
            elif name == "Referee":
                ident = attrs["PersonId"]
                if ident in officials:
                    raise RuntimeError("Duplicate supplied official ID")
                officials[ident] = dict(attrs)
        elif name == "Team":
            team_id = None
    if general.get("MatchId") != f"DFL-MAT-{key}":
        raise RuntimeError("Match-information key mismatch")
    home = [ident for ident, attrs in teams.items() if attrs.get("Role") == "home"]
    away = [ident for ident, attrs in teams.items() if attrs.get("Role") in ("guest", "away")]
    if len(teams) != 2 or len(home) != 1 or len(away) != 1:
        raise RuntimeError("Expected explicit home and away team metadata")
    team_ids = [home[0], away[0]]
    slots, lookup = [], {}
    for axis, ident in enumerate(team_ids):
        ids = sorted(pid for pid, attrs in people.items() if attrs["TeamId"] == ident)
        if len(ids) > 32:
            raise RuntimeError("More than 32 supplied roster slots")
        rows = []
        for slot, pid in enumerate(ids):
            attrs = people[pid]
            lookup[pid] = (axis, slot)
            rows.append(dict(person_id=pid, team_id=ident,
                             playing_position=attrs.get("PlayingPosition"),
                             goalkeeper=attrs.get("PlayingPosition") == "TW",
                             starting=attrs.get("Starting"), shirt_number=attrs.get("ShirtNumber")))
        slots.append(rows+[None]*(32-len(rows)))
    pitch = [float(environment["PitchX"]), float(environment["PitchY"])]
    if any(not math.isfinite(value) or value <= 0 for value in pitch):
        raise RuntimeError("Invalid supplied pitch dimensions")
    guard.check("roster", force=True, match=key, roster_players=len(people))
    if set(officials) & set(people):
        raise RuntimeError("Official/player roster IDs overlap")
    return dict(general=general, team_ids=team_ids, teams=teams, slots=slots, officials=officials,
                lookup=lookup, pitch_dimensions_m=pitch)


def identity(attrs, roster, key):
    section = attrs.get("GameSection")
    if section not in ("firstHalf", "secondHalf") or attrs.get("MatchId") != f"DFL-MAT-{key}":
        raise RuntimeError("Unexpected half or FrameSet match ID")
    ball = attrs.get("TeamId") == "BALL"
    pid = attrs.get("PersonId")
    if ball:
        if not pid or pid in roster["lookup"] or pid in roster.get("officials", {}):
            raise RuntimeError("BALL identity is missing or belongs to supplied player roster")
        known = roster.setdefault("provider_ball_person_id", pid)
        if pid != known:
            raise RuntimeError("Inconsistent provider ball identity within match")
        axis, slot = None, None
    elif attrs.get("TeamId") == "referee":
        if pid not in roster.get("officials", {}):
            raise RuntimeError("Tracked referee absent from supplied official metadata")
        axis, slot = -1, None
    else:
        if pid not in roster["lookup"]:
            raise RuntimeError(f"Tracked PersonId absent from roster: {pid}")
        axis, slot = roster["lookup"][pid]
        if attrs.get("TeamId") != roster["team_ids"][axis]:
            raise RuntimeError("Tracked player assigned to wrong supplied team")
    return section, ball, pid, axis, slot


def parse_timestamp(raw):
    if raw is None:
        return None, "missing_timestamp"
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Unzoned timestamp")
    except (ValueError, TypeError):
        return None, "invalid_timestamp"
    return parsed, None


def scan(path, roster, key, guard):
    halves, schema, order, current, total, pitch = {}, {}, [], None, 0, None
    previous_n = previous_t = None
    for event, name, attrs in xml_events(path):
        if event == "start":
            schema.setdefault(name, set()).update(attrs)
            if len(schema) > 64 or any(len(keys) > 64 for keys in schema.values()):
                raise RuntimeError("Unexpected schema cardinality")
            if name == "PitchSize":
                pitch = [float(attrs["X"]), float(attrs["Y"])]
            if name == "FrameSet":
                section, ball, pid, axis, slot = identity(attrs, roster, key)
                if not order or order[-1] != section:
                    if section in order:
                        raise RuntimeError("Noncontiguous source halves; refusing multiple live half maps")
                    order.append(section)
                half = halves.setdefault(section, dict(min_n=None, max_n=None, object_frame_records=0,
                    ball_records=0, origin_frame=None, origin_timestamp=None, framesets=[],
                    clock_counts=collections.Counter(), clock_examples=[]))
                if len(half["framesets"]) >= 200:
                    raise RuntimeError("Too many FrameSets")
                current = dict(attributes=dict(attrs), frames=0, first_n=None, last_n=None,
                               excluded_official=(axis == -1),
                               missing_internal_frame_numbers=0)
                half["framesets"].append(current)
                previous_n = previous_t = None
            continue
        if name == "Frame":
            if current is None:
                raise RuntimeError("Frame outside FrameSet")
            try:
                number = int(attrs["N"])
            except (KeyError, ValueError):
                raise RuntimeError("Invalid/missing source frame counter; cannot index")
            if not 0 <= number <= 10_000_000:
                raise RuntimeError("Source counter outside bounded range")
            current["frames"] += 1
            if current["first_n"] is None:
                current["first_n"] = number
            current["last_n"] = number
            if current["excluded_official"]:
                half["clock_counts"]["excluded_official_records"] += 1
                official_t, issue = parse_timestamp(attrs.get("T"))
                if issue:
                    half["clock_counts"]["excluded_official_"+issue] += 1
                if previous_n is not None:
                    step = number-previous_n
                    if step <= 0:
                        half["clock_counts"]["excluded_official_nonincreasing_counter"] += 1
                    if step > 1:
                        current["missing_internal_frame_numbers"] += step-1
                    if official_t is not None and previous_t is not None:
                        residual = round((official_t-previous_t).total_seconds()*1_000_000)-step*40000
                        if residual:
                            half["clock_counts"]["excluded_official_timestamp_counter_residual"] += 1
                previous_n, previous_t = number, official_t
                total += 1
                if total % 2048 == 0:
                    guard.check("scan_excluded_official", match=key, records=total)
                continue
            half["object_frame_records"] += 1
            half["ball_records"] += int(ball)
            half["min_n"] = number if half["min_n"] is None else min(number, half["min_n"])
            half["max_n"] = number if half["max_n"] is None else max(number, half["max_n"])
            if half["max_n"]-half["min_n"]+1 > 120000:
                raise RuntimeError("Half frame grid exceeds 120000")
            if previous_n is not None:
                delta = number-previous_n
                if delta <= 0:
                    half["clock_counts"]["nonincreasing_counter"] += 1
                if delta > 1:
                    current["missing_internal_frame_numbers"] += delta-1
            timestamp, issue = parse_timestamp(attrs.get("T"))
            if timestamp is not None:
                if half["origin_frame"] is None:
                    half["origin_frame"], half["origin_timestamp"] = number, timestamp.isoformat()
                    half["_origin"] = timestamp
                residual = round((timestamp-half["_origin"]).total_seconds()*1_000_000)-(number-half["origin_frame"])*40000
                if residual:
                    half["clock_counts"]["timestamp_counter_residual"] += 1
                    if len(half["clock_examples"]) < 8:
                        half["clock_examples"].append(dict(n=number, timestamp=attrs["T"], residual_us=residual))
                if previous_t is not None and timestamp <= previous_t:
                    half["clock_counts"]["nonincreasing_timestamp"] += 1
            if issue:
                half["clock_counts"][issue] += 1
                if len(half["clock_examples"]) < 8:
                    half["clock_examples"].append(dict(n=number, issue=issue))
            previous_n, previous_t = number, timestamp
            total += 1
            if total % 2048 == 0:
                guard.check("schema_clock_scan", match=key, records=total)
        elif name == "FrameSet":
            current = None
    if set(halves) != {"firstHalf", "secondHalf"} or pitch != roster["pitch_dimensions_m"]:
        raise RuntimeError("Missing half or inconsistent pitch metadata")
    for half in halves.values():
        if half["origin_frame"] is None:
            raise RuntimeError("No valid timestamp origin")
        origin = half.pop("_origin")
        half["grid_origin_timestamp"] = (origin+dt.timedelta(milliseconds=40*(half["min_n"]-half["origin_frame"]))).isoformat()
        half["T"] = half["max_n"]-half["min_n"]+1
    catalog = dict(match=key, halves=halves, order=order, schema={name: sorted(keys) for name, keys in schema.items()},
                   pitch_dimensions_m=pitch, total_object_frame_records=total)
    save(LOCAL / "audits" / f"{key}_catalog.json", catalog)
    return catalog


def code(raw, name):
    if raw is None:
        return -128
    if not re.fullmatch(r"0|-?[1-9][0-9]*", raw):
        raise RuntimeError(f"Unknown/noncanonical raw {name} token: {raw!r}")
    value = int(raw)
    if not -127 <= value <= 127:
        raise RuntimeError(f"Raw {name} cannot be represented without sentinel collision: {raw}")
    return value


def coordinates(attrs, quality):
    xy = []
    for axis in ("X", "Y"):
        raw = attrs.get(axis)
        if raw is None:
            quality[f"{axis}_missing"] += 1
            value = math.nan
        else:
            try:
                value = float(raw)
            except ValueError:
                quality[f"{axis}_invalid"] += 1
                value = math.nan
            else:
                if not math.isfinite(value):
                    quality[f"{axis}_nonfinite"] += 1
                    value = math.nan
        if math.isfinite(value) and abs(value) > 1e10:
            raise RuntimeError("Coordinate magnitude outside finite float32 audit bound")
        xy.append(value)
    return xy


def missing_ranges(mask, first):
    ranges, start = [], None
    for index in range(len(mask)+1):
        missing = index < len(mask) and not bool(mask[index])
        if missing and start is None:
            start = first+index
        elif not missing and start is not None:
            ranges.append([start, first+index-1])
            start = None
    return ranges


def close_maps(arrays):
    for array in arrays.values():
        array.flush()
        array._mmap.close()


def allocate_half(np, key, section, half, roster, guard):
    folder = DATASET / key / section
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise RuntimeError("Half output already exists; preserving, no implicit preprocessing restart")
    T = half["T"]
    specs = dict(players=("<f4", (T, 2, 32, 2)), ball=("<f4", (T, 2)),
                 provider_possession=("i1", (T,)), provider_status=("i1", (T,)),
                 frame_numbers=("<i8", (T,)), goalkeeper=("?", (2, 32)))
    expected = sum(math.prod(shape)*np.dtype(dtype).itemsize+256 for dtype, shape in specs.values())
    if array_bytes()+expected > ARRAY_CAP:
        raise RuntimeError("Derived NPY allocation cap would be exceeded")
    arrays = {name: np.lib.format.open_memmap(folder / f"{name}.npy", mode="w+", dtype=dtype,
                                             shape=shape, fortran_order=False, version=(1, 0))
              for name, (dtype, shape) in specs.items()}
    arrays["players"][:] = np.nan
    arrays["ball"][:] = np.nan
    arrays["provider_possession"][:] = -128
    arrays["provider_status"][:] = -128
    arrays["frame_numbers"][:] = np.arange(half["min_n"], half["max_n"]+1, dtype="<i8")
    arrays["goalkeeper"][:] = False
    for axis, slots in enumerate(roster["slots"]):
        for slot, info in enumerate(slots):
            arrays["goalkeeper"][axis, slot] = info is not None and info["goalkeeper"]
    guard.check("half_allocated", force=True, match=key, half=section, T=T)
    return dict(folder=folder, arrays=arrays, seen_players=np.zeros((T, 2, 32), dtype=bool),
                seen_ball=np.zeros(T, dtype=bool), raw_possession=collections.Counter(),
                raw_status=collections.Counter(), coordinate_quality=collections.Counter(),
                jump_counts=collections.Counter(), jump_examples=[], max_step_m={"ball": 0., "player": 0.})


def finish_half(np, state, key, section, half, roster, sources, catalog, guard):
    arrays, folder, T = state["arrays"], state["folder"], half["T"]
    availability = np.zeros((2, 32), dtype=np.int64)
    finite_counts = np.zeros((2, 32), dtype=np.int64)
    active = [collections.Counter(), collections.Counter()]
    finite_active = [collections.Counter(), collections.Counter()]
    finite_ball = 0
    xy_extents = {name: [None, None] for name in ("players_X", "players_Y", "ball_X", "ball_Y")}
    for first in range(0, T, 4096):
        last = min(T, first+4096)
        observed = state["seen_players"][first:last]
        finite = np.isfinite(arrays["players"][first:last]).all(axis=-1)
        availability += observed.sum(axis=0)
        finite_counts += finite.sum(axis=0)
        finite_ball += int(np.isfinite(arrays["ball"][first:last]).all(axis=-1).sum())
        for axis in range(2):
            values, counts = np.unique(observed[:, axis].sum(axis=-1), return_counts=True)
            active[axis].update({str(int(v)): int(n) for v, n in zip(values, counts)})
            values, counts = np.unique(finite[:, axis].sum(axis=-1), return_counts=True)
            finite_active[axis].update({str(int(v)): int(n) for v, n in zip(values, counts)})
            for kind in ("players", "ball"):
                block = arrays[kind][first:last, ..., axis]
                values = block[np.isfinite(block)]
                if values.size:
                    entry = xy_extents[f"{kind}_{'XY'[axis]}"]
                    low, high = float(values.min()), float(values.max())
                    entry[0] = low if entry[0] is None else min(low, entry[0])
                    entry[1] = high if entry[1] is None else max(high, entry[1])
        for axis, slots in enumerate(roster["slots"]):
            for slot, info in enumerate(slots):
                if info is None and not np.isnan(arrays["players"][first:last, axis, slot]).all():
                    raise RuntimeError("Padded roster coordinate is populated")
        guard.check("half_validation", match=key, half=section, checked_frames=last)
    label_stats = {}
    for name, valid in (("provider_possession", (1, 2)), ("provider_status", (0, 1))):
        values, counts = np.unique(arrays[name], return_counts=True)
        numeric = {str(int(v)): int(n) for v, n in zip(values, counts)}
        label_stats[name] = dict(serialized_counts=numeric,
            raw_token_counts=dict(state["raw_possession" if name.endswith("possession") else "raw_status"]),
            documented_count=sum(n for v, n in numeric.items() if int(v) in valid),
            missing_count=numeric.get("-128", 0),
            unknown_count=sum(n for v, n in numeric.items() if int(v) not in (*valid, -128)))
    metadata = dict(contract_version=2, match=key, half=section, planned_partition="HOLDOUT" if key == "J03WR9" else "TRAIN",
        sources=sources, pitch_dimensions_m=roster["pitch_dimensions_m"],
        units="metres (publisher paper DOI 10.1038/s41597-025-04505-y); no spatial transforms",
        team_axis_roles=["home", "away"], team_ids=roster["team_ids"], team_metadata=roster["teams"],
        provider_ball_person_id=roster["provider_ball_person_id"],
        slots=roster["slots"], slot_ids=[[row["person_id"] if row else None for row in team] for team in roster["slots"]],
        T=T, frame_number_range=[half["min_n"], half["max_n"]],
        time_origin=half["grid_origin_timestamp"], origin_frame=half["min_n"],
        rate_hz=25, timestamp_increment_us=40000, clock_audit=half,
        kept_ranges=[[half["min_n"], half["max_n"]]], excluded_ranges=[],
        excluded_official_entity_ranges=[dict(person_id=row["attributes"]["PersonId"],
            supplied_role=roster["officials"][row["attributes"]["PersonId"]]["Role"],
            first_n=row["first_n"], last_n=row["last_n"], frames=row["frames"])
            for row in half["framesets"] if row.get("excluded_official")],
        ball_missing_grid_ranges=missing_ranges(state["seen_ball"], half["min_n"]),
        ball_observed_count=int(state["seen_ball"].sum()), ball_finite_xy_count=finite_ball,
        player_availability_counts=availability.tolist(), player_finite_xy_counts=finite_counts.tolist(),
        active_player_count_histograms=active, finite_player_count_histograms=finite_active,
        labels=label_stats, coordinate_quality=state["coordinate_quality"], xy_extents_m=xy_extents,
        jumps=dict(threshold_m=20, adjacent_frame_only=True, counts=state["jump_counts"],
                   max_step_m=state["max_step_m"], examples=state["jump_examples"]),
        schema=catalog["schema"], label_caveat="Raw provider annotations in the original categories")
    save(folder / "metadata.json", metadata)
    samples = sorted({0, T//2, T-1})
    snapshots = {name: np.array(array if name == "goalkeeper" else array[samples], copy=True)
                 for name, array in arrays.items()}
    expected_specs = {name: (array.shape, array.dtype.str) for name, array in arrays.items()}
    close_maps(arrays)
    arrays.clear()
    manifests = {}
    for name, (shape, dtype) in expected_specs.items():
        path = folder / f"{name}.npy"
        reopened = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if reopened.shape != shape or reopened.dtype.str != dtype or not reopened.flags.c_contiguous:
                raise RuntimeError("Serialized NPY shape/dtype/order mismatch")
            actual = reopened if name == "goalkeeper" else reopened[samples]
            if not np.array_equal(actual, snapshots[name], equal_nan=True):
                raise RuntimeError("Serialization spot check mismatch")
            if name == "frame_numbers" and (int(reopened[0]) != half["min_n"] or int(reopened[-1]) != half["max_n"]):
                raise RuntimeError("Serialized frame grid endpoints mismatch")
            offset = reopened.offset
        finally:
            reopened._mmap.close()
        with path.open("rb") as stream:
            header = stream.read(offset)
        manifests[f"{name}.npy"] = dict(shape=list(shape), dtype=dtype, order="C", data_offset=offset,
                                         npy_header_hex=header.hex(), size=path.stat().st_size)
    manifest = dict(match=key, half=section, contract_version=2, arrays=manifests,
                    validation="read-only reopen, no pickle; 3 bounded sample rows and roster/sentinel/count checks")
    save(folder / "manifest.json", manifest)
    guard.check("half_complete", force=True, match=key, half=section, T=T,
                ball_observed=int(state["seen_ball"].sum()))
    return dict(path=str(folder / "manifest.json"), frames=T, arrays=manifests)


def write_match(np, key, sources, roster, catalog, guard):
    state, current_section, current, previous_xy, previous_n = None, None, None, None, None
    results, total = {}, 0
    try:
        for event, name, attrs in xml_events(Path(sources["positions"]["path"])):
            if event == "start" and name == "FrameSet":
                section, ball, pid, axis, slot = identity(attrs, roster, key)
                if section != current_section:
                    if state is not None:
                        results[current_section] = finish_half(np, state, key, current_section,
                            catalog["halves"][current_section], roster, sources, catalog, guard)
                        state = None
                    current_section = section
                    state = allocate_half(np, key, section, catalog["halves"][section], roster, guard)
                current = (ball, pid, axis, slot)
                previous_xy = previous_n = None
            elif event == "end" and name == "Frame":
                if current is None or state is None:
                    raise RuntimeError("Unparented output frame")
                ball, pid, axis, slot = current
                if axis == -1:
                    total += 1
                    if total % 2048 == 0:
                        guard.check("audit_excluded_official", match=key, object_records=total)
                    continue
                number = int(attrs["N"])
                index = number-catalog["halves"][current_section]["min_n"]
                if not 0 <= index < catalog["halves"][current_section]["T"]:
                    raise RuntimeError("Frame outside scanned grid")
                seen = state["seen_ball"][index] if ball else state["seen_players"][index, axis, slot]
                if seen:
                    raise RuntimeError(f"Duplicate {'ball' if ball else pid}/frame {number}")
                xy = coordinates(attrs, state["coordinate_quality"])
                if ball:
                    state["seen_ball"][index] = True
                    state["arrays"]["ball"][index] = xy
                    state["arrays"]["provider_possession"][index] = code(attrs.get("BallPossession"), "BallPossession")
                    state["arrays"]["provider_status"][index] = code(attrs.get("BallStatus"), "BallStatus")
                    state["raw_possession"][attrs.get("BallPossession", "<missing>")] += 1
                    state["raw_status"][attrs.get("BallStatus", "<missing>")] += 1
                else:
                    state["seen_players"][index, axis, slot] = True
                    state["arrays"]["players"][index, axis, slot] = xy
                kind = "ball" if ball else "player"
                if previous_xy is not None and previous_n == number-1 and all(math.isfinite(v) for v in xy+previous_xy):
                    distance = math.hypot(xy[0]-previous_xy[0], xy[1]-previous_xy[1])
                    state["max_step_m"][kind] = max(state["max_step_m"][kind], distance)
                    if distance > 20:
                        state["jump_counts"][kind] += 1
                        if len(state["jump_examples"]) < 8:
                            state["jump_examples"].append(dict(person_id=pid, kind=kind, n=number, step_m=distance))
                previous_xy, previous_n = xy, number
                total += 1
                if total % 2048 == 0:
                    guard.check("write_arrays", match=key, half=current_section, object_records=total)
            elif event == "end" and name == "FrameSet":
                current = None
        if state is not None:
            results[current_section] = finish_half(np, state, key, current_section,
                catalog["halves"][current_section], roster, sources, catalog, guard)
            state = None
    finally:
        if state is not None:
            close_maps(state["arrays"])
    if set(results) != {"firstHalf", "secondHalf"}:
        raise RuntimeError("Incomplete serialized match")
    return results


def prepare(inputs, guard):
    import numpy as np
    outputs = dict(contract_version=2, matches={}, numpy_version=np.__version__,
                   python_executable=sys.executable, split=dict(train=list(KEYS[:-1]), holdout="J03WR9"))
    for key in KEYS:
        sources = inputs["matches"][key]
        roster = roster_metadata(Path(sources["matchinformation"]["path"]), key, guard)
        catalog = scan(Path(sources["positions"]["path"]), roster, key, guard)
        outputs["matches"][key] = write_match(np, key, sources, roster, catalog, guard)
        save(LOCAL / "output_manifest.json", outputs)
    paths = dict(ingest_source=BASE / "ingest.py", data_contract=BASE / "DATA_CONTRACT.md",
                 publisher_metadata=OLD / ".local/figshare_article_28196177.json",
                 input_manifest=LOCAL / "input_manifest.json", output_manifest=LOCAL / "output_manifest.json")
    seal = dict(status="complete", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                ownership="Only ingestion source, contract, verified public raw inputs and dataset outputs; excludes DESIGN/model sources and model_preparation",
                artifacts={name: str(path) for name, path in paths.items()},
                source_file_count=14, half_count=14, array_count=84,
                total_frames=sum(half["frames"] for halves in outputs["matches"].values() for half in halves.values()),
                resources=guard.stats())
    save(LOCAL / "seal.json", seal)


def deadline_watchdog(pid, deadline):
    """Hold an exact Windows process handle; terminate only at the shared deadline."""
    limit = dt.datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    guard = Guard((limit+dt.timedelta(seconds=60)).isoformat())
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p]+[ctypes.POINTER(ctypes.c_uint64)]*4
    handle = kernel.OpenProcess(0x100000 | 0x1000 | 0x1, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    times = [ctypes.c_uint64() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
        kernel.CloseHandle(handle)
        raise ctypes.WinError(ctypes.get_last_error())
    receipt = dict(pid=pid, exact_process_creation_filetime=times[0].value,
                   absolute_deadline_utc=limit.isoformat(), status="watching")
    destination = LOCAL / f"watchdog_{pid}.json"
    save(destination, receipt)
    try:
        while True:
            result = kernel.WaitForSingleObject(handle, 1000)
            if result == 0:
                receipt["status"] = "worker_exited"
                break
            if result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
            if dt.datetime.now(dt.timezone.utc) >= limit-dt.timedelta(seconds=8):
                if not kernel.TerminateProcess(handle, 124):
                    raise ctypes.WinError(ctypes.get_last_error())
                receipt["status"] = "terminated_at_absolute_deadline"
                kernel.WaitForSingleObject(handle, 1000)
                break
            guard.check("deadline_watchdog", watched_pid=pid)
    finally:
        kernel.CloseHandle(handle)
        receipt.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), resources=guard.stats())
        save(destination, receipt)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fetch", "prepare", "all"), default="all")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--watchdog-pid", type=int,
                        help="Independently enforce deadline for this exact live Windows process")
    args = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(exist_ok=True)
    if args.watchdog_pid is not None:
        return deadline_watchdog(args.watchdog_pid, args.deadline_utc)
    guard = Guard(args.deadline_utc)
    status = dict(status="running", mode=args.mode)
    try:
        metadata, pairs = selected_sources()
        inputs = dict(publisher_metadata_path=str(OLD / ".local/figshare_article_28196177.json"),
                      license=metadata["license"], authors=[a["full_name"] for a in metadata["authors"]],
                      paper_doi="10.1038/s41597-025-04505-y", dataset_doi=metadata["doi"], matches={})
        free = shutil.disk_usage("C:/").free
        if free < sum(e["size"] for p in pairs.values() for e in p.values() if e["id"] not in REUSED_IDS)+ARRAY_CAP:
            raise RuntimeError("C drive cannot accommodate full planned allocation")
        for key, pair in pairs.items():
            inputs["matches"][key] = {}
            for kind in ("matchinformation", "positions"):
                inputs["matches"][key][kind] = fetch(pair[kind], guard, args.offline or args.mode == "prepare")
                save(LOCAL / "input_manifest.json", inputs)
        status["status"] = "sources_verified"
        if args.mode != "fetch":
            prepare(inputs, guard)
            status["status"] = "complete"
    except Exception as error:
        status.update(status="error", error_type=type(error).__name__, error=str(error))
        print(json.dumps(status), flush=True)
    finally:
        status["resources"] = guard.stats()
        save(LOCAL / f"{args.mode}_run_receipt.json", status)
        print(json.dumps(status), flush=True)
    return 0 if status["status"] in ("complete", "sources_verified") else 1


if __name__ == "__main__":
    sys.exit(main())
