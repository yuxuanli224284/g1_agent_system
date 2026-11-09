"""Per-barcode clips of a task-3 episode video: one clip per barcode from its barcode_start to its barcode_end stage
(stages.jsonl, matched to video frames through episode.frames.jsonl t_unix), 720p H.264, plus the whole episode at 720p.
Clip names carry the barcode's order, its tour face (scan_plan face_id, I4; multi-face stacks only), height (the
planned centre) and final status. --per_face also cuts one clip per tour face (face_start .. face_end stages, I3).
python scripts/make_barcode_clips.py <episode_dir> <out_dir> [--prefix rule_] [--per_face]"""
import argparse
import bisect
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1agent.eval.scan_order import FACE_NAMES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ep"); ap.add_argument("out"); ap.add_argument("--prefix", default="")
ap.add_argument("--per_face", action="store_true", help="also one clip per tour face (face_start .. face_end)")
a = ap.parse_args()
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
os.makedirs(a.out, exist_ok=True)
fr = [json.loads(l) for l in open(os.path.join(a.ep, "episode.frames.jsonl"))]
tu = [f["t_unix"] for f in fr]
st = [json.loads(l) for l in open(os.path.join(a.ep, "stages.jsonl"))]
res = json.load(open(os.path.join(a.ep, "result.json")))
plan = {p["id"]: p for p in res["world_state"]["scan_plan"]}


def frame_at(t):
    return fr[min(max(0, bisect.bisect_left(tu, t)), len(fr) - 1)]["frame"]


def enc(vf, path):
    subprocess.run([FF, "-nostdin", "-v", "error", "-y", "-i", os.path.join(a.ep, "episode.mp4"), "-vf", vf, "-r", "5",
                    "-c:v", "libx264", "-crf", "26", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                   check=True)


def clip(t0, t1, name):
    f0, f1 = frame_at(t0), frame_at(t1) + 5
    enc(f"select=between(n\\,{f0}\\,{f1}),setpts=N/5/TB,scale=1280:720", os.path.join(a.out, name))
    print("CLIP", name, f0, f1, flush=True)


starts, face_starts = {}, {}
k = 0
for s in st:
    i = s["info"] or {}
    if s["stage"] == "barcode_start":
        starts[i["id"]] = s["t_unix"]
    elif s["stage"] == "barcode_end" and i.get("id") in starts:
        k += 1
        p = plan.get(i["id"], {})
        z = (p.get("center_odom") or [0, 0, float("nan")])[2]
        face = f"_f{p['face_id']}" if p.get("face_id") is not None else ""      # legacy single-face plans: no face_id
        clip(starts[i["id"]], s["t_unix"], f"{a.prefix}{k}{face}_{i['id']}_z{z:.2f}m_{i['status']}.mp4")
    elif a.per_face and s["stage"] == "face_start":
        face_starts[i["face"]] = s["t_unix"]
    elif a.per_face and s["stage"] == "face_end" and i.get("face") in face_starts:
        st_ = list((i.get("statuses") or {}).values())
        clip(face_starts[i["face"]], s["t_unix"],
             f"{a.prefix}face{i['face']}_{FACE_NAMES[i['face']]}_{sum(x == 'scanned' for x in st_)}of{len(st_)}_scanned.mp4")
enc("scale=1280:720", os.path.join(a.out, f"{a.prefix}0_full_episode.mp4"))
print("FULL", flush=True)
