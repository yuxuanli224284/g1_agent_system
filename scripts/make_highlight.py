"""Caption each recorded episode (scenario, mode, result, key errors) and join them into one highlight video.
python scripts/make_highlight.py --runs runs/videos_rule_gear [more dirs] --out runs/highlight_rule_gear.mp4
Writes <episode>/episode_h264.mp4 (captioned, H.264, plays in browsers / players) and the concatenated --out."""
import argparse, json, os, subprocess, tempfile

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"     # has CJK glyphs
LEGEND = "红=root目标 蓝=头部估计手腕目标 紫=腕部修正 绿=真值 黄=实际到达"


def caption(r):
    m = r["metrics"]
    ok = "成功" if m.get("success") else "失败"
    f = r.get("factor") or {}
    fac = f" | 因素 {f.get('name')}={f.get('value') if not isinstance(f.get('value'), float) else round(f['value'], 3)}" if f.get("name") else ""
    if r["task"] == 1:
        err = (f"到位误差 {m.get('execution_err_pos_m', float('nan')) * 100:.1f} cm / {m.get('execution_err_yaw_deg', float('nan')):.1f}°, "
               f"观察距离 {m.get('d_actual_m', float('nan')):.2f} m (目标 {m.get('d_obs_harness', float('nan')):.2f})")
    else:
        err = (f"手腕执行误差 {m.get('execution_err_m', float('nan')) * 100:.1f} cm, "
               f"感知误差 {m.get('wrist_perception_err_m', float('nan')) * 100:.1f} cm")
    line1 = f"{r['scenario']} | 任务{r['task']} | {r['mode']} | {ok}{fac}"
    reason = "" if m.get("success") else f" | {str((r.get('harness_finish') or {}).get('summary', ''))[:70]}"
    return line1, err + reason


def esc(s):
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parts = []
    for d in a.runs:
        for ep in sorted(os.listdir(d)):
            src, rp = os.path.join(d, ep, "episode.mp4"), os.path.join(d, ep, "result.json")
            if not (os.path.exists(src) and os.path.exists(rp)):
                continue
            r = json.load(open(rp))
            l1, l2 = caption(r)
            dst = os.path.join(d, ep, "episode_h264.mp4")
            color = "0x2e7d32" if r["metrics"].get("success") else "0xc62828"
            vf = (f"drawbox=x=0:y=0:w=iw:h=92:color=black@0.6:t=fill,"
                  f"drawtext=fontfile={FONT}:text='{esc(l1)}':x=16:y=10:fontsize=26:fontcolor=white:box=1:boxcolor={color}@0.9:boxborderw=6,"
                  f"drawtext=fontfile={FONT}:text='{esc(l2)}':x=16:y=54:fontsize=22:fontcolor=white,"
                  f"drawtext=fontfile={FONT}:text='{esc(LEGEND)}':x=16:y=h-34:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=5")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "23", "-r", "10", dst], check=True)
            parts.append(dst)
            print("captioned", dst, "|", l1, flush=True)
    if not parts:
        raise SystemExit("no recorded episodes found")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
        lst = f.name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", a.out], check=True)
    os.remove(lst)
    print("HIGHLIGHT", a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB", len(parts), "episodes")


if __name__ == "__main__":
    main()
