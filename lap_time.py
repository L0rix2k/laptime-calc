#!/usr/bin/env python3
"""
Estimate a lap time for a track made by track_calc.py (reads track.json).

Usage:    python lap_time.py track.json
          python lap_time.py track.json --top-speed 225 --grip 2 --accel 7 --brake 25
          python lap_time.py track.json --top-speed 225 --grip 2 --accel 7 --brake 25 --target-time 1:35.911
          (speeds you type and see are in the game's speedometer units; track length stays real)

Needs only numpy:  pip install numpy

How it works: the slowest safe speed in every corner comes from the corner's radius and the
car's grip (v = sqrt(grip * g * radius)). Then it speeds up out of corners (limited by engine
power and drag) and brakes before them (limited by braking force) to find the fastest speed
at every point of the lap, and adds up the time.

Car numbers: top speed in km/h, grip in g (1 = as much sideways force as gravity),
accel and brake in m/s^2. They are rough guesses: tune them to match your Roblox car.
"""
import argparse
import json

import numpy as np

G = 9.81


def load_track(path, meters_per_stud):
    with open(path) as f:
        d = json.load(f)
    pts = np.array([[p["x"], p["z"]] for p in d["points"]], dtype=float) * meters_per_stud
    return pts, d


def curvature(pts):
    """Curvature (1/m) at each point of a closed loop, from each point and its neighbours."""
    n = len(pts)
    k = np.zeros(n)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        ab, bc, ca = np.hypot(*(b - a)), np.hypot(*(c - b)), np.hypot(*(a - c))
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))  # 2 x triangle area
        k[i] = 2 * area2 / (ab * bc * ca + 1e-12)
    return np.convolve(np.r_[k[-1], k, k[0]], np.ones(3) / 3, mode="valid")  # light smoothing


def speed_profile(pts, grip_g, top, accel, brake):
    n = len(pts)
    nxt = np.roll(pts, -1, axis=0)
    ds = np.hypot(*(nxt - pts).T)  # distance from point i to point i+1
    kappa = curvature(pts)

    v_corner = np.minimum(top, np.sqrt(grip_g * G / np.maximum(kappa, 1e-9)))
    v = v_corner.copy()

    for _ in range(3):  # two-ish laps so the loop closes on itself
        for i in range(n * 2):  # braking: look ahead, never be faster than we can slow down from
            j, k = (-i - 1) % n, (-i) % n
            v[j] = min(v[j], np.sqrt(v[k] ** 2 + 2 * brake * ds[j]))
        for i in range(n * 2):  # acceleration, weaker as drag builds up toward top speed
            j, k = i % n, (i + 1) % n
            a = accel * max(0.0, 1 - (v[j] / top) ** 2)
            v[k] = min(v[k], np.sqrt(v[j] ** 2 + 2 * a * ds[j]), v_corner[k])
    v_mid = np.maximum((v + np.roll(v, -1)) / 2, 1.0)
    dt = ds / v_mid
    return ds, v, dt


def parse_time(t):
    """'1:35.911' or '95.911' -> seconds"""
    if ":" in t:
        m, sec = t.split(":")
        return 60 * float(m) + float(sec)
    return float(t)


def fit_scale(stud_pts, target, top, grip, accel, brake):
    """Find meters-per-stud so the simulated lap equals a known lap time (keeps the car numbers)."""
    lo, hi = 0.005, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        t = speed_profile(stud_pts * mid, grip, top, accel, brake)[2].sum()
        lo, hi = (mid, hi) if t < target else (lo, mid)
    return (lo + hi) / 2


def fit_speed(pts, target, top, grip, accel, brake):
    """Keep the real track length; find how much faster than the speedometer the car really goes."""
    lo, hi = 0.3, 8.0
    for _ in range(50):
        k = (lo + hi) / 2
        t = speed_profile(pts, grip * k * k, top * k, accel * k, brake * k)[2].sum()
        lo, hi = (lo, k) if t < target else (k, hi)  # a faster car -> shorter lap
    return (lo + hi) / 2


def fmt(t):
    m = int(t // 60)
    return f"{m}:{t - 60 * m:06.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("track", help="track.json from track_calc.py")
    p.add_argument("--top-speed", type=float, default=324.0, help="km/h")
    p.add_argument("--grip", type=float, default=4.0, help="cornering grip in g")
    p.add_argument("--accel", type=float, default=11.0, help="m/s^2 at low speed")
    p.add_argument("--brake", type=float, default=45.0, help="m/s^2 braking")
    p.add_argument("--meters-per-stud", type=float, default=0.28)
    p.add_argument("--sectors", type=int, default=3)
    p.add_argument("--target-time", help="a known lap time (e.g. 1:35.911): calibrates the calculator to match it")
    p.add_argument("--fit", choices=["speed", "track"], default="speed",
                   help="speed (default): keep the real track length, find how much faster the car really is "
                        "than its speedometer. track: keep the car, shrink the track instead")
    p.add_argument("--speed-factor", type=float, default=1.0,
                   help="real speed / speedometer speed (printed by --target-time; reuse it here)")
    p.add_argument("--csv", help="also save distance / speed / time to this CSV file")
    a = p.parse_args()

    pts, _ = load_track(a.track, a.meters_per_stud)
    top, grip, accel, brake, k = a.top_speed / 3.6, a.grip, a.accel, a.brake, a.speed_factor
    if a.target_time and a.fit == "track":
        stud_pts = pts / a.meters_per_stud
        scale = fit_scale(stud_pts, parse_time(a.target_time), top, grip, accel, brake)
        pts = stud_pts * scale
        km = sum(np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)) / 1000
        print(f"Fitted to {a.target_time}: scale = {scale:.4f} m/stud  ->  effective track length {km:.2f} km")
        print(f"(next time use --meters-per-stud {scale:.4f} instead of --target-time)\n")
    elif a.target_time:
        k = fit_speed(pts, parse_time(a.target_time), top, grip, accel, brake)
        print(f"Fitted to {a.target_time}: the car really goes about {k:.2f}x faster than its speedometer shows")
        print(f"(next time use --speed-factor {k:.3f} instead of --target-time)\n")
    # k > 1: real speeds are k x the speedometer numbers (grip is a squared effect, so it scales with k^2)
    ds, v, dt = speed_profile(pts, grip * k * k, top * k, accel * k, brake * k)
    v = v / k  # show speeds the way the game's speedometer would

    total_m, lap = ds.sum(), dt.sum()
    print(f"Track length:  {total_m / 1000:.2f} km")
    print(f"Lap time:      {fmt(lap)}")
    print(f"Average speed: {total_m / lap * 3.6 / k:.0f} km/h")
    print(f"Top speed:     {v.max() * 3.6:.0f} km/h   |   slowest corner: {v.min() * 3.6:.0f} km/h")

    cum = np.cumsum(dt)
    edges = np.searchsorted(np.cumsum(ds), np.linspace(0, total_m, a.sectors + 1)[1:-1])
    t_prev = 0.0
    for s, e in enumerate(list(edges) + [len(dt) - 1], 1):
        print(f"Sector {s}:      {cum[e] - t_prev:.3f} s")
        t_prev = cum[e]

    if a.csv:
        with open(a.csv, "w") as f:
            f.write("distance_m,speed_kmh,time_s\n")
            for d_, v_, t_ in zip(np.r_[0, np.cumsum(ds)[:-1]], v * 3.6, np.r_[0, cum[:-1]]):
                f.write(f"{d_:.1f},{v_:.1f},{t_:.3f}\n")
        print(f"Wrote {a.csv}")


if __name__ == "__main__":
    main()
