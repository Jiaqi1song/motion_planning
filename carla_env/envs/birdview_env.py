#!/usr/bin/env python3
"""
Draw every spawn point in the map and (optionally) the driving route
between two spawn indices as a red poly-line.  All primitives stay
on screen for 60 s so you have time to look around.

Run:
    python carla_birdview.py --map Town05 --route 97 92
"""

import argparse
import math
import os
import subprocess
import sys
import time
from typing import List, Optional

import carla


# ──────────────────────────────────────────────────────────────────────────────
# Helper: launch CARLA locally (Linux / macOS).  Comment out if not needed.
# ──────────────────────────────────────────────────────────────────────────────
# def _launch_carla(fps: int) -> subprocess.Popen:
#     carla_root = os.environ.get("CARLA_ROOT")
#     if not carla_root:
#         print("ERROR: Set $CARLA_ROOT or launch CARLA manually.", file=sys.stderr)
#         sys.exit(1)

#     exe = os.path.join(carla_root, "CarlaUE4.sh")
#     cmd = [
#         exe,
#         "-quality_level=Low",
#         "-benchmark",
#         f"-fps={fps}",
#         "-prefernvidia",
#     ]
#     print("Starting CARLA:\n ", " ".join(cmd))
#     return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# ──────────────────────────────────────────────────────────────────────────────
# Main bird-view drawer
# ──────────────────────────────────────────────────────────────────────────────
class BirdViewDrawer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        town: str = "Town05",
        fps: int = 20,
        carla_proc: Optional[subprocess.Popen] = None,
    ):
        self.carla_proc = carla_proc

        # Connect
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)

        # Load map (only if the running server is on another map)
        if self.client.get_world().get_map().name != town:
            self.client.load_world(town)
        self.world = self.client.get_world()

        # Put the server into synchronous mode
        settings = self.world.get_settings()
        if not settings.synchronous_mode or settings.fixed_delta_seconds == 0:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / fps
            self.world.apply_settings(settings)

        # Move the spectator straight above spawn point 0
        self._set_birdview()

        # Convenience handle
        self.debug = self.world.debug
        print(f"Connected to {town} | {len(self.world.get_map().get_spawn_points())} spawn points")

    # ------------------------------------------------------------------ spectator
    def _set_birdview(self):
        sp0 = self.world.get_map().get_spawn_points()[0]
        spectator = self.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                sp0.location + carla.Location(z=60),
                carla.Rotation(pitch=-90, yaw=0, roll=0),
            )
        )

    # ------------------------------------------------------------------ primitives
    def draw_all_spawns(self, life=60.0):
        for idx, sp in enumerate(self.world.get_map().get_spawn_points()):
            loc = sp.location + carla.Location(z=1.25)
            yaw_rad = math.radians(sp.rotation.yaw)
            arrow_end = loc + carla.Location(x=math.cos(yaw_rad), y=math.sin(yaw_rad))

            # arrow
            self.debug.draw_arrow(
                loc,
                arrow_end,
                thickness=0.4,
                arrow_size=2.0,
                color=carla.Color(255, 0, 0),
                life_time=life,
                persistent_lines=True,
            )
            # label
            offset = carla.Location(x=-math.sin(yaw_rad) * 3, y=math.cos(yaw_rad) * 3, z=2)
            self.debug.draw_string(
                sp.location + offset,
                str(idx),
                draw_shadow=False,
                color=carla.Color(0, 255, 255),
                life_time=life,
            )

    def draw_route(self, start_idx: int, end_idx: int, life=60.0, skip: int = 2):
        spawn_pts = self.world.get_map().get_spawn_points()
        start_wp = self.world.get_map().get_waypoint(spawn_pts[start_idx].location)
        end_wp   = self.world.get_map().get_waypoint(spawn_pts[end_idx].location)

        route = self._compute_route(start_wp, end_wp)
        z = 1.25
        for i in range(0, len(route) - 1, skip + 1):
            w0, w1 = route[i][0], route[i + 1][0]

            self.debug.draw_line(
                w0.transform.location + carla.Location(z=z),
                w1.transform.location + carla.Location(z=z),
                thickness=0.15,
                color=carla.Color(255, 0, 0),
                life_time=life,
                persistent_lines=True,
            )

            self.debug.draw_point(
                w0.transform.location + carla.Location(z=z),
                0.2,
                carla.Color(0, 255 if i == 0 else 0, 0 if i == 0 else 255),
                life,
                False,
            )

        # final destination dot
        self.debug.draw_point(
            route[-1][0].transform.location + carla.Location(z=z),
            0.2,
            carla.Color(0, 0, 255),
            life,
            False,
        )


    # helper ----------------------------------------------------------
    def _compute_route(self, start_wp, end_wp, resolution=1.0):
        """
        Replacement for carla_env.navigation.planner.compute_route_waypoints
        in case that module is not available.
        """
        route = [(start_wp, 0)]

        current = start_wp
        while current.transform.location.distance(end_wp.transform.location) > resolution:
            next_wps = current.next(resolution)
            current = min(next_wps, key=lambda w: w.transform.location.distance(end_wp.transform.location))
            route.append((current, 0))
        route.append((end_wp, 0))
        return route

    # ------------------------------------------------------------------
    def tick(self):
        self.world.tick()

    def close(self):
        if self.carla_proc is not None:
            self.carla_proc.kill()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--fps", default=20, type=int)
    parser.add_argument("--map", default="Town05")
    parser.add_argument("--route", nargs=2, type=int, metavar=("START", "END"),
                        help="indices of spawn points to connect with a poly-line")
    parser.add_argument("--start-carla", action="store_true",
                        help="launch CARLA from $CARLA_ROOT automatically")
    args = parser.parse_args(argv)

    carla_proc = _launch_carla(args.fps) if args.start_carla else None
    if carla_proc is not None:
        time.sleep(6)  # give UE4 a moment to boot

    drawer = BirdViewDrawer(
        host=args.host,
        port=args.port,
        town=args.map,
        fps=args.fps,
        carla_proc=carla_proc,
    )

    try:
        drawer.draw_all_spawns()
        if args.route:
            drawer.draw_route(args.route[0], args.route[1])
        drawer.tick()  # flush the draw queue
        print("Primitives sent to server.  Keep the script alive to keep them on-screen.")
        time.sleep(60)  # keep alive long enough to inspect
    finally:
        drawer.close()


if __name__ == "__main__":
    main()
