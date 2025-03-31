import carla
import numpy as np
import cv2
import math
from collections import defaultdict
import time

FULL_MAP_RES = 0.25
FULL_MAP_SIZE = 2048  # Full static map size (2048x2048)
FULL_MAP_CENTER = carla.Location(x=0, y=0, z=0)


def generate_static_bev_map(map_waypoints, res=FULL_MAP_RES, full_size=FULL_MAP_SIZE, center=FULL_MAP_CENTER):
    import numpy as np
    import cv2
    import math
    from collections import defaultdict

    IMG_W = IMG_H = full_size
    bev_base = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    def world_to_bev(loc):
        dx = loc.x - center.x
        dy = loc.y - center.y
        px = int(dx / res + IMG_W / 2)
        py = int(IMG_H / 2 - dy / res)
        return px, py

    lane_lines = defaultdict(lambda: {"left": [], "right": [], "center": [], "world_left": [], "world_right": []})
    for wp in map_waypoints:
        road_id = (wp.road_id, wp.lane_id, wp.lane_type)
        lane_width = wp.lane_width / 2.0
        yaw = math.radians(wp.transform.rotation.yaw)
        center_loc = wp.transform.location

        left = carla.Location(
            x=center_loc.x + lane_width * math.cos(yaw + math.pi / 2),
            y=center_loc.y + lane_width * math.sin(yaw + math.pi / 2))
        right = carla.Location(
            x=center_loc.x + lane_width * math.cos(yaw - math.pi / 2),
            y=center_loc.y + lane_width * math.sin(yaw - math.pi / 2))

        lp = world_to_bev(left)
        rp = world_to_bev(right)
        cp = world_to_bev(center_loc)

        lane_lines[road_id]["left"].append(lp)
        lane_lines[road_id]["right"].append(rp)
        lane_lines[road_id]["center"].append(cp)
        lane_lines[road_id]["world_left"].append(left)
        lane_lines[road_id]["world_right"].append(right)

    for segment in lane_lines.values():
        left_pts = segment["world_left"]
        right_pts = segment["world_right"]
        if len(left_pts) >= 2 and len(right_pts) >= 2:
            for i in range(len(left_pts) - 1):
                p1 = world_to_bev(left_pts[i])
                p2 = world_to_bev(left_pts[i + 1])
                p3 = world_to_bev(right_pts[i + 1])
                p4 = world_to_bev(right_pts[i])
                quad = np.array([p1, p2, p3, p4], dtype=np.int32)
                cv2.fillPoly(bev_base, [quad], (60, 60, 60))

    for segment in lane_lines.values():
        for side in ["left", "right"]:
            pts = segment[side]
            if len(pts) > 1:
                cv2.polylines(bev_base, [np.array(pts, dtype=np.int32)], isClosed=False, color=(255, 255, 255), thickness=1)
        centers = segment["center"]
        if len(centers) > 1:
            for i in range(0, len(centers) - 1, 4):
                cv2.line(bev_base, centers[i], centers[i + 1], (255, 255, 255), 1, lineType=cv2.LINE_4)

    road_groups = defaultdict(list)
    for key, segment in lane_lines.items():
        road_groups[key[0]].append(segment["center"])
    for centerlines in road_groups.values():
        if len(centerlines) >= 2:
            merged = [np.mean(pts, axis=0).astype(int) for pts in zip(*centerlines) if len(pts) > 0]
            if len(merged) > 1:
                cv2.polylines(bev_base, [np.array(merged)], False, (0, 255, 255), thickness=1)
    cv2.imwrite("bev_base.png", bev_base)
    return bev_base

def generate_dynamic_bev_layer(world, ego_vehicle, full_map_image, route_wps=None, res=FULL_MAP_RES, crop_size=224, center=FULL_MAP_CENTER):
    import numpy as np
    import cv2
    import math
    import time


    ego_transform = ego_vehicle.get_transform()
    ego_id = ego_vehicle.id
    IMG_H, IMG_W = full_map_image.shape[:2]
    crop_px = int(crop_size)

    def world_to_bev(loc):
        dx = loc.x - center.x
        dy = loc.y - center.y
        px = int(dx / res + IMG_W / 2)
        py = int(IMG_H / 2 - dy / res)
        return px, py

    ego_px, ego_py = world_to_bev(ego_transform.location)
    half = crop_px // 2
    x1, y1 = ego_px - half, ego_py - half
    x2, y2 = ego_px + half, ego_py + half
    bev_image = full_map_image[max(0, y1):min(IMG_H, y2), max(0, x1):min(IMG_W, x2)].copy()

    offset_x = max(0, x1)
    offset_y = max(0, y1)

    def relative_px(loc):
        px, py = world_to_bev(loc)
        return px - offset_x, py - offset_y

    carla_map = world.get_map()
    if route_wps is None:
        wp = carla_map.get_waypoint(ego_transform.location)
        route_wps = [wp]
        for _ in range(15):
            wp = route_wps[-1].next(3.0)[0]
            route_wps.append(wp)

    route_pts = [relative_px(wp.transform.location) for wp in route_wps]
    if len(route_pts) > 1:
        wp_thickness = int(2.0 / res)
        cv2.polylines(bev_image, [np.array(route_pts, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=wp_thickness)

    def draw_vehicle_box(bev_img, tf, color):
        length, width = 4.5, 2.0
        corners = np.array([
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
            [length / 2, width / 2],
            [length / 2, -width / 2]
        ])
        yaw = math.radians(tf.rotation.yaw)
        R = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw)]
        ])
        rotated = (R @ corners.T).T
        pts = []
        for dx, dy in rotated:
            world_x = tf.location.x + dx
            world_y = tf.location.y + dy
            px, py = relative_px(carla.Location(x=world_x, y=world_y))
            pts.append((px, py))
        cv2.fillPoly(bev_img, [np.array(pts, dtype=np.int32)], color)

        heading_len = 4.0
        heading_wid = 1.5
        cx = tf.location.x + heading_len * math.cos(math.radians(tf.rotation.yaw))
        cy = tf.location.y + heading_len * math.sin(math.radians(tf.rotation.yaw))
        l1 = tf.location.x + heading_wid * math.cos(math.radians(tf.rotation.yaw + 150)), tf.location.y + heading_wid * math.sin(math.radians(tf.rotation.yaw + 150))
        l2 = tf.location.x + heading_wid * math.cos(math.radians(tf.rotation.yaw - 150)), tf.location.y + heading_wid * math.sin(math.radians(tf.rotation.yaw - 150))
        tip = relative_px(carla.Location(x=cx, y=cy))
        rl = relative_px(carla.Location(x=l1[0], y=l1[1]))
        rr = relative_px(carla.Location(x=l2[0], y=l2[1]))
        triangle = np.array([tip, rl, rr], dtype=np.int32)
        cv2.polylines(bev_img, [triangle], isClosed=True, color=(0, 0, 0), thickness=2)

    vehicles = world.get_actors().filter('vehicle.*')
    for v in vehicles:
        color = (0, 255, 0)
        if v.id == ego_id:
            color = (0, 0, 255)
        draw_vehicle_box(bev_image, v.get_transform(), color)

    rotation_angle = 90 - ego_transform.rotation.yaw
    M = cv2.getRotationMatrix2D((crop_px / 2, crop_px / 2), rotation_angle, 1.0)
    bev_image = cv2.warpAffine(bev_image, M, (crop_px, crop_px), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    bev_image = cv2.flip(bev_image, 1)
    # cv2.imshow("bev", bev_image)
    # convert to RGB
    bev_image = bev_image[..., ::-1].copy()
    return bev_image.transpose(2, 0, 1)
