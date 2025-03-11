import os
import gym
import numpy as np
import carla
import random
import math
import time
from gym import spaces
import cv2
import pygame
from pygame.locals import *
from model.agent_state import *

def smooth_action(current, target, smoothing_factor):
    return current * smoothing_factor + target * (1 - smoothing_factor)

def transform_to_ego_frame(ego_transform, target_location):
    dx = target_location.x - ego_transform.location.x
    dy = target_location.y - ego_transform.location.y
    yaw = math.radians(ego_transform.rotation.yaw)
    rel_x = dx * math.cos(-yaw) - dy * math.sin(-yaw)
    rel_y = dx * math.sin(-yaw) + dy * math.cos(-yaw)
    return rel_x, rel_y

# Dictionary for discrete actions mapping indices to [steer, throttle]
discrete_actions = {
    0: [-1.0, 1.0],
    1: [0.0, 1.0],
    2: [1.0, 1.0],
    3: [0.0, 0.0],  # braking / stopping
}

class CarlaDrivingEnv(gym.Env):
    """
    Gym environment for autonomous driving in CARLA designed for DQ-GAT.
    
    Observations:
      A dict with two keys:
        - "bev_image": A (3, 224, 224) float32 image (CHW) representing the bird's-eye view,
                       obtained from a semantic segmentation camera.
        - "agent_feats": A (20, 10) float32 array of agent features in the ego-vehicle's coordinate frame.
    
    Actions:
      - Continuous: A 2-element array [steer, throttle] with steer in [-1, 1] and throttle in [0, 1].
      - Discrete: An integer index mapping to [steer, throttle] using the discrete_actions dict.
    
    Reward:
      - -50 if a collision occurs.
      - Otherwise, reward = current speed (km/h)/40, capped at 1.
    
    Additional features:
      - Customizable reward, state encoding, and decoding hooks.
      - Supports both training and evaluation modes and various rendering modes.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 host="127.0.0.1",
                 port=2000,
                 fps=15,
                 map='Town07',
                 action_space_type="discrete",  # "continuous" or "discrete"
                 reward_fn=None,
                 encode_state_fn=None,
                 decode_vae_fn=None,
                 action_smoothing=0.2,
                 eval=False,
                 activate_render=True):
        super(CarlaDrivingEnv, self).__init__()

        # Connect to CARLA and set synchronous mode.
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / fps
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.fps = fps
        self.map = map
        self.action_space_type = action_space_type
        self.action_smoothing = action_smoothing
        self.eval = eval
        self.activate_render = activate_render

        # Custom hooks.
        self.reward_fn = reward_fn if callable(reward_fn) else self.default_reward_fn
        self.encode_state_fn = encode_state_fn if callable(encode_state_fn) else (lambda obs: obs)
        self.decode_vae_fn = decode_vae_fn if callable(decode_vae_fn) else None

        # Define action space.
        if self.action_space_type == "continuous":
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]),
                                           high=np.array([1.0, 1.0]),
                                           dtype=np.float32)
        elif self.action_space_type == "discrete":
            self.action_space = spaces.Discrete(len(discrete_actions))
        else:
            raise ValueError("Unsupported action_space_type. Use 'continuous' or 'discrete'.")

        # Define observation space.
        # BEV image: now directly (3, 224, 224) from the segmentation sensor.
        # self.observation_space = spaces.Dict({
        #     "bev_image": spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.float32),
        #     "agent_feats": spaces.Box(low=-np.inf, high=np.inf, shape=(20, 10), dtype=np.float32)
        # })
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32)

        # Initialize actors.
        self.vehicle = None
        self.collision_sensor = None
        self.bev_camera = None
        self.bev_image = None
        self.collision_history = []
        self.step_count = 0
        self.max_episode_steps = 1000
        self.episode_ended = False

        # Pygame display for rendering.
        if self.activate_render:
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((224, 224), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()

        # Spawn ego vehicle.
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        # Attach collision sensor.
        collision_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Attach semantic segmentation BEV camera.
        # We directly set the image size to 224x224.
        camera_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        camera_bp.set_attribute("image_size_x", "224")
        camera_bp.set_attribute("image_size_y", "224")
        camera_bp.set_attribute("fov", "90")
        # Optionally, you could set a custom palette:
        # camera_bp.set_attribute("custom_palette", "True")
        camera_transform = carla.Transform(carla.Location(x=0, y=0, z=50),
                                           carla.Rotation(pitch=-90))
        self.bev_camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.bev_image = None
        self.bev_camera.listen(lambda image: self._on_bev_image(image))

    def default_reward_fn(self, env):
        # Default: -50 if collision; else speed (km/h)/40, capped at 1.
        speed = self.vehicle.get_speed() * 3.6
        if len(self.collision_history) > 0:
            return -50.0
        return min(speed / 40.0, 1.0)

    def _on_collision(self, event):
        self.collision_history.append(event)
        self.episode_ended = True

    def _on_bev_image(self, image):
        # For semantic segmentation sensor, the image is already 224x224.
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.bev_image = array[:, :, :3]  # drop alpha

    def _process_bev_image(self, image):
        """
        Since the sensor directly outputs a 224x224 image, we only need to convert
        from HWC to CHW and cast to float32.
        """
        chw = np.transpose(image, (2, 0, 1))
        return chw.astype(np.float32)

    def get_agent_features(self):
        """
        Returns an array (shape: (20, 12)) of agent features.
        Each row is a NpSequenceArray with:
        [x, y, d, psi, v_x, v_y, a_x, a_y, w, l, class_type, token_type]
        Coordinates and kinematics are expressed in the ego-vehicle's frame.
        Pads with zeros if fewer than 20 agents are detected.
        For the ego-vehicle, token_type is set to EGO_TOKEN;
        for others, it's set to AGENT_TOKEN.
        """
        ego_transform = self.vehicle.get_transform()
        ego_velocity = self.vehicle.get_velocity()
        vehicles = self.world.get_actors().filter("vehicle.*")
        feature_list = []
        
        for veh in vehicles:
            # Use ego data if vehicle is the ego.
            if veh.id == self.vehicle.id:
                veh_transform = ego_transform
                veh_velocity = ego_velocity
                veh_accel = self.vehicle.get_acceleration()
                token = TokenType.EGO_TOKEN.value
            else:
                veh_transform = veh.get_transform()
                veh_velocity = veh.get_velocity()
                veh_accel = veh.get_acceleration()
                token = TokenType.AGENT_TOKEN.value

            # Compute relative position.
            rel_x, rel_y = transform_to_ego_frame(ego_transform, veh_transform.location)
            d = math.sqrt(rel_x**2 + rel_y**2)
            # Relative yaw.
            psi = veh_transform.rotation.yaw - ego_transform.rotation.yaw
            # Transform velocities to ego frame.
            vx_world = veh_velocity.x
            vy_world = veh_velocity.y
            yaw_rad = math.radians(ego_transform.rotation.yaw)
            rel_vx = vx_world * math.cos(-yaw_rad) - vy_world * math.sin(-yaw_rad)
            rel_vy = vx_world * math.sin(-yaw_rad) + vy_world * math.cos(-yaw_rad)
            # Transform acceleration similarly.
            ax_world = veh_accel.x
            ay_world = veh_accel.y
            rel_ax = ax_world * math.cos(-yaw_rad) - ay_world * math.sin(-yaw_rad)
            rel_ay = ax_world * math.sin(-yaw_rad) + ay_world * math.cos(-yaw_rad)
            # Get vehicle dimensions (CARLA gives half extents).
            bb = veh.bounding_box
            w = bb.extent.y * 2
            l = bb.extent.x * 2
            
            # Create a zero array and fill the fields.
            arr = np.zeros((NpSequence_DIM,), dtype=np.float32)
            arr[X_DIM] = rel_x
            arr[Y_DIM] = rel_y
            arr[DIST_DIM] = d
            arr[YAW_DIM] = psi
            arr[VX_DIM] = rel_vx
            arr[VY_DIM] = rel_vy
            arr[AX_DIM] = rel_ax
            arr[AY_DIM] = rel_ay
            arr[WIDTH_DIM] = w
            arr[LENGTH_DIM] = l
            arr[CLASS_TYPE_DIM] = ClassType.VEHICLE.value  # Change if needed.
            arr[TOKEN_TYPE_DIM] = token
            # Convert to NpSequenceArray (if desired).
            feature = NpSequenceArray(arr)
            feature_list.append(feature)
        
        # Convert list to numpy array.
        features = np.array(feature_list, dtype=np.float32)
        max_agents = 20
        num_agents = features.shape[0]
        if num_agents < max_agents:
            pad = np.zeros((max_agents - num_agents, NpSequence_DIM), dtype=np.float32)
            features = np.vstack([features, pad])
        elif num_agents > max_agents:
            # Choose the closest agents based on distance.
            distances = features[:, DIST_DIM]
            idx = np.argsort(distances)[:max_agents]
            features = features[idx]
        return features


    def step(self, action):
        """
        Executes one simulation step.
        For continuous actions, action is [steer, throttle].
        For discrete actions, action is an index mapping to [steer, throttle] in discrete_actions.
        Action smoothing is applied.
        """
        if self.episode_ended:
            return self.reset()

        if self.action_space_type == "continuous":
            target_steer, target_throttle = action
        elif self.action_space_type == "discrete":
            target_steer, target_throttle = discrete_actions[action]
        else:
            raise ValueError("Unsupported action_space_type.")

        current_control = self.vehicle.get_control()
        new_steer = smooth_action(current_control.steer, target_steer, self.action_smoothing)
        new_throttle = smooth_action(current_control.throttle, target_throttle, self.action_smoothing)
        new_brake = 0.0
        if target_throttle == 0 and self.vehicle.get_speed() * 3.6 > 5:
            new_brake = 0.5

        control = carla.VehicleControl()
        control.steer = new_steer
        control.throttle = new_throttle
        control.brake = new_brake
        self.vehicle.apply_control(control)

        self.world.tick()
        self.step_count += 1

        timeout = time.time() + 1.0
        while self.bev_image is None and time.time() < timeout:
            time.sleep(0.01)
        raw_bev = self.bev_image if self.bev_image is not None else np.zeros((224, 224, 3), dtype=np.uint8)
        bev_obs = self._process_bev_image(raw_bev)
        agent_feats = self.get_agent_features()
        observation = {"bev_image": bev_obs, "agent_feats": agent_feats}

        reward = self.reward_fn(self)
        done = self.episode_ended or (self.step_count >= self.max_episode_steps)
        info = {"step": self.step_count, "speed": self.vehicle.get_speed() * 3.6}

        if self.activate_render:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    self.close()
                    done = True
            self.render(mode="human")
            self.clock.tick(self.fps)

        encoded_obs = self.encode_state_fn(observation)
        return encoded_obs, reward, done, info

    def reset(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.bev_camera is not None:
            self.bev_camera.destroy()

        self.collision_history = []
        self.episode_ended = False
        self.step_count = 0

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        collision_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        camera_bp = self.world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        camera_bp.set_attribute("image_size_x", "224")
        camera_bp.set_attribute("image_size_y", "224")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(carla.Location(x=0, y=0, z=50), carla.Rotation(pitch=-90))
        self.bev_camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.bev_image = None
        self.bev_camera.listen(lambda image: self._on_bev_image(image))

        for _ in range(3):
            self.world.tick()
            time.sleep(0.1)

        raw_bev = self.bev_image if self.bev_image is not None else np.zeros((224, 224, 3), dtype=np.uint8)
        bev_obs = self._process_bev_image(raw_bev)
        agent_feats = self.get_agent_features()
        observation = {"bev_image": bev_obs, "agent_feats": agent_feats}
        return self.encode_state_fn(observation)

    def render(self, mode="human"):
        if mode == "rgb_array":
            if self.bev_image is not None:
                img = self._process_bev_image(self.bev_image)
                img = np.transpose(img, (1, 2, 0))
                return img
            else:
                return np.zeros((224, 224, 3), dtype=np.uint8)
        elif mode == "human":
            if self.bev_image is not None:
                img = self._process_bev_image(self.bev_image)
                img = np.transpose(img, (1, 2, 0))
                cv2.imshow("Bird's-Eye View", img)
                cv2.waitKey(1)

    def close(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.bev_camera is not None:
            self.bev_camera.destroy()
        pygame.quit()
        self.world.apply_settings(carla.WorldSettings())

if __name__ == "__main__":
    env = CarlaDrivingEnv(action_space_type="discrete", activate_render=True)
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print("Total Reward:", total_reward)
    env.close()
