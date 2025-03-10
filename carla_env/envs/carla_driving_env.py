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

# Helper: Smooth control signals.
def smooth_action(current, target, smoothing_factor):
    return current * (1 - smoothing_factor) + target * smoothing_factor

def transform_to_ego_frame(ego_transform, target_location):
    # Compute relative position in ego frame.
    dx = target_location.x - ego_transform.location.x
    dy = target_location.y - ego_transform.location.y
    yaw = math.radians(ego_transform.rotation.yaw)
    rel_x = dx * math.cos(-yaw) - dy * math.sin(-yaw)
    rel_y = dx * math.sin(-yaw) + dy * math.cos(-yaw)
    return rel_x, rel_y

# Dictionary mapping discrete actions to [steer, throttle] pairs.
discrete_actions = {
    0: [-1.0, 1.0],
    1: [0.0, 1.0],
    2: [1.0, 1.0],
    3: [0.0, 0.0],  # e.g., braking (stop)
}

class CarlaDrivingEnv(gym.Env):
    """
    Gym environment for autonomous driving in CARLA designed for DQ-GAT.
    
    Observations:
      A dict with two keys:
        - "bev_image": A (3, 224, 224) float32 image (CHW) representing the bird's-eye view.
        - "agent_feats": A (20, 10) float32 array of agent features.
          Each row is [x, y, d, psi, v_x, v_y, a_x, a_y, w, l] in the ego-vehicle frame.
          (Up to 20 agents; padded with zeros if fewer are detected.)
    
    Actions:
      If action_space_type == "continuous":
        - Action is a Box([steer, throttle]) with values in [-1,1] for steer and [0,1] for throttle.
      If action_space_type == "discrete":
        - Action is an integer index mapping to [steer, throttle] as defined in discrete_actions.
    
    Reward (default):
      - -50 if a collision occurs.
      - Otherwise, reward = current speed (km/h) / 40 (capped at 1).
    
    Additional features:
      - Customizable reward function (reward_fn), state encoding (encode_state_fn), and decoding (decode_vae_fn).
      - Supports both training and evaluation modes.
      - Supports multiple rendering modes.
      - Action smoothing is applied to the control signals.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 host="127.0.0.1",
                 port=2000,
                 fps=15,
                 map_name='Town07',
                 action_space_type="discrete",  # "continuous" or "discrete"
                 reward_fn=None,
                 encode_state_fn=None,
                 decode_vae_fn=None,
                 action_smoothing=0.2,
                 eval=False,
                 activate_render=True):
        super(CarlaDrivingEnv, self).__init__()

        # Connect to CARLA and configure the world.
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / fps
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        
        # Save parameters.
        self.fps = fps
        self.map_name = map_name
        self.action_space_type = action_space_type
        self.action_smoothing = action_smoothing
        self.eval = eval
        self.activate_render = activate_render

        # Setup custom hooks.
        self.reward_fn = reward_fn if callable(reward_fn) else self.default_reward_fn
        self.encode_state_fn = encode_state_fn if callable(encode_state_fn) else (lambda obs: obs)
        self.decode_vae_fn = decode_vae_fn if callable(decode_vae_fn) else None

        # Define action space.
        if self.action_space_type == "continuous":
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        elif self.action_space_type == "discrete":
            self.action_space = spaces.Discrete(len(discrete_actions))
        else:
            raise ValueError("Unsupported action_space_type. Use 'continuous' or 'discrete'.")

        # Define observation space.
        # BEV image: (3, 224, 224) float32; Agent features: (20, 10) float32.
        self.observation_space = spaces.Dict({
            "bev_image": spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.float32),
            "agent_feats": spaces.Box(low=-np.inf, high=np.inf, shape=(20, 10), dtype=np.float32)
        })

        # Initialize actors.
        self.vehicle = None
        self.collision_sensor = None
        self.bev_camera = None
        self.bev_image = None
        self.collision_history = []
        self.step_count = 0
        self.max_episode_steps = 1000
        self.episode_ended = False

        # For rendering with pygame.
        if self.activate_render:
            pygame.init()
            pygame.font.init()
            # Set display resolution; can be customized.
            self.display = pygame.display.set_mode((224, 224), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()

        # Spawn the ego vehicle.
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.lincoln.mkz2017")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        # Attach collision sensor.
        collision_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Attach a BEV camera.
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "160")
        camera_bp.set_attribute("image_size_y", "80")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(carla.Location(x=0, y=0, z=50), carla.Rotation(pitch=-90))
        self.bev_camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.bev_image = None
        self.bev_camera.listen(lambda image: self._on_bev_image(image))
        
    def default_reward_fn(self, env):
        # Default reward: -50 if collision, else speed (km/h)/40 capped at 1.
        speed = self.vehicle.get_speed() * 3.6  # convert m/s to km/h
        if len(self.collision_history) > 0:
            return -50.0
        return min(speed / 40.0, 1.0)
    
    def _on_collision(self, event):
        self.collision_history.append(event)
        self.episode_ended = True
    
    def _on_bev_image(self, image):
        # Convert CARLA raw image to a NumPy array.
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.bev_image = array[:, :, :3]  # drop alpha
    
    def _process_bev_image(self, image):
        """
        Resize the raw BEV image (80x160) to (224x224),
        convert from HWC to CHW, and cast to float32.
        """
        resized = cv2.resize(image, (224, 224))
        chw = np.transpose(resized, (2, 0, 1))
        return chw.astype(np.float32)
    
    def get_agent_features(self):
        """
        Returns an array (shape: (20, 10)) of agent features.
        Each row is: [x, y, d, psi, v_x, v_y, a_x, a_y, w, l] in the ego frame.
        Pads with zeros if fewer than 20 agents are detected.
        """
        ego_transform = self.vehicle.get_transform()
        ego_velocity = self.vehicle.get_velocity()
        vehicles = self.world.get_actors().filter("vehicle.*")
        features = []
        for veh in vehicles:
            if veh.id == self.vehicle.id:
                veh_transform = ego_transform
                veh_velocity = ego_velocity
                veh_accel = self.vehicle.get_acceleration()
            else:
                veh_transform = veh.get_transform()
                veh_velocity = veh.get_velocity()
                veh_accel = veh.get_acceleration()
            rel_x, rel_y = transform_to_ego_frame(ego_transform, veh_transform.location)
            d = math.sqrt(rel_x ** 2 + rel_y ** 2)
            psi = veh_transform.rotation.yaw - ego_transform.rotation.yaw
            vx_world = veh_velocity.x
            vy_world = veh_velocity.y
            yaw_rad = math.radians(ego_transform.rotation.yaw)
            rel_vx = vx_world * math.cos(-yaw_rad) - vy_world * math.sin(-yaw_rad)
            rel_vy = vx_world * math.sin(-yaw_rad) + vy_world * math.cos(-yaw_rad)
            ax_world = veh_accel.x
            ay_world = veh_accel.y
            rel_ax = ax_world * math.cos(-yaw_rad) - ay_world * math.sin(-yaw_rad)
            rel_ay = ax_world * math.sin(-yaw_rad) + ay_world * math.cos(-yaw_rad)
            bb = veh.bounding_box
            w = bb.extent.y * 2
            l = bb.extent.x * 2
            feat = [rel_x, rel_y, d, psi, rel_vx, rel_vy, rel_ax, rel_ay, w, l]
            features.append(feat)
        features = np.array(features, dtype=np.float32)
        max_agents = 20
        if features.shape[0] < max_agents:
            pad = np.zeros((max_agents - features.shape[0], 10), dtype=np.float32)
            features = np.vstack([features, pad])
        elif features.shape[0] > max_agents:
            distances = features[:, 2]
            idx = np.argsort(distances)[:max_agents]
            features = features[idx]
        return features
    
    def step(self, action):
        """
        Executes one simulation step.
        Depending on action_space_type:
          - For "continuous": action is [steer, throttle] in respective ranges.
          - For "discrete": action is an index into discrete_actions.
        Applies action smoothing.
        """
        if self.episode_ended:
            return self.reset()
        
        # Determine target control signals.
        if self.action_space_type == "continuous":
            # action is assumed to be a 2-element array: [steer, throttle]
            target_steer, target_throttle = action
        elif self.action_space_type == "discrete":
            target_steer, target_throttle = discrete_actions[action]
        else:
            raise ValueError("Unsupported action_space_type.")
        
        # Get current control and apply smoothing.
        current_control = self.vehicle.get_control()
        new_steer = smooth_action(current_control.steer, target_steer, self.action_smoothing)
        new_throttle = smooth_action(current_control.throttle, target_throttle, self.action_smoothing)
        # For braking, if throttle is low and target throttle is zero, we apply a brake.
        new_brake = 0.0
        if target_throttle == 0 and self.vehicle.get_speed() * 3.6 > 5:
            new_brake = 0.5
        
        control = carla.VehicleControl()
        control.steer = new_steer
        control.throttle = new_throttle
        control.brake = new_brake
        self.vehicle.apply_control(control)
        
        # Advance simulation.
        self.world.tick()
        self.step_count += 1
        
        # Wait for BEV image update.
        timeout = time.time() + 1.0
        while self.bev_image is None and time.time() < timeout:
            time.sleep(0.01)
        raw_bev = self.bev_image if self.bev_image is not None else np.zeros((80, 160, 3), dtype=np.uint8)
        bev_obs = self._process_bev_image(raw_bev)
        agent_feats = self.get_agent_features()
        observation = {"bev_image": bev_obs, "agent_feats": agent_feats}
        
        # Calculate reward using the custom reward function.
        reward = self.reward_fn(self)
        
        done = self.episode_ended or (self.step_count >= self.max_episode_steps)
        info = {"step": self.step_count, "speed": self.vehicle.get_speed() * 3.6}
        
        # Render if enabled.
        if self.activate_render:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    self.close()
                    done = True
            self.render(mode="human")
            self.clock.tick(self.fps)
        
        # Optionally encode the observation.
        encoded_obs = self.encode_state_fn(observation)
        return encoded_obs, reward, done, info
    
    def reset(self):
        # Clean up existing actors.
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
        vehicle_bp = blueprint_library.filter("vehicle.lincoln.mkz2017")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
        
        # Attach collision sensor.
        collision_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        # Attach BEV camera.
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "160")
        camera_bp.set_attribute("image_size_y", "80")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(carla.Location(x=0, y=0, z=50), carla.Rotation(pitch=-90))
        self.bev_camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.bev_image = None
        self.bev_camera.listen(lambda image: self._on_bev_image(image))
        
        # Tick the world a few times to gather sensor data.
        for _ in range(3):
            self.world.tick()
            time.sleep(0.1)
        
        raw_bev = self.bev_image if self.bev_image is not None else np.zeros((80, 160, 3), dtype=np.uint8)
        bev_obs = self._process_bev_image(raw_bev)
        agent_feats = self.get_agent_features()
        observation = {"bev_image": bev_obs, "agent_feats": agent_feats}
        return observation
    
    def render(self, mode="human"):
        # For "rgb_array", return the BEV image (converted back to HWC).
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
    # Test loop.
    env = CarlaDrivingEnv(action_space_type="discrete", activate_render=True)
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()  # random action
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print("Total Reward:", total_reward)
    env.close()
