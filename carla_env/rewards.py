import numpy as np
from config import CONFIG

low_speed_timer = 0

min_speed = CONFIG["reward_params"]["min_speed"]
max_speed = CONFIG["reward_params"]["max_speed"]
target_speed = CONFIG["reward_params"]["target_speed"]
max_distance = CONFIG["reward_params"]["max_distance"]
max_std_center_lane = CONFIG["reward_params"]["max_std_center_lane"]
max_angle_center_lane = CONFIG["reward_params"]["max_angle_center_lane"]
penalty_reward = CONFIG["reward_params"]["penalty_reward"]
early_stop = CONFIG["reward_params"]["early_stop"]
reward_functions = {}


def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            # Stop if speed is less than 1.0 km/h after the first 5s of an episode
            global low_speed_timer
            low_speed_timer += 1.0 / env.fps
            speed = env.vehicle.get_speed()
            if low_speed_timer > 5.0 and speed < 1.0 and env.current_waypoint_index >= 1:
                env.terminal_state = True
                terminal_reason = "Vehicle stopped"

            # Stop if distance from center > max distance
            if env.distance_from_center > max_distance:
                env.terminal_state = True
                terminal_reason = "Off-track"

            # Stop if speed is too high
            if max_speed > 0 and speed > max_speed:
                env.terminal_state = True
                terminal_reason = "Too fast"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            low_speed_timer = 0.0
            reward += penalty_reward
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func


# Reward_fn5
def reward_fn5(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than max_angle_center_lane degress off)
               * distance_std_factor (1 when std from center lane is low, 0 when not)
    """

    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    # Final reward
    reward = speed_reward * centering_factor * angle_factor * distance_std_factor

    return reward


reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)


def reward_fn_waypoints(env):
    """
        reward
            - Each time the vehicle overpasses a waypoint, it will receive a reward of 1.0
            - When the vehicle does not pass a waypoint, it receives a reward of 0.0
    """
    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    reward = (env.current_waypoint_index - env.prev_waypoint_index) + speed_reward * centering_factor
    return reward


reward_functions["reward_fn_waypoints"] = create_reward_fn(reward_fn_waypoints)

# ================================================================================
W_PROGRESS = 1.0
W_COMFORT = 0.5
W_AREA = 1.0

COMFORT_ACCEL_MAX = 5.0                # m/s^2
COMFORT_JERK_MAX = 5.0                 # m/s^3
COMFORT_STEERING_MAX = np.deg2rad(40)  # in radians


def reward_fn_av(env):
    """
    Overall reward function for autonomous vehicle training combining:
    
    - Progress Reward: Rewards progress by counting passed waypoints; penalizes if stuck.
    - Comfort Reward: Rewards smooth driving (acceptable acceleration, jerk, and steering).
    - Drive-in-Area Reward: Rewards staying centered in the driving lane.
    
    Each reward is weighted differently.
    """

    # --- Progress Component ---
    # Here we assume the environment tracks waypoint progress via indices.
    progress_delta = env.current_waypoint_index - env.prev_waypoint_index
    if progress_delta > 0:
        progress_reward = progress_delta
    else:
        # Penalize if there is no progress (stuck or not moving forward)
        progress_reward = -0.5

    # --- Comfort Component ---
    acceleration = env.vehicle.get_acceleration_ego() 
    jerk = env.vehicle.get_jerk_ego() 
    steering = env.vehicle.control.steer 
    
    # Compute factors (in [0,1]) that are 1 when the value is perfectly comfortable and decrease if too high.
    accel_factor = max(1.0 - abs(acceleration) / COMFORT_ACCEL_MAX, 0.0)
    jerk_factor = max(1.0 - abs(jerk) / COMFORT_JERK_MAX, 0.0)
    steering_factor = max(1.0 - abs(steering) / COMFORT_STEERING_MAX, 0.0)
    comfort_reward = accel_factor * jerk_factor * steering_factor

    # --- Drive-in-Area Component ---
    # Use a centering factor based on the vehicle's distance from the lane center.
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    angle = env.vehicle.get_angle(env.current_waypoint)
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)
    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    drive_area_reward = centering_factor * angle_factor * distance_std_factor

    # --- Total Reward ---
    total_reward = (W_PROGRESS * progress_reward +
                    W_COMFORT * comfort_reward +
                    W_AREA * drive_area_reward)
                    
    return total_reward

reward_functions["av_reward"] = create_reward_fn(reward_fn_av)