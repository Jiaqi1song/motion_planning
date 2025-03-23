from __future__ import annotations

from enum import Enum
from typing import Tuple
import numpy as np
import torch

class TokenType(Enum):
    """
    Enum representing different types of tokens used in sequence modeling
    of vehicle and pedestrian tracking data.

    Attributes:
        DUMMY_TOKEN: Placeholder token for initialization or error handling.
        EGO_TOKEN: Represents the ego vehicle in the sequence.
        AGENT_TOKEN: Represents other agents (vehicles, pedestrians) in the sequence.
        PAD_TOKEN: Padding token used to fill sequence length.
    """
    DUMMY_TOKEN = -1
    EGO_TOKEN = 1
    AGENT_TOKEN = 2
    PAD_TOKEN = 3

class ClassType(Enum):
    """
    Enum representing different class types of agents in traffic scenes.

    Attributes:
        VEHICLE: Represents a vehicle.
        PEDESTRIAN: Represents a pedestrian.
        CYCLIST: Represents a cyclist.
    """
    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2

class NpSequenceArray(np.ndarray):
    """
    A numpy array subclass for handling sequences of tracking data efficiently.

    The array is structured with specific dimensions representing different data fields.

    Attributes:
        dim (int): Number of dimensions in the array.
        x_dim (int): Index for the X coordinate.
        y_dim (int): Index for the Y coordinate.
        dist_dim (int): Index for the distance to the ego.
        yaw_dim (int): Index for the yaw angle.
        vx_dim (int): Index for the velocity in X.
        vy_dim (int): Index for the velocity in Y.
        ax_dim (int): Index for the acceleration in X.
        ay_dim (int): Index for the acceleration in Y.
        width_dim (int): Index for the width.
        length_dim (int): Index for the length.
        class_type_dim (int): Index for the class type.
        token_type (int): Index for token type in the sequence.

    """
    dim = 12
    x_dim = 0
    y_dim = 1
    dist_dim = 2
    yaw_dim = 3
    vx_dim = 4
    vy_dim = 5
    ax_dim = 6
    ay_dim = 7
    width_dim = 8
    length_dim = 9
    class_type_dim = 10
    token_type_dim = 11 

    def __new__(cls, input_array):
        """Create a new instance of the array."""
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def x(self):
        """Array slice of X coordinates."""
        return self[..., self.x_dim]

    @property
    def y(self):
        """Array slice of Y coordinates."""
        return self[..., self.y_dim]
    
    @property
    def dist(self):
        """Array slice of distance to ego."""
        return self[..., self.dist_dim]

    @property
    def yaw(self):
        """Array slice of yaw angle."""
        return self[..., self.yaw_dim]

    @property
    def vx(self):
        """Array slice of velocities in X."""
        return self[..., self.vx_dim]

    @property
    def vy(self):
        """Array slice of velocities in Y."""
        return self[..., self.vy_dim]

    @property
    def ax(self):
        """Array slice of acceleration in X."""
        return self[..., self.ax_dim]

    @property
    def ay(self):
        """Array slice of acceleration in Y."""
        return self[..., self.ay_dim]

    @property
    def width(self):
        """Array slice of headings."""
        return self[..., self.width_dim]
      
    @property
    def length(self):
        """Array slice of headings."""
        return self[..., self.length_dim]
    
    @property
    def token_type(self):
        """Array slice of headings."""
        return self[..., self.token_type_dim]

# Constants for NpSequenceArray
NpSequence_DIM = NpSequenceArray.dim

X_DIM = NpSequenceArray.x_dim
Y_DIM = NpSequenceArray.y_dim
DIST_DIM = NpSequenceArray.dist_dim
YAW_DIM = NpSequenceArray.yaw_dim
VX_DIM = NpSequenceArray.vx_dim
VY_DIM = NpSequenceArray.vy_dim
AX_DIM = NpSequenceArray.ax_dim
AY_DIM = NpSequenceArray.ay_dim
WIDTH_DIM = NpSequenceArray.width_dim
LENGTH_DIM = NpSequenceArray.length_dim
CLASS_TYPE_DIM = NpSequenceArray.class_type_dim
TOKEN_TYPE_DIM = NpSequenceArray.token_type_dim

class VocabularyStateType(Enum):
    """Enum of classification types for TrackedObject with integer ranges."""

    X = (0, 199), 'x', 0          # 200, 0.2, -50, 50
    Y = (200, 399), 'y', 1        # 200, 0.2, -50, 50
    DIST = (400, 599), "dist", 2  # 200, 0.2, -100, 100
    YAW = (600, 699), 'yaw', 3    # 100, np.pi/100, -np.pi, np.pi
    VX = (700, 799), 'vx', 4      # 100, 0.25, -25, 25
    VY = (800, 899), 'vy', 5      # 100, 0.25, -25, 25
    AX = (900, 999), 'ax', 6      # 100, 0.25, -10, 10
    AY = (1000, 1099), 'ay', 7    # 100, 0.25, -10, 10
    WIDTH = (1100, 1114), 'width', 8 # 15
    LENGTH = (1115, 1144), 'length', 9 #30
    PAD_TOKEN = (1145, 1145), 'pad_token', 10

    @property
    def num_agent_attributes(self) -> str:
        return 3

    @property
    def vocal_size(self) -> int:
        return self.PAD_TOKEN.end+1

    @property
    def x_range(self) -> Tuple[float, float]:
        return (-50, 50)

    @property
    def y_range(self) -> Tuple[float, float]:
        return (-50, 50)

    @property
    def dist_range(self) -> Tuple[float, float]:
        return (-100, 100)
    
    @property
    def yaw_range(self) -> Tuple[float, float]:
        return (-np.pi, np.pi)

    @property
    def vx_range(self) -> Tuple[float, float]:
        return (-25, 25)

    @property
    def vy_range(self) -> Tuple[float, float]:
        return (-25, 25)
    
    @property
    def ax_range(self) -> Tuple[float, float]:
        return (-10, 10)

    @property
    def ay_range(self) -> Tuple[float, float]:
        return (-10, 10)

    @property
    def width_range(self) -> Tuple[float, float]:
        return (0, 7)

    @property
    def length_range(self) -> Tuple[float, float]:
        return (0, 15)

    @property
    def x_step(self) -> float:
        return (self.x_range[1] - self.x_range[0])/self.nx

    @property
    def y_step(self) -> float:
        return (self.y_range[1] - self.y_range[0])/self.ny
    
    @property
    def dist_step(self) -> float:
        return (self.dist_range[1] - self.dist_range[0])/self.nd

    @property
    def yaw_step(self) -> float:
        return (self.yaw_range[1] - self.yaw_range[0])/self.nyaw

    @property
    def vx_step(self) -> float:
        return (self.vx_range[1] - self.vx_range[0])/self.nvx

    @property
    def vy_step(self) -> float:
        return (self.vy_range[1] - self.vy_range[0])/self.nvy
    
    @property
    def ax_step(self) -> float:
        return (self.ax_range[1] - self.ax_range[0])/self.nax

    @property
    def ay_step(self) -> float:
        return (self.ay_range[1] - self.ay_range[0])/self.nay

    @property
    def width_step(self) -> float:
        return (self.width_range[1] - self.width_range[0])/self.nw

    @property
    def length_step(self) -> float:
        return (self.length_range[1] - self.length_range[0])/self.nl

    @property
    def nx(self) -> int:
        return self.X.end - self.X.start + 1

    @property
    def ny(self) -> int:
        return self.Y.end - self.Y.start + 1
    
    @property
    def nd(self) -> int:
        return self.DIST.end - self.DIST.start + 1

    @property
    def nyaw(self) -> int:
        return self.YAW.end - self.YAW.start + 1

    @property
    def nvx(self) -> int:
        return self.VX.end - self.VX.start + 1

    @property
    def nvy(self) -> int:
        return self.VY.end - self.VY.start + 1

    @property
    def nax(self) -> int:
        return self.AX.end - self.AX.start + 1

    @property
    def nay(self) -> int:
        return self.AY.end - self.AY.start + 1
    
    @property
    def nw(self) -> int:
        return self.WIDTH.end - self.WIDTH.start + 1

    @property
    def nl(self) -> int:
        return self.LENGTH.end - self.LENGTH.start + 1

    @property
    def start(self) -> int:
        """Get the start value of the range."""
        return self.value[0][0]

    @property
    def end(self) -> int:
        """Get the end value of the range."""
        return self.value[0][1]

    @property
    def index(self) -> int:
        """Get the index of the range."""
        return self.value[2]

    @classmethod
    def index_to_state(cls, index: int) -> VocabularyStateType:
        """Convert an index to its corresponding VocabularyStateType."""
        for state in cls:
            if state.index == index:
                return state
        raise ValueError(f"No VocabularyStateType found for index {index}")

    def __contains__(self, value) -> bool:
        """Check if a number is within the range of this enum member."""
        if isinstance(value, VocabularyStateType):
            return self.start <= value.start <= self.end
        elif isinstance(value, int):
            return self.start <= value <= self.end
        else:
            raise TypeError("Unsupported type for containment check")

    def get_sampling_mask(self):
        sampling_mask = np.zeros(self.vocal_size)
        sampling_mask[self.start:self.end+1] = 1
        return sampling_mask > 0

def within_valid_range(data, valid_range):
    """
    Check if the given data points (x, y) are within the specified valid range.
    
    Args:
        data (np.ndarray): The data array containing x and y coordinates.
        valid_range (np.ndarray): An array specifying the valid [xmin, xmax, ymin, ymax] range.
    
    Returns:
        bool: True if the data is within the valid range, False otherwise.
    """
    # Checking if the data's coordinates are within the provided range
    if data[X_DIM] > valid_range[0] and data[X_DIM] < valid_range[1] and \
       data[Y_DIM] > valid_range[2] and data[Y_DIM] < valid_range[3]:
        return True
    return False

X_START, X_END, X_RANGE, X_STEP = VocabularyStateType.X.start, VocabularyStateType.X.end, VocabularyStateType.X.x_range, VocabularyStateType.X.x_step
Y_START, Y_END, Y_RANGE, Y_STEP = VocabularyStateType.Y.start, VocabularyStateType.Y.end, VocabularyStateType.Y.y_range, VocabularyStateType.Y.y_step
DIST_START, DIST_END, DIST_RANGE, DIST_STEP = VocabularyStateType.DIST.start, VocabularyStateType.DIST.end, VocabularyStateType.DIST.y_range, VocabularyStateType.DIST.y_step
YAW_START, YAW_END, YAW_RANGE, YAW_STEP = VocabularyStateType.YAW.start, VocabularyStateType.YAW.end, VocabularyStateType.YAW.yaw_range, VocabularyStateType.YAW.yaw_step
WIDTH_START, WIDTH_END, WIDTH_RANGE, WIDTH_STEP = VocabularyStateType.WIDTH.start, VocabularyStateType.WIDTH.end, VocabularyStateType.WIDTH.width_range, VocabularyStateType.WIDTH.width_step
LENGTH_START, LENGTH_END, LENGTH_RANGE, LENGTH_STEP = VocabularyStateType.LENGTH.start, VocabularyStateType.LENGTH.end, VocabularyStateType.LENGTH.length_range, VocabularyStateType.LENGTH.length_step
VX_START, VX_END, VX_RANGE, VX_STEP = VocabularyStateType.VX.start, VocabularyStateType.VX.end, VocabularyStateType.VX.x_range, VocabularyStateType.VX.x_step
VY_START, VY_END, VY_RANGE, VY_STEP = VocabularyStateType.VY.start, VocabularyStateType.VY.end, VocabularyStateType.VY.y_range, VocabularyStateType.VY.y_step
AX_START, AX_END, AX_RANGE, AX_STEP = VocabularyStateType.AX.start, VocabularyStateType.AX.end, VocabularyStateType.AX.x_range, VocabularyStateType.AX.x_step
AY_START, AY_END, AY_RANGE, AY_STEP = VocabularyStateType.AY.start, VocabularyStateType.AY.end, VocabularyStateType.AY.y_range, VocabularyStateType.AY.y_step

def tokenize_data(data):
    """
    Convert data points into tokens based on specified ranges and steps.
    Operates entirely on GPU using PyTorch tensor operations.
    
    Args:
        data (torch.Tensor): The data tensor to be tokenized, assumed to be on GPU.
    
    Returns:
        torch.Tensor: The tokenized data tensor.
    """
    batch_size, num_agents, _ = data.shape
    # Pre-allocate the output tensor on the same device (GPU) as the input data.
    output = torch.empty(batch_size, num_agents, 10, device=data.device, dtype=torch.int32)

    def calculate_token_tensor(value, value_range, start, step):
        clamped = torch.clamp(value, min=value_range[0], max=value_range[1])
        token = start + torch.round((clamped - value_range[0]) / step).to(torch.int32)
        return token

    # Tokenize each dimension attribute using vectorized operations.
    output[..., 0] = calculate_token_tensor(data[..., X_DIM], X_RANGE, X_START, X_STEP)
    output[..., 1] = calculate_token_tensor(data[..., Y_DIM], Y_RANGE, Y_START, Y_STEP)
    output[..., 2] = calculate_token_tensor(data[..., DIST_DIM], DIST_RANGE, DIST_START, DIST_STEP)
    output[..., 3] = calculate_token_tensor(data[..., YAW_DIM], YAW_RANGE, YAW_START, YAW_STEP)
    output[..., 4] = calculate_token_tensor(data[..., VX_DIM], VX_RANGE, VX_START, VX_STEP)
    output[..., 5] = calculate_token_tensor(data[..., VY_DIM], VY_RANGE, VY_START, VY_STEP)
    output[..., 6] = calculate_token_tensor(data[..., AX_DIM], AX_RANGE, AX_START, AX_STEP)
    output[..., 7] = calculate_token_tensor(data[..., AY_DIM], AY_RANGE, AY_START, AY_STEP)
    output[..., 8] = calculate_token_tensor(data[..., WIDTH_DIM], WIDTH_RANGE, WIDTH_START, WIDTH_STEP)
    output[..., 9] = calculate_token_tensor(data[..., LENGTH_DIM], LENGTH_RANGE, LENGTH_START, LENGTH_STEP)

    return output