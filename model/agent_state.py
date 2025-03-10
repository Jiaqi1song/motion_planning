import numpy as np
from enum import Enum

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

def get_agent_array():
    """
    Generate an array representing an agent's state in a specific frame.

    Returns:
        np.ndarray: A numpy array representing the agent's state in the sequence.
    """

    agent_array = np.zeros((NpSequence_DIM, ))

    # TODO: Generatet the agent array for the current agent
    agent_array[X_DIM] = ...
    agent_array[Y_DIM] = ...
    agent_array[DIST_DIM] = ...
    agent_array[YAW_DIM] = ...
    agent_array[VX_DIM] = ...
    agent_array[VY_DIM] = ...
    agent_array[AX_DIM] = ...
    agent_array[AY_DIM] = ...
    agent_array[WIDTH_DIM] = ...
    agent_array[LENGTH_DIM] = ...
    agent_array[CLASS_TYPE_DIM] = ClassType.VEHICLE.value

    # Note: if it is padding, set special value or use mask
    agent_array[TOKEN_TYPE_DIM] = TokenType.PAD_TOKEN.value

    return agent_array

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

def process_data(max_seq_len=50):
    """
    Process data
    
    Args:
        max_seq_len (int): Maximum sequence length for the output tokenized data.
       
    Returns:
        np.ndarray: The data array.
    """
    output = np.ones((max_seq_len, NpSequence_DIM))

    for i in range(max_seq_len):
        output[i] = get_agent_array()
        
        # TODO: handle special case and out of valid range case

    return output