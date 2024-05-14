from gym.spaces import Box
from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
import numpy as np

class CustomBox(Box):
    def __init__(
        self,
        low: Union[SupportsFloat, np.ndarray],
        high: Union[SupportsFloat, np.ndarray],
        shape: Optional[Sequence[int]] = None,
        dtype: Type = np.float32,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):

      super().__init__(low, high, shape, dtype, seed)
