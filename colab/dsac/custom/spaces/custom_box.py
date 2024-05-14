from gymnasium.spaces import Box
from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
import numpy as np
import torch
import torch.nn.functional as F


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

    def sample(self, mask: None = None) -> np.ndarray:
        logits = torch.ones((1, self.high.shape[0]))
        return F.gumbel_softmax(logits, tau=1.0, hard=True).numpy().astype(self.dtype)


