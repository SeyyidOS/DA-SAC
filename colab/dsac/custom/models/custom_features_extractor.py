from custom.models.attention_models import AttentionConv, DenseAttention, LargeKernelAttention
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import torch.nn as nn


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, attention_type='san'):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()

        # Choose the attention mechanism
        if attention_type == 'csa':
            self.attn = AttentionConv(64, 8, 1)
        elif attention_type == 'dsa':
            self.attn = DenseAttention(64, 8)
        elif attention_type == 'lka':
            self.attn = LargeKernelAttention(64, 8, 1)
        else:
            self.attn = None

        self.linear = nn.Linear(8 * 7 * 7, features_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        if self.attn:
            x = self.attn(x)
        x = self.flatten(x)
        return F.relu(self.linear(x))
