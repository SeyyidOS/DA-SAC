from src.models.attention_models import AttentionConv, DenseAttention, LargeKernelAttention, DenseAttention
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import torch.nn as nn


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, attention_type='san', conv_channels=64, conv_kernel_size=8):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.attention_type = attention_type

        self.flatten = nn.Flatten()

        # Choose the attention mechanism
        if attention_type == 'lcsa':
            self.conv1 = nn.Conv2d(n_input_channels, conv_channels, kernel_size=conv_kernel_size, stride=conv_kernel_size // 2)
            self.conv2 = nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=conv_kernel_size // 2, stride=conv_kernel_size // 4)
            self.conv3 = nn.Conv2d(conv_channels * 2, conv_channels * 2, kernel_size=conv_kernel_size // 4, stride=conv_kernel_size // 8)
            self.attn = AttentionConv(conv_channels * 2, conv_channels // 4, 1)
        elif attention_type == 'csa':
            self.conv1 = nn.Conv2d(n_input_channels, conv_channels, kernel_size=conv_kernel_size, stride=conv_kernel_size // 2)
            self.attn = AttentionConv(conv_channels, conv_channels // 2, 1)
            self.conv2 = nn.Conv2d(conv_channels // 2, conv_channels // 2, kernel_size=conv_kernel_size // 2, stride=conv_kernel_size // 4)
            self.conv3 = nn.Conv2d(conv_channels // 2, conv_channels // 4, kernel_size=conv_kernel_size // 4, stride=conv_kernel_size // 8)
        elif attention_type == 'dsa':
            self.conv1 = nn.Conv2d(n_input_channels, conv_channels, kernel_size=conv_kernel_size, stride=conv_kernel_size // 2)
            self.conv2 = nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=conv_kernel_size // 2, stride=conv_kernel_size // 4)
            self.conv3 = nn.Conv2d(conv_channels * 2, conv_channels * 2, kernel_size=conv_kernel_size // 4, stride=conv_kernel_size // 8)
            self.attn = DenseAttention(conv_channels * 2 * 7 * 7, conv_channels // 4)
        elif attention_type == 'llka':
            self.conv1 = nn.Conv2d(n_input_channels, conv_channels, kernel_size=conv_kernel_size, stride=conv_kernel_size // 2)
            self.conv2 = nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=conv_kernel_size // 2, stride=conv_kernel_size // 4)
            self.conv3 = nn.Conv2d(conv_channels * 2, conv_channels * 2, kernel_size=conv_kernel_size // 4, stride=conv_kernel_size // 8)
            self.attn = LargeKernelAttention(conv_channels * 2, conv_channels // 4, 1)
        elif attention_type == 'lka':
            self.conv1 = nn.Conv2d(n_input_channels, conv_channels, kernel_size=conv_kernel_size, stride=conv_kernel_size // 2)
            self.attn = LargeKernelAttention(conv_channels, conv_channels // 2, 1)
            self.conv2 = nn.Conv2d(conv_channels // 2, conv_channels // 2, kernel_size=conv_kernel_size // 2, stride=conv_kernel_size // 4)
            self.conv3 = nn.Conv2d(conv_channels // 2, conv_channels // 4, kernel_size=conv_kernel_size // 4, stride=conv_kernel_size // 8)
        else:
            self.attn = None

        if attention_type == 'dsa':
            self.linear = nn.Linear(conv_channels // 4, features_dim)
        else:
            result = size_calculator(observation_space.shape[1], 0, conv_kernel_size, conv_kernel_size // 2)
            result = size_calculator(result, 0, conv_kernel_size // 2, conv_kernel_size // 4)
            result = size_calculator(result, 0, conv_kernel_size // 4, conv_kernel_size // 8)
            self.linear = nn.Linear(conv_channels // 4 * result * result, features_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        if self.attention_type in ['csa', 'lka']:
            x = self.attn(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        if self.attention_type == 'dsa':
            x = self.flatten(x)
            x = self.attn(x)
        elif self.attention_type == 'llka' or self.attention_type == 'lcsa':
            x = self.attn(x)
            x = self.flatten(x)
        else:
            x = self.flatten(x)

        return F.relu(self.linear(x))


def size_calculator(N, P, K, S):
    return ((N + (2 * P) - K) // S) + 1