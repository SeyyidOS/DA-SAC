import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import torch
import math

class DenseAttention(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=None, bias=True):
        super(DenseAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim or in_features

        self.key_dense = nn.Linear(in_features, self.hidden_dim, bias=bias)
        self.query_dense = nn.Linear(in_features, self.hidden_dim, bias=bias)
        self.value_dense = nn.Linear(in_features, self.hidden_dim, bias=bias)
        self.output_dense = nn.Linear(self.hidden_dim, out_features, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch_size, seq_len = x.size()

        # Compute key, query, value
        k = self.key_dense(x)  # (batch_size, seq_len, hidden_dim)
        q = self.query_dense(x)  # (batch_size, seq_len, hidden_dim)
        v = self.value_dense(x)  # (batch_size, seq_len, hidden_dim)

        # Compute attention scores

        attn_scores = q.T @ k
        # attn_scores = torch.bmm(q, k.transpose(0, 1))  # (batch_size, seq_len, seq_len)
        attn_scores = attn_scores / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Apply attention weights to value
        attn_output = attn_weights @ v.T
        # attn_output = torch.bmm(attn_weights, v)  # (batch_size, seq_len, hidden_dim)

        attn_output = torch.reshape(attn_output, (batch_size, -1))
        # Final linear projection
        output = self.output_dense(attn_output)  # (batch_size, seq_len, out_features)

        return output

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.key_dense.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.query_dense.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.value_dense.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.output_dense.weight, a=math.sqrt(5))

        if self.key_dense.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.key_dense.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.key_dense.bias, -bound, bound)

        if self.query_dense.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.query_dense.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.query_dense.bias, -bound, bound)

        if self.value_dense.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.value_dense.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.value_dense.bias, -bound, bound)

        if self.output_dense.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.output_dense.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.output_dense.bias, -bound, bound)

class LargeKernelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(LargeKernelAttention, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divisible by groups."

        # Relative position encodings for height and width
        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        # Convolutional layers for key, query, and value
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        # Large kernel convolutional layer for capturing wider context
        self.large_kernel_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, groups=groups, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights for convolutional layers
        nn.init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.large_kernel_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.key_conv.bias is not None:
            nn.init.constant_(self.key_conv.bias, 0)
            nn.init.constant_(self.query_conv.bias, 0)
            nn.init.constant_(self.value_conv.bias, 0)
            nn.init.constant_(self.large_kernel_conv.bias, 0)

    def forward(self, x):
        batch, channels, height, width = x.size()

        # Apply padding to input
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        # Unfold to extract sliding local blocks
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        # Split and concatenate key tensor with relative position encodings
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        # Reshape tensors for group-wise operations
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        # Compute attention and apply softmax
        out = q_out * k_out
        out = F.softmax(out, dim=-1)

        # Compute weighted sum using einsum
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        # Apply large kernel convolution
        out = self.large_kernel_conv(out)

        return out


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)
