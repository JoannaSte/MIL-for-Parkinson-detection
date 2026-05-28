import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

from distribution_pooling_filter import DistributionPoolingFilter


class SpectrogramResNet(ResNet):
    """ResNet-18 adapted for single-channel mel spectrograms.

    Input shape:  [batch_size, num_segments, 1, H, W]
    Output shape: [batch_size, num_segments, num_features]
    """

    def __init__(self, num_features):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_features)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        batch_size, num_segments, channels, height, width = x.shape
        x = x.view(batch_size * num_segments, channels, height, width)
        x = super().forward(x)
        x = torch.softmax(x, dim=-1)
        x = x.view(batch_size, num_segments, -1)
        return x  # [batch_size, num_segments, num_features]


class FeatureExtractor(nn.Module):
    """Extracts instance-level features from spectrograms using ResNet."""

    def __init__(self, num_features=32):
        super().__init__()
        self._resnet = SpectrogramResNet(num_features=num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self._resnet(x)   # [B, N, num_features]
        out = self.relu(out)
        return out


class MLPClassifier(nn.Module):
    """MLP that maps bag-level distribution features to class logits."""

    def __init__(self, num_features=32, num_bins=11, num_classes=2,
                 hidden_sizes=(192, 96), dropout_rate=0.5):
        super().__init__()
        layers = [nn.Dropout(dropout_rate)]
        in_size = num_features * num_bins
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)  # [B, num_classes]


class MILModel(nn.Module):
    """Full MIL pipeline:
        1. FeatureExtractor   – ResNet extracts per-instance feature vectors
        2. DistributionPoolingFilter – aggregates instances into a bag distribution
        3. MLPClassifier      – classifies the bag as PD (1) or HC (0)
    """

    def __init__(self, num_features=32, num_bins=11, sigma=0.1, num_classes=2,
                 mlp_hidden_sizes=(192, 96), mlp_dropout=0.5):
        super().__init__()
        self._feature_extractor = FeatureExtractor(num_features=num_features)
        self._pooling_filter = DistributionPoolingFilter(num_bins=num_bins, sigma=sigma)
        self._classifier = MLPClassifier(
            num_features=num_features,
            num_bins=num_bins,
            num_classes=num_classes,
            hidden_sizes=mlp_hidden_sizes,
            dropout_rate=mlp_dropout,
        )

    def forward(self, x):
        # x: [B, N, 1, H, W]
        features = self._feature_extractor(x)          # [B, N, num_features]
        distribution = self._pooling_filter(features)  # [B, num_features, num_bins]
        flat = torch.flatten(distribution, start_dim=1)  # [B, num_features * num_bins]
        logits = self._classifier(flat)                # [B, num_classes]
        return logits
