"""Task 2 corruption classifier: image -> logits over [clean, salt_pepper, blur, occlusion].

No early down-sampling: the first stage runs at full 128x128 resolution because salt-and-pepper
noise and mild blur are pixel-level cues that strided/pooled stems would hide."""
import torch
import torch.nn as nn

from .autoencoder import conv_block

CHANNELS = {"small": [16, 32, 64, 128], "medium": [32, 64, 128, 256], "large": [64, 128, 256, 512]}


class CorruptionClassifier(nn.Module):
    def __init__(self, channels="medium", dropout=0.3, n_classes=4):
        super().__init__()
        layers, cin = [], 3
        for c in CHANNELS[channels]:
            layers += [conv_block(cin, c), conv_block(c, c), nn.MaxPool2d(2)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(cin, n_classes))

    def forward(self, x):                      # returns LOGITS (use softmax for probabilities)
        return self.head(self.features(x))
