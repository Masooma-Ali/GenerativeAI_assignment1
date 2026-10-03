"""Task 1 universal denoising autoencoder.

128x128x3 --(5 strided conv stages)--> 4x4xC --flatten+Linear--> z (latent_dim)
          --Linear+reshape--> 4x4xC --(5 upsample+conv stages)--> 128x128x3

* Genuine bottleneck: with latent_dim=256 the image (49,152 values) is squeezed
  into 256 numbers, so the network cannot simply copy its input.
* NO skip connections (nothing bypasses z), so the bottleneck requirement is
  satisfied by construction.
* Decoder uses nearest-neighbour upsample + conv (avoids checkerboard artefacts
  of transposed convolutions).
"""
import torch
import torch.nn as nn


def conv_block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, base=32, latent_dim=256):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 8]   # channels grow
        layers, cin = [], 3
        for c in chs:                                           # 128->64->32->16->8->4
            layers += [conv_block(cin, c, stride=2), conv_block(c, c, stride=1)]
            cin = c
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(chs[-1] * 4 * 4, latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, base=32, latent_dim=256, dropout=0.1):
        super().__init__()
        chs = [base * 8, base * 8, base * 4, base * 2, base]
        self.c0 = chs[0]
        self.drop = nn.Dropout(dropout)                         # dropout on the latent code
        self.fc = nn.Sequential(nn.Linear(latent_dim, chs[0] * 4 * 4), nn.LeakyReLU(0.2, inplace=True))
        layers, cin = [], chs[0]
        for c in chs:                                           # 4->8->16->32->64->128
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       conv_block(cin, c), conv_block(c, c)]
            cin = c
        self.conv = nn.Sequential(*layers)
        self.out = nn.Conv2d(cin, 3, 3, 1, 1)

    def forward(self, z):
        x = self.fc(self.drop(z)).view(-1, self.c0, 4, 4)
        return torch.sigmoid(self.out(self.conv(x)))


class UniversalAE(nn.Module):
    def __init__(self, base=32, latent_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(base, latent_dim)
        self.decoder = Decoder(base, latent_dim, dropout)

    def forward(self, x):
        return self.decoder(self.encoder(x))
