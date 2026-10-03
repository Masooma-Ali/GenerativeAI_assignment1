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


class SpatialAE(nn.Module):
    """Convolutional-bottleneck autoencoder (ablation vs. the flat/FC bottleneck).

    128x128x3 --(4 strided conv stages)--> 8x8xC --1x1 conv--> 8x8xlatent_ch (the bottleneck)
              --1x1 conv--> 8x8xC --(4 upsample+conv stages)--> 128x128x3
    Latent size = latent_ch * 64 values (e.g. 32 -> 2048, i.e. 24x compression of the 49,152
    input values). Keeping the 8x8 layout preserves WHERE things are in the image.
    Still no skip connections: everything passes through the bottleneck."""
    def __init__(self, base=32, latent_ch=32, dropout=0.1):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8]
        enc, cin = [], 3
        for c in chs:                                           # 128->64->32->16->8
            enc += [conv_block(cin, c, stride=2), conv_block(c, c, stride=1)]
            cin = c
        enc.append(nn.Conv2d(cin, latent_ch, 1))                # bottleneck: latent_ch x 8 x 8
        self.encoder = nn.Sequential(*enc)
        self.drop = nn.Dropout(dropout)
        dchs = [base * 8, base * 4, base * 2, base]
        dec = [nn.Conv2d(latent_ch, dchs[0], 1), nn.LeakyReLU(0.2, inplace=True)]
        cin = dchs[0]
        for c in dchs:                                          # 8->16->32->64->128
            dec += [nn.Upsample(scale_factor=2, mode="nearest"), conv_block(cin, c), conv_block(c, c)]
            cin = c
        dec.append(nn.Conv2d(cin, 3, 3, 1, 1))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return torch.sigmoid(self.decoder(self.drop(self.encoder(x))))