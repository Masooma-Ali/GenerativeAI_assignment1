"""Task 4 style-conditioned face-to-sketch cGAN (pix2pix-style).

Generator : U-Net (7 down / 6 up blocks + output layer for 128x128). The style is a learned categorical
            embedding that enters in TWO places: (1) tiled and concatenated to the photo at the input,
            (2) FiLM modulation (scale/shift of the normalised features) in every decoder block.
Discriminator: conditional PatchGAN; it receives photo + sketch + the tiled style embedding (own embedding table).
"""
import torch
import torch.nn as nn


def down_block(cin, cout, norm=True):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
    if norm: layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class FiLMBN(nn.Module):
    """BatchNorm without its own affine, followed by a style-dependent scale and shift."""
    def __init__(self, c, style_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(c, affine=False)
        self.fc = nn.Linear(style_dim, 2 * c)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)      # starts as plain BatchNorm

    def forward(self, x, e):
        g, b = self.fc(e).chunk(2, 1)
        return self.bn(x) * (1 + g[:, :, None, None]) + b[:, :, None, None]


class UpBlock(nn.Module):
    def __init__(self, cin, cout, style_dim, dropout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False)
        self.norm = FiLMBN(cout, style_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, e):
        return torch.relu(self.drop(self.norm(self.conv(x), e)))


class UNetGenerator(nn.Module):
    def __init__(self, base=64, style_dim=16, dropout=0.3, n_styles=3):
        super().__init__()
        b = base
        self.emb = nn.Embedding(n_styles, style_dim)
        chs = [b, 2 * b, 4 * b, 8 * b, 8 * b, 8 * b, 8 * b]               # 128 -> 64 -> ... -> 1
        self.downs, cin = nn.ModuleList(), 3 + style_dim
        for i, c in enumerate(chs):
            self.downs.append(down_block(cin, c, norm=0 < i < len(chs) - 1)); cin = c
        up_out = [8 * b, 8 * b, 8 * b, 4 * b, 2 * b, b]                     # 1 -> 2 -> ... -> 64
        skip_ch = [8 * b, 8 * b, 8 * b, 4 * b, 2 * b, b]
        self.ups, cin = nn.ModuleList(), chs[-1]
        for i, (co, sc) in enumerate(zip(up_out, skip_ch)):
            self.ups.append(UpBlock(cin, co, style_dim, dropout if i < 3 else 0.0)); cin = co + sc
        self.final = nn.ConvTranspose2d(cin, 1, 4, 2, 1)                    # 64 -> 128

    def forward(self, photo, style):
        e = self.emb(style)
        h = torch.cat([photo, e[:, :, None, None].expand(-1, -1, photo.size(2), photo.size(3))], 1)
        skips = []
        for d in self.downs:
            h = d(h); skips.append(h)
        h = skips[-1]
        for i, up in enumerate(self.ups):
            h = torch.cat([up(h, e), skips[-2 - i]], 1)
        return torch.tanh(self.final(h))


class PatchDiscriminator(nn.Module):
    def __init__(self, base=64, style_dim=16, n_styles=3):
        super().__init__()
        b = base
        self.emb = nn.Embedding(n_styles, style_dim)
        def blk(ci, co, s): return [nn.Conv2d(ci, co, 4, s, 1, bias=False), nn.BatchNorm2d(co), nn.LeakyReLU(0.2, inplace=True)]
        self.net = nn.Sequential(nn.Conv2d(3 + 1 + style_dim, b, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),   # 64
                                 *blk(b, 2 * b, 2), *blk(2 * b, 4 * b, 2), *blk(4 * b, 8 * b, 1),             # 32, 16, 15
                                 nn.Conv2d(8 * b, 1, 4, 1, 1))                                                # 14x14 patch logits

    def forward(self, photo, sketch, style):
        e = self.emb(style)[:, :, None, None].expand(-1, -1, photo.size(2), photo.size(3))
        return self.net(torch.cat([photo, sketch, e], 1))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)


def build_gan(cfg, device):
    G = UNetGenerator(cfg["base"], cfg["style_dim"], cfg["dropout"]).to(device)
    D = PatchDiscriminator(cfg["base"], cfg["style_dim"]).to(device)
    G.apply(init_weights); D.apply(init_weights)
    return G, D
