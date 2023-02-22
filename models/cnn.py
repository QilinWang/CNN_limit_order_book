import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


import math

import os
import warnings

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


warnings.filterwarnings("ignore")

from pytorch_forecasting.models import BaseModel

 
def get_conv2d_subnet(dilation_num, channel=8):
    layers = [nn.Conv2d(in_channels=1, out_channels=channel, kernel_size=1),
            nn.ConstantPad2d((1,1,0,1), 0),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(2, 3),  groups=channel),]

    for i in range(dilation_num):  # layernum = 3
        layers.append(nn.ConstantPad2d((2,2,0,1), 0))
        layers.append(nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(2, 3), dilation=(1, 2), groups=channel))
        #     # [n+2p-d(k-1)-1]/s+1 = n => [n+2*2-2(3-1)-1]/2+1=0 => p=2 on each side

    layers.append(nn.BatchNorm2d(channel))
    layers.append(nn.Mish())
    return nn.Sequential(*layers)

def get_conv1d_embed():
    return nn.Sequential(
            Rearrange("b t d -> b d t"),
            nn.ConstantPad1d((2, 0), 0),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3),
            # nn.Dropout(0.1),
            Rearrange("b d t -> b t d"),
        )

class CNNExtractor(nn.Module):
    def __init__(self, channel=8):
        super().__init__()

        # self.conv1 = nn.Sequential(
        #     Rearrange("b t c v d -> (b t) c v d "),
        #     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 3), padding=(0, 1), groups=32),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2), groups=32),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2), groups=32),
        #     nn.Mish(),
        #     nn.BatchNorm2d(32),
        #     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
        # )
        self.net_1 = get_conv2d_subnet(0)
        self.net_2 = get_conv2d_subnet(1)
        self.net_3 = get_conv2d_subnet(2)
        self.net_4 = get_conv2d_subnet(3)

        self.embed_time_p = get_conv1d_embed()
        self.embed_time_v = get_conv1d_embed()


        self.conv_down = nn.Conv2d(in_channels=channel*4, out_channels=1, kernel_size=1)

        # self.conv_time_embed = nn.Sequential(
        #     Rearrange("b t d -> b d t"),
        #     nn.ConstantPad1d((2, 0), 0),
        #     nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3),
        #     nn.Dropout(0.3),
        #     Rearrange("b d t -> b t d"),
        # )

        self.mlp_time_features = nn.Sequential(
            nn.Linear(8, 20),
            # nn.SiLU(),
            # nn.Dropout(0.3),
            # nn.Linear(20, 20),
            # nn.SiLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):  # [b t d]
        # x = self.conv_time_embed(x)

        p, v, time = x.split_with_sizes((20, 20, 12), dim=-1)  # [b t d]
        p = self.embed_time_p(p)
        v = self.embed_time_v(v)
        p = rearrange(p, 'b t d -> b t 1 1 d')
        v = rearrange(v, 'b t d -> b t 1 1 d')
        pv = torch.concat((p, v), dim=-2)  # [b t 1 v d]

        b, t, c, n, d = pv.shape

        embedding = nn.Embedding(20, t * c * n).to(pv.device)  # encode positions of d
        position_ids = repeat(torch.arange(20, device=pv.device, dtype=torch.long), "d -> b d", b=b)
        position_embeddings = embedding(position_ids)
        pv = rearrange(pv, "b t c n d -> b d (t c n)", b=b, d=d, t=t, n=n, c=c)
        pv = pv + position_embeddings
        pv = rearrange(pv, "b d (t c n) -> b t c n d", b=b, d=d, t=t, n=n, c=c)

        time_mod = time[:, :, [1, 2, 4, 5, 7, 8, 10, 11]]  # [b t 1 d]
        # time_feature = self.mlp_time_features(time_mod.squeeze(2))
        time_mod = nn.Parameter(repeat(time_mod, 'b t d -> b t c n d', c=c, n=n), requires_grad=False)
        time_feature = self.mlp_time_features(time_mod)

        seq = pv   + time_feature  # time_feature

        # seq = self.conv_time_embed(seq)

        seq = rearrange(seq, "b t c n d -> (b t) c n d ")
        # seq = self.conv1(pv)  # [b t c n d] -> [(b t) c n d]

        # a = self.net_1(seq)
        seq = torch.concat([self.net_1(seq), self.net_2(seq),self.net_3(seq),self.net_4(seq) ],dim=1)
        seq = self.conv_down(seq)
        seq = rearrange(seq, "(b t) c n d -> b t (c n d)", b=b)
        
        
        return seq  # + time_feature


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=8, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b h t d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, hidden_dim=mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MlpHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1)
        )

    def forward(self, x):
        x = reduce(x, "b t d -> b d", "mean")
        return self.mlp_head(x)


class Lob(pl.LightningModule):
    def __init__(self) -> None:
        self.save_hyperparameters()
        super().__init__()
        self.cnn = CNNExtractor()
        self.transformer = Transformer(dim=40, depth=3, heads=8, dim_head=8, mlp_dim=40 * 3)
        self.mlp_head = MlpHead(dim=40)

    def forward(self, x):
        x = self.cnn(x)
        x = self.transformer(x)
        x_hat = self.mlp_head(x)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, y, target = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, target)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y, target = batch
        x_hat = self(x)
        val_loss = F.mse_loss(x_hat, target)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y, target = batch
        x_hat = self(x)
        test_loss = F.mse_loss(x_hat, target)
        self.log("test_loss", test_loss)
