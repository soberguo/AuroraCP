import torch
from small_model.swin_model import Swin3DTransformerBackbone
from aurora import Batch,Metadata
from tqdm import tqdm
from einops import rearrange
import math
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from datetime import timedelta
from aurora.model.posencoding import pos_scale_enc
from aurora.model.fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    levels_expansion,
    pos_expansion,
    scale_expansion,
)


class LevelPatchEmbed(nn.Module):
    """At either the surface or at a single pressure level, maps all variables into a single
    embedding."""

    def __init__(
        self,
        var_names: tuple[str, ...],
        patch_size: int,
        embed_dim: int,
        history_size: int = 1,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ) -> None:
        """Initialise.

        Args:
            var_names (tuple[str, ...]): Variables to embed.
            patch_size (int): Patch size.
            embed_dim (int): Embedding dimensionality.
            history_size (int, optional): Number of history dimensions. Defaults to `1`.
            norm_layer (torch.nn.Module, optional): Normalisation layer to be applied at the very
                end. Defaults to no normalisation layer.
            flatten (bool): At the end of the forward pass, flatten the two spatial dimensions
                into a single dimension. See :meth:`LevelPatchEmbed.forward` for more details.
        """
        super().__init__()

        self.var_names = var_names

        self.kernel_size = (history_size,) + to_2tuple(patch_size)
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.weights = nn.ParameterDict(
            {
                # Shape (C_out, C_in, T, H, W). `C_in = 1` here because we're embedding every
                # variable separately.
                name: nn.Parameter(torch.empty(embed_dim, 1, *self.kernel_size))
                for name in var_names
            }
        )

        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise weights."""
        # Setting `a = sqrt(5)` in kaiming_uniform is the same as initialising with
        # `uniform(-1/sqrt(k), 1/sqrt(k))`, where `k = weight.size(1) * prod(*kernel_size)`.
        # For more details, see
        #
        #   https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        #
        for weight in self.weights.values():
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # The following initialisation is taken from
        #
        #   https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv3d
        #
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(next(iter(self.weights.values())))
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, var_names: tuple[str, ...]) -> torch.Tensor:
        """Run the embedding.

        Args:
            x (:class:`torch.Tensor`): Tensor to embed of a shape of `(B, V, T, H, W)`.
            var_names (tuple[str, ...]): Names of the variables in `x`. The length should be equal
                to `V`.

        Returns:
            :class:`torch.Tensor`: Embedded tensor a shape of `(B, L, D]) if flattened,
                where `L = H * W / P^2`. Otherwise, the shape is `(B, D, H', W')`.

        """
        B, V, T, H, W = x.shape
        assert len(var_names) == V, f"{V} != {len(var_names)}."
        assert self.kernel_size[0] >= T, f"{T} > {self.kernel_size[0]}."
        assert H % self.kernel_size[1] == 0, f"{H} % {self.kernel_size[0]} != 0."
        assert W % self.kernel_size[2] == 0, f"{W} % {self.kernel_size[1]} != 0."
        assert len(set(var_names)) == len(var_names), f"{var_names} contains duplicates."

        # Select the weights of the variables and history dimensions that are present in the batch.
        weight = torch.cat(
            [
                # (C_out, C_in, T, H, W)
                self.weights[name][:, :, :T, ...]
                for name in var_names
            ],
            dim=1,
        )
        # Adjust the stride if history is smaller than maximum.
        stride = (T,) + self.kernel_size[1:]

        # The convolution maps (B, V, T, H, W) to (B, D, 1, H/P, W/P)
        proj = F.conv3d(x, weight, self.bias, stride=stride)
        if self.flatten:
            proj = proj.reshape(B, self.embed_dim, -1)  # (B, D, L)
            proj = proj.transpose(1, 2)  # (B, L, D)

        x = self.norm(proj)
        return x

def unpatchify(x: torch.Tensor, V: int, H: int, W: int, P: int) -> torch.Tensor:
    """Unpatchify hidden representation.

    Args:
        x (torch.Tensor): Patchified input of shape `(B, L, C, V * P^2)` where `P` is the
            patch size.
        V (int): Number of variables.
        H (int): Number of latitudes.
        W (int): Number of longitudes.

    Returns:
        torch.Tensor: Unpatchified representation of shape `(B, V, C, H, W)`.
    """
    assert x.dim() == 4, f"Expected 4D tensor, but got {x.dim()}D."
    B, C = x.size(0), x.size(2)
    H = H // P
    W = W // P
    assert x.size(1) == H * W
    assert x.size(-1) == V * P**2

    x = x.reshape(shape=(B, H, W, C, P, P, V))
    x = rearrange(x, "B H W C P1 P2 V -> B V C H P1 W P2")
    x = x.reshape(shape=(B, V, C, H * P, W * P))
    return x
class MLP(nn.Module):
    """A simple one-hidden-layer MLP."""

    def __init__(self, dim: int, hidden_features: int, dropout: float = 0.0) -> None:
        """Initialise.

        Args:
            dim (int): Input dimensionality.
            hidden_features (int): Width of the hidden layer.
            dropout (float, optional): Drop-out rate. Defaults to no drop-out.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""
        return self.net(x)



class Small_model(torch.nn.Module):
    def __init__(
        self,
        surf_stats,
        window_size: tuple[int, int, int] = (2, 6, 12),

        encoder_depths: tuple[int, ...] = (6, 10, 8),
        encoder_num_heads: tuple[int, ...] = (8, 16, 32),
        decoder_depths: tuple[int, ...] = (8, 10, 6),
        decoder_num_heads: tuple[int, ...] = (32, 16, 8),
        embed_dim: int = 512,

        # encoder_depths: tuple[int, ...] = (2, 6, 2),
        # encoder_num_heads: tuple[int, ...] = (4, 8, 16),
        # decoder_depths: tuple[int, ...] = (2, 6, 2),
        # decoder_num_heads: tuple[int, ...] = (16, 8, 4),
        # embed_dim: int = 256,

        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        drop_rate: float = 0.0,
        timestep: timedelta = timedelta(hours=6),
    ) -> None:

        super().__init__()

        self.surf_stats = surf_stats
        self.patch_size = 4
        self.latent_levels=4
        self.timestep=timestep
        self.embed_dim=embed_dim

        surf_vars=("2t", "sshf", "slhf")  # Surface variables to use.
        max_history_size= 2
        assert max_history_size > 0, "At least one history step is required."
        self.surf_token_embeds = LevelPatchEmbed(surf_vars, self.patch_size, embed_dim, max_history_size)

        # Learnable embedding to encode the surface level.
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)
        self.backbone = Swin3DTransformerBackbone(
            window_size=window_size,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path,
            drop_rate=drop_rate,
        )
        self.surf_heads = nn.ParameterDict(
            {name: nn.Linear(embed_dim*2, self.patch_size**2) for name in surf_vars}
        )
        self.head=nn.Conv2d(3,1,kernel_size=3,padding=1)

        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

    def biaozhunhua(self, batch):
        p = next(self.parameters())#[3,512]
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)
        batch = batch.crop(patch_size=self.patch_size)
        batch = batch.to(p.device)
        return batch
    def anti_biaozhunhua(self,batch):
        batch =batch.unnormalise(surf_stats=self.surf_stats)
        return batch
    def forward(self, batch):
        batch=self.biaozhunhua(batch)
        H, W = batch.spatial_shape#128,256
        patch_res = (self.latent_levels,H // (self.patch_size*2),W // (self.patch_size*2))

        #encoder
        x_surf = torch.stack(tuple(batch.surf_vars.values()), dim=2)#[bs,2,3,128,256]
        surf_vars = tuple(batch.surf_vars.keys())

        B, T,  C, H, W = x_surf.size()
        lat, lon = batch.metadata.lat, batch.metadata.lon#128,256

        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        assert lat.shape[0] == H and lon.shape[-1] == W

        # Patch embed the surface level.
        x_surf = rearrange(x_surf, "b t v h w -> b v t h w")#[bs, 3, 2, 128, 256]
        x_surf = self.surf_token_embeds(x_surf, surf_vars)  # (B, L, D) [bs,2048,512]
        dtype = x_surf.dtype  # When using mixed precision, we need to keep track of the dtype.
        # Add surface level encoding. This helps the model distinguish between surface and
        # atmospheric levels./
        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)# [10,2048,512]
        # Since the surface level is not aggregated, we add a Perceiver-like MLP only.
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))# [10,2048,512]
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        # Encodings are (L, D).
        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        x = x_surf + pos_encode + scale_encode#[bs,4,2048,512]

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D) [bs, 2048, 512]

        # Add lead time embedding.
        lead_hours = self.timestep.total_seconds() / 3600#6
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)#[bs]
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)#[bs,512]
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L', D) + (B, 1, D)

        # Add absolute time embedding.
        absolute_times_list = [t.timestamp() / 3600 for t in batch.metadata.time]  # Times in hours
        absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x)#torch.Size([40, 2048, 256])


        #backbone
        x = self.backbone(x,
                          lead_time=self.timestep,
                          patch_res=patch_res,
                          rollout_step=batch.metadata.rollout_step,)#[bs, 2048, 1024]
        #decoder
        x=x.unsqueeze(2)
        # x = rearrange(
        #     x,
        #     "B (C H W) D -> B (H W) C D",
        #     C=patch_res[0],
        #     H=patch_res[1],
        #     W=patch_res[2],
        # )#(bs, 512,4, 1024)

        # Decode surface vars. Run the head for every surface-level variable.
        x_surf = torch.stack([self.surf_heads[name](x) for name in surf_vars], dim=-1)#[bs,512,1,64,3]
        x_surf =x_surf[...,0]
        # x_surf = x_surf.reshape(*x_surf.shape[:3], -1)  # (B, L, 1, V_S*p*p)  [bs, 512, 1, 192]
        surf_preds = x_surf.reshape(shape=(B, 32, 64, 1, 4, 4, 1))#torch.Size([2, 2048, 1, 96])->torch.Size([2, 32, 64, 1, 4, 4, 6])
        surf_preds = rearrange(surf_preds, "B H W C P1 P2 V -> B V C H P1 W P2")#torch.Size([2, 6, 1, 32, 4, 64, 4])
        surf_preds = surf_preds.reshape(shape=(B, 1, 1, 32 * 4, 64 * 4))
        # surf_preds = unpatchify(x_surf, 1, H, W, self.patch_size)#[bs, 3, 1, 128, 256]
        surf_preds = surf_preds.squeeze(2) # (B, V_S, H, W)[bs, 3, 128, 256]
        surf_preds = torch.clamp(surf_preds, min=-10, max=10)
        # y=self.head(surf_preds)
        # y_=y.unnormalise(surf_stats=self.surf_stats)
        return surf_preds[:,0]







        # Add position and scale embeddings to the 3D tensor.
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        # Encodings are (L, D).
        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        x = x + pos_encode + scale_encode#[bs,4,2048,512]

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D) [10, 8192, 512]

        # Add lead time embedding.
        lead_hours = lead_time.total_seconds() / 3600#6
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)#[bs]
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)#[bs,512]
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L', D) + (B, 1, D)

        # Add absolute time embedding.
        absolute_times_list = [t.timestamp() / 3600 for t in batch.metadata.time]  # Times in hours
        absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x)
        return x




if __name__ == "__main__":
    from weather_dataset import WeatherBench128, custom_collate
    from torch.utils.data import Dataset, DataLoader
    from utils import hours_to_datetime,mean_std_1d
    dataset= WeatherBench128(data_folder = "/sharefiles2/guoyixin/datasets/new_weather_tensors2",n=6,train=True)
    train_loader = DataLoader(
        dataset=dataset,
        collate_fn=custom_collate, batch_size=2,
        num_workers=4, pin_memory=False, drop_last=True)
    surf_stats=mean_std_1d()
    model=Small_model(surf_stats=surf_stats).cuda().train()
    start_epoch = 0
    epochs=20
    save_dir = "./checkpoints"
    for epoch in range(start_epoch,epochs):
        model.train()
        total_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for i, (images, targets) in enumerate(train_iter):

            images_1 = torch.stack([im[0].float() for im in images], dim=0).cuda()
            images_2 = torch.stack([im[1].float() for im in images], dim=0).cuda()
            target = torch.stack([t['tgt'].float() for t in targets],dim=0).cuda()
            if torch.isnan(images_1).any() or torch.isinf(images_1).any() or torch.isnan(images_2).any() or torch.isinf(images_2).any():
                # print("Input has NaN or Inf!")
                continue  # 跳过这个 batch，防止训练崩溃
            var_2t=torch.stack([images_1[:, 0], images_2[:, 0]], dim=1)
            var_sshf = torch.stack([images_1[:, 69], images_2[:, 69]], dim=1) # [B, sshf, H, W]
            var_slhf = torch.stack([images_1[:, 70], images_2[:, 70]], dim=1)
            time=tuple(hours_to_datetime(t['filename']) for t in targets)
            batch=Batch(
                surf_vars={"2t": var_2t, "sshf":var_sshf, "slhf":var_slhf},
                static_vars={},
                atmos_vars={},
                metadata=Metadata(
                    lat=torch.linspace(90, -90, 128),
                    lon=torch.linspace(0, 360, 256 + 1)[:-1],
                    time=time,
                    atmos_levels=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
                ))
            preds=model(batch)
