import torch.nn as nn
import torch
import dataclasses
import contextlib
from datetime import timedelta
from aurora.model.lora import LoRAMode
from timm.models.layers import DropPath, to_3tuple
from aurora import Batch, Metadata
from aurora.model.util import (
    check_lat_lon_dtype,
    init_weights,
)
from aurora.model.patchembed import LevelPatchEmbed
from einops import rearrange
from aurora.model.swin3d import pad_3d, window_partition_3d,PatchMerging3D
from aurora.model.posencoding import pos_scale_enc
from aurora.model.fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    levels_expansion,
    pos_expansion,
    scale_expansion,
)
from small_model.swin_model import Swin3DTransformerBackbone

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
    
class CustomAurora(nn.Module):
    def __init__(self, model,
                 window_size: tuple[int, int, int] = (2, 6, 12),
                # big Swin 3D Transformer backbone
                # encoder_depths: tuple[int, ...] = (6, 10, 8),
                # encoder_num_heads: tuple[int, ...] = (8, 16, 32),
                # decoder_depths: tuple[int, ...] = (8, 10, 6),
                # decoder_num_heads: tuple[int, ...] = (32, 16, 8),
                # embed_dim: int = 512,
                # small Swin 3D Transformer backbone.
                encoder_depths: tuple[int, ...] = (2, 2, 2),
                encoder_num_heads: tuple[int, ...] = (4, 8, 16),
                decoder_depths: tuple[int, ...] = (2, 2, 2),
                decoder_num_heads: tuple[int, ...] = (16, 8, 4),
                embed_dim: int = 256,
            
                mlp_ratio: float = 4.0,
                drop_path: float = 0.0,
                drop_rate: float = 0.0,
                timestep: timedelta = timedelta(hours=6),
                
                 ):
        super(CustomAurora, self).__init__()
        self.model = model
        self.surf_stats=self.model.surf_stats
        

        self.hypernetwork_backbone = Swin3DTransformerBackbone(
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
        self.patch_size = 4
        self.latent_levels=4
        self.timestep=timestep
        self.embed_dim=embed_dim

        surf_vars=("sshf", "slhf")  # Surface variables to use.
        max_history_size= 2
        assert max_history_size > 0, "At least one history step is required."
        self.hypernetwork_surf_token_embeds = LevelPatchEmbed(surf_vars, self.patch_size, embed_dim, max_history_size)

        # Learnable embedding to encode the surface level.
        self.hypernetwork_surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.hypernetwork_surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.hypernetwork_surf_norm = nn.LayerNorm(embed_dim)

        self.hypernetwork_pos_embed = nn.Linear(embed_dim, embed_dim)
        self.hypernetwork_scale_embed = nn.Linear(embed_dim, embed_dim)
        self.hypernetwork_lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.hypernetwork_absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.hypernetwork_pos_drop = nn.Dropout(p=drop_rate)

        self.hypernetwork_surf_heads = nn.ParameterDict(
            {name: nn.Linear(embed_dim*2, self.patch_size**2) for name in surf_vars}
        )
        # ========== 优化1: 预先创建常用的张量，避免重复创建 ==========
        self.register_buffer('lat_buffer', torch.linspace(90, -90, 128))
        self.register_buffer('lon_buffer', torch.linspace(0, 360, 256 + 1)[:-1])
        self.atmos_levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    def _create_metadata(self, time):
        """复用metadata创建逻辑"""
        return Metadata(
            lat=self.lat_buffer,
            lon=self.lon_buffer,
            time=time,
            atmos_levels=self.atmos_levels,
        )
    def biaozhunhua(self,batch):
        batch = self.model.batch_transform_hook(batch)
        # Get the first parameter. We'll derive the data type and device from this parameter.
        p = next(self.model.parameters())#[3,512]
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)
        return batch
    def anti_biaozhunhua(self,batch):
        batch =batch.unnormalise(surf_stats=self.surf_stats)
        return batch
    
    def small_model(self,batch):
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
        x_surf = self.hypernetwork_surf_token_embeds(x_surf, surf_vars)  # (B, L, D) [bs,2048,512]
        dtype = x_surf.dtype  # When using mixed precision, we need to keep track of the dtype.
        # Add surface level encoding. This helps the model distinguish between surface and
        # atmospheric levels./
        x_surf = x_surf + self.hypernetwork_surf_level_encoding[None, None, :].to(dtype=dtype)# [10,2048,512]
        # Since the surface level is not aggregated, we add a Perceiver-like MLP only.
        x_surf = x_surf + self.hypernetwork_surf_norm(self.hypernetwork_surf_mlp(x_surf))# [10,2048,512]
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        pos_encode, scale_encode =pos_encode.to(x_surf.device), scale_encode.to(x_surf.device)
        # Encodings are (L, D).
        pos_encode = self.hypernetwork_pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.hypernetwork_scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        x = x_surf + pos_encode + scale_encode#[bs,4,2048,512]

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D) [bs, 2048, 512]

        # Add lead time embedding.
        lead_hours = self.timestep.total_seconds() / 3600#6
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)#[bs]
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)#[bs,512]
        lead_time_emb = self.hypernetwork_lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L', D) + (B, 1, D)

        # Add absolute time embedding.
        absolute_times_list = [t.timestamp() / 3600 for t in batch.metadata.time]  # Times in hours
        absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.hypernetwork_absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.hypernetwork_pos_drop(x)#torch.Size([40, 2048, 256])
        #提取backbone feature    
        x,enc_feat,dec_feat = self.hypernetwork_backbone(x,
                          lead_time=self.timestep,
                          patch_res=patch_res,
                          rollout_step=batch.metadata.rollout_step,)#[bs, 2048, 1024]
        #decoder
        # x=x.unsqueeze(2)

        # # Decode surface vars. Run the head for every surface-level variable.
        # x_surf = torch.stack([self.hypernetwork_surf_heads[name](x) for name in surf_vars], dim=-1)#[bs,512,1,16,2]
        
        # x_surf = x_surf.reshape(*x_surf.shape[:3], -1)  # (B, L, 1, V_S*p*p)  [bs, 512, 1, 32]
        # surf_preds = x_surf.reshape(shape=(B, 32, 64, 1, 4, 4, len(surf_vars)))#torch.Size([2, 2048, 1, 96])->torch.Size([2, 32, 64, 1, 4, 4, 6])
        # surf_preds = rearrange(surf_preds, "B H W C P1 P2 V -> B V C H P1 W P2")#torch.Size([2, 6, 1, 32, 4, 64, 4])
        # surf_preds = surf_preds.reshape(shape=(B, len(surf_vars), 1, 32 * 4, 64 * 4))#[2, len(surf_vars), 1, 128, 256]
        # surf_preds = surf_preds.squeeze(2) # (B, V_S, H, W)[bs, len(surf_vars), 128, 256]
        # surf_preds = torch.clamp(surf_preds, min=-10, max=10)
        
        return enc_feat,dec_feat
    def forward(self, batch,args):
        metadata = self._create_metadata(batch.metadata.time)
        big_batch=Batch(
                surf_vars={"2t": batch.surf_vars['2t'], "10u":batch.surf_vars['10u'], "10v":batch.surf_vars['10v'],"tp":batch.surf_vars['tp'],"sshf":batch.surf_vars['sshf'], "slhf":batch.surf_vars['slhf']},
                static_vars={"lsm":batch.static_vars['lsm'], "z":batch.static_vars['z'], "slt":batch.static_vars['slt']},
                atmos_vars={"z":batch.atmos_vars['z'], "u":batch.atmos_vars['u'], "v":batch.atmos_vars['v'], "t":batch.atmos_vars['t'], "r":batch.atmos_vars['r']},
                metadata=metadata)
        small_batch=Batch(
                surf_vars={"sshf":batch.surf_vars['sshf'], "slhf":batch.surf_vars['slhf']},
                static_vars={},
                atmos_vars={},
                metadata=metadata)
        enc_feat,_=self.small_model(small_batch)
        hyp_x=tuple((enc_feat,_))
        
        
        return self.model.forward(big_batch,hyp_x)
        