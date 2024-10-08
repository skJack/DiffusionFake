import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import pdb
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.modules.linear import Linear
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.dropout import Dropout2d
import copy
from torch.nn.modules.linear import Linear
import pdb
try:
    from timm.models import vit_base_patch16_224,tf_efficientnet_b0_ns,tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, \
    tf_efficientnet_b6_ns, tf_efficientnet_b7_ns, \
    xception

    from efficientnet_pytorch.model import EfficientNet
except:
    from timm.models import vit_base_patch16_224,tf_efficientnet_b0_ns,tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, \
    tf_efficientnet_b6_ns, tf_efficientnet_b7_ns, \
    xception

    from efficientnet_pytorch.model import EfficientNet

  
encoder_params = {
    "vit_base_patch16_224": {
        "features": 2048,
        "init_op": partial(vit_base_patch16_224, pretrained=True)
    },
    
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True)
    },
    
}


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)



        
class FeatureFilter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # 增加通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 增加空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.self_attention = nn.MultiheadAttention(in_channels, num_heads=4)

    def forward(self, x):
        feat = self.conv1(x)
        feat1 = nn.SiLU()(feat)
        # 应用通道注意力
        channel_weights = self.channel_attention(feat1)
        feat = feat * channel_weights
        # 应用空间注意力
        spatial_weights = self.spatial_attention(feat1)
        feat = feat * spatial_weights
        
        feat = feat.permute(2, 3, 0, 1).contiguous()  # (H, W, B, C)
        H,W,B,C = feat.shape
        feat = feat.view(H*W, B, C)  # (H, W, B*C)
        feat1 = feat1.permute(2, 3, 0, 1).contiguous()  # (H, W, B, C)
        feat1 = feat1.view(H*W, B, C)
        feat = self.self_attention(feat1, feat, feat, need_weights=False)[0]
        feat = feat.view(H, W, B, C).permute(2, 3, 0, 1).contiguous()

        feat = self.conv2(feat)
        
        return feat
  
class WeightNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

class GuideNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )
        

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
        self.use_feature_filter = False
        self.ch = ch
        self.input_block_chans = input_block_chans
        
        
        
    def define_feature_filter(self,encoder='tf_efficientnet_b4_ns'):
        # 初始化source和target的特征过滤器

        
        # 初始化特征提取器
        encoder_ori = encoder_params[encoder]["init_op"]()

        
        self.input_hint_block = encoder_ori


        self.upsample_conv_s1 = nn.ConvTranspose2d(
            in_channels=1792,
            out_channels=1792,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.upsample_conv_s2 = nn.ConvTranspose2d(
            in_channels=1792,
            out_channels=1792,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.upsample_conv_t1 = nn.ConvTranspose2d(
            in_channels=1792,
            out_channels=1792,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.upsample_conv_t2 = nn.ConvTranspose2d(
            in_channels=1792,
            out_channels=1792,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.feature_s = FeatureFilter(1792,320) 
        self.feature_t = FeatureFilter(1792,320) 
        self.fc = Linear(1792, 1) 
        self.fc_s = Linear(1280, 1) 
        self.fc_t = Linear(1280, 1) 

        self.contribution_source = WeightNet(1792)
        self.contribution_target = WeightNet(1792)
        self.global_pool = AdaptiveAvgPool2d((1, 1))

        self.use_feature_filter = True

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x_source,x_target, hint, timesteps, context,**kwargs):

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        feature = self.input_hint_block.forward_features(hint) #[bs,320,32,32]
        up_feat_s = self.upsample_conv_s1(feature)
        up_feat_s = self.upsample_conv_s2(up_feat_s)

        up_feat_t = self.upsample_conv_t1(feature)
        up_feat_t = self.upsample_conv_t2(up_feat_t)
        guided_hint_source = self.feature_s(up_feat_s)
        guided_hint_target = self.feature_t(up_feat_t)

        contribution_s = self.contribution_source(feature)
        contribution_t = self.contribution_target(feature)
        contribution = torch.cat((contribution_s,contribution_t),dim=1)


        
        output = self.global_pool(feature).flatten(1)
        output = self.fc(output)


        source_outs = []
        target_outs = []

        h_source = x_source.type(self.dtype) #[4,4,32,32]
        h_target = x_target.type(self.dtype) #[4,4,32,32]

        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint_source is not None:
                h_source = module(h_source, emb, context)
                h_target = module(h_target, emb, context)

                h_source += guided_hint_source
                h_target += guided_hint_target

                guided_hint_source = None
                guided_hint_target = None

            else:
                h_source = module(h_source, emb, context)
                h_target = module(h_target, emb, context)

            
            source_outs.append(zero_conv(h_source, emb, context))
            target_outs.append(zero_conv(h_target, emb, context))

            

        h_source = self.middle_block(h_source, emb, context) #[4,1280,4,4]
        h_target = self.middle_block(h_target, emb, context) #[4,1280,4,4]


        source_outs.append(self.middle_block_out(h_source, emb, context))
        target_outs.append(self.middle_block_out(h_target, emb, context))
        




        return source_outs,target_outs,output,contribution



class DiffusionFake(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, label_key,target_stage_key,only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.target_stage_key = target_stage_key
        self.label_key = label_key
        self.only_mid_control = only_mid_control
        self.control_scales_s = [1.0] * 13
        self.control_scales_t = [0.3] * 13

        self.criterion = nn.BCELoss()

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        source, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        target, _ = super().get_input(batch, self.target_stage_key, *args, **kwargs)
        label = batch[self.label_key].float()
        control = batch[self.control_key]
        source_score = batch['source_score']
        target_score = batch['target_score']

        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return source,target, dict(c_crossattn=[c], c_concat=[control]),(label,source_score,target_score)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        if isinstance(x_noisy,tuple):
            x_noisy_s = x_noisy[0]
            x_noisy_t = x_noisy[1]
        diffusion_model = self.model.diffusion_model #ControlledUnetModel

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy_s, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            if isinstance(x_noisy,tuple):
                control_source,control_target,output,contribution = self.control_model(x_source=x_noisy_s,x_target=x_noisy_t, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt) #返回的control net的list
                self.source_weight, self.target_weight = contribution[:, 0], contribution[:, 1]

                control_source = [c * self.source_weight.view(-1, 1, 1, 1) for c in control_source]
                control_target = [c * self.target_weight.view(-1, 1, 1, 1) for c in control_target]


                eps_source = diffusion_model(x=x_noisy_s, timesteps=t, context=cond_txt, control=control_source, only_mid_control=self.only_mid_control)
                eps_target = diffusion_model(x=x_noisy_t, timesteps=t, context=cond_txt, control=control_target, only_mid_control=self.only_mid_control)
            else:
                control_source,control_target,output,contribution = self.control_model(x_source=x_noisy,x_target=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt) #返回的control net的list
                self.weight = contribution
                self.source_weight, self.target_weight = contribution[:, 0], contribution[:, 1]

                
                control_source = [c * self.source_weight.view(-1, 1, 1, 1) for c in control_source]
                control_target = [c * self.target_weight.view(-1, 1, 1, 1) for c in control_target]

                

                eps_source = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control_source, only_mid_control=self.only_mid_control)
                eps_target = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control_target, only_mid_control=self.only_mid_control)

        return eps_source,eps_target,output


    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z_s,z_t, c,label = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z_s.shape[0], N)
        n_row = min(z_s.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z_s)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z_s[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def shared_step(self, batch, **kwargs):
        source, target,c,label = self.get_input(batch, self.first_stage_key)
        loss = self(source, target, c,label)
        return loss

    def forward(self, source, target, c, label,*args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (source.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        loss,bceloss,loss_dict = self.p_losses((source,target), c, t, label,*args, **kwargs)
        
        loss = loss+bceloss
        # loss = loss


        return loss,loss_dict
    
    def p_losses(self, x_start,cond, t, labels, noise=None):
        if isinstance(x_start,tuple):
            x_start_s = x_start[0]
            x_start_t = x_start[1]
            noise = default(noise, lambda: torch.randn_like(x_start_s))
            x_noisy_s = self.q_sample(x_start=x_start_s, t=t, noise=noise)
            x_noisy_t = self.q_sample(x_start=x_start_t, t=t, noise=noise)
            source_output, target_output, output = self.apply_model((x_noisy_s,x_noisy_t), t, cond)

        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            source_output, target_output, output = self.apply_model(x_noisy, t, cond)

        if isinstance(labels,tuple):
            source_score = labels[1].float()
            target_score = labels[2].float()
            ce_labels = labels[0]

       
        cls_output = output

        loss_dict = {}
        prefix = f't' if self.training else f'v'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # real fake loss
        cls_output = nn.Sigmoid()(cls_output)
        cls_output = cls_output.squeeze()
        #control
        prediction = (cls_output >= 0.5).float()
        bce_loss = self.criterion(cls_output, ce_labels)
        
        

        loss_dict.update({f'{prefix}/l_ce': bce_loss})
        loss_dict.update({f'{prefix}/acc': prediction.mean()})

        loss_simple_source = self.get_loss(source_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/l_sour': loss_simple_source.mean()})
        loss_simple_target = self.get_loss(target_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/l_targ': loss_simple_target.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss_source = loss_simple_source / torch.exp(logvar_t) + logvar_t
        loss_target = loss_simple_target / torch.exp(logvar_t) + logvar_t
        loss = loss_source + loss_target

        source_weight_loss = nn.MSELoss()(self.source_weight, source_score)
        target_weight_loss = nn.MSELoss()(self.target_weight, target_score)
        weight_loss = source_weight_loss+target_weight_loss
        loss_dict["w_l"] = weight_loss
        loss = loss+weight_loss


        

        if self.learn_logvar:
            loss_dict.update({f'{prefix}/l_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb_source = self.get_loss(source_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb_source = (self.lvlb_weights[t] * loss_vlb_source).mean()
        loss_vlb_target = self.get_loss(target_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb_target = (self.lvlb_weights[t] * loss_vlb_target).mean()
        loss_vlb = loss_vlb_target + loss_vlb_source
        loss_dict.update({f'{prefix}/l_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)

        loss_dict.update({f'{prefix}/loss': loss})

        return loss, bce_loss, loss_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
