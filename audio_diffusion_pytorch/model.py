import random
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .diffusion import (
    ADPM2Sampler,
    Diffusion,
    DiffusionSampler,
    Distribution,
    KarrasSchedule,
    LogNormalDistribution,
    Sampler,
    Schedule,
)
from .modules import UNet1d, UNetConditional1d
from .utils import default, exists, to_list

"""
Diffusion Classes (generic for 1d data)
"""


class Model1d(nn.Module):
    def __init__(
        self,
        diffusion_sigma_distribution: Distribution,
        diffusion_sigma_data: int,
        diffusion_dynamic_threshold: float,
        use_classifier_free_guidance: bool = False,
        **kwargs
    ):
        super().__init__()

        UNet = UNetConditional1d if use_classifier_free_guidance else UNet1d

        self.unet = UNet(**kwargs)

        self.diffusion = Diffusion(
            net=self.unet,
            sigma_distribution=diffusion_sigma_distribution,
            sigma_data=diffusion_sigma_data,
            dynamic_threshold=diffusion_dynamic_threshold,
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.diffusion(x, **kwargs)

    def sample(
        self,
        noise: Tensor,
        num_steps: int,
        sigma_schedule: Schedule,
        sampler: Sampler,
        **kwargs
    ) -> Tensor:
        diffusion_sampler = DiffusionSampler(
            diffusion=self.diffusion,
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            num_steps=num_steps,
        )
        return diffusion_sampler(noise, **kwargs)


class DiffusionUpsampler1d(Model1d):
    def __init__(
        self, factor: Union[int, Sequence[int]], in_channels: int, *args, **kwargs
    ):
        self.factor = to_list(factor)
        default_kwargs = dict(
            in_channels=in_channels,
            context_channels=[in_channels],
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})  # type: ignore

    def forward(self, x: Tensor, factor: Optional[int] = None, **kwargs) -> Tensor:
        # Either user provides factor or we pick one at random
        factor = default(factor, random.choice(self.factor))
        # Downsample by picking every `factor` item
        downsampled = x[:, :, ::factor]
        # Upsample by interleaving to get context
        channels = torch.repeat_interleave(downsampled, repeats=factor, dim=2)
        return self.diffusion(x, channels_list=[channels], **kwargs)

    def sample(  # type: ignore
        self, undersampled: Tensor, factor: Optional[int] = None, *args, **kwargs
    ):
        # Either user provides factor or we pick the first
        factor = default(factor, self.factor[0])
        # Upsample channels by interleaving
        channels = torch.repeat_interleave(undersampled, repeats=factor, dim=2)
        noise = torch.randn_like(channels)
        default_kwargs = dict(channels_list=[channels])
        return super().sample(noise, **{**default_kwargs, **kwargs})  # type: ignore


class Bottleneck(nn.Module):
    """Bottleneck interface (subclass can be provided to DiffusionAutoencoder1d)"""

    def forward(self, x: Tensor) -> Tuple[Tensor, Any]:
        raise NotImplementedError()


"""
Audio Diffusion Classes (specific for 1d audio data)
"""


def get_default_model_kwargs():
    return dict(
        channels=128,
        patch_blocks=4,
        patch_factor=2,
        kernel_sizes_init=[1, 3, 7],
        multipliers=[1, 2, 4, 4, 4, 4, 4],
        factors=[4, 4, 4, 2, 2, 2],
        num_blocks=[2, 2, 2, 2, 2, 2],
        attentions=[False, False, False, True, True, True],
        attention_heads=8,
        attention_features=64,
        attention_multiplier=2,
        use_attention_bottleneck=True,
        resnet_groups=8,
        kernel_multiplier_downsample=2,
        use_nearest_upsample=False,
        use_skip_scale=True,
        use_context_time=True,
        diffusion_sigma_distribution=LogNormalDistribution(mean=-3.0, std=1.0),
        diffusion_sigma_data=0.1,
        diffusion_dynamic_threshold=0.0,
    )


def get_default_sampling_kwargs():
    return dict(
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        sampler=ADPM2Sampler(rho=1.0),
    )


class AudioDiffusionModel(Model1d):
    def __init__(self, **kwargs):
        super().__init__(**{**get_default_model_kwargs(), **kwargs})

    def sample(self, *args, **kwargs):
        return super().sample(*args, **{**get_default_sampling_kwargs(), **kwargs})


class AudioDiffusionUpsampler(DiffusionUpsampler1d):
    def __init__(self, in_channels: int, **kwargs):
        default_kwargs = dict(
            **get_default_model_kwargs(),
            in_channels=in_channels,
            context_channels=[in_channels],
        )
        super().__init__(**{**default_kwargs, **kwargs})  # type: ignore

    def sample(self, *args, **kwargs):
        return super().sample(*args, **{**get_default_sampling_kwargs(), **kwargs})


class AudioDiffusionConditional(Model1d):
    def __init__(
        self,
        embedding_features: int,
        embedding_max_length: int,
        rhythm_max_length: int,
        genre_features: Optional[int],
        embedding_mask_proba: float = 0.1,
        **kwargs
    ):
        self.embedding_mask_proba = embedding_mask_proba
        default_kwargs = dict(
            **get_default_model_kwargs(),
            context_embedding_features=embedding_features,
            context_embedding_max_length=embedding_max_length,
            context_rhythm_max_length=rhythm_max_length,
            use_classifier_free_guidance=True,
            context_genre_features=genre_features,
        )
        super().__init__(**{**default_kwargs, **kwargs})

    def forward(self, *args, **kwargs):
        default_kwargs = dict(embedding_mask_proba=self.embedding_mask_proba)
        return super().forward(*args, **{**default_kwargs, **kwargs})

    def sample(self, *args, **kwargs):
        default_kwargs = dict(
            **get_default_sampling_kwargs(),
            embedding_scale=5.0,
        )
        return super().sample(*args, **{**default_kwargs, **kwargs})
