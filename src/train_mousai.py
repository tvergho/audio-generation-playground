import os
import time
import warnings
import random
import dotenv
import pytorch_lightning as pl
from main import utils
from omegaconf import DictConfig, open_dict
from audio_data_pytorch import WAVDataset, AllTransform
from audio_diffusion_pytorch import DiffusionModel, LinearSchedule, UniformDistribution, Sampler, Schedule, Distribution, UNetV0, VDiffusion, VSampler, Diffusion
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import auraloss
import librosa
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import torch
import torch.nn.functional as F
import torchaudio
import wandb
from audio_data_pytorch.utils import fractional_random_split
from einops import rearrange, reduce
from ema_pytorch import EMA
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch import LongTensor, Tensor, nn
from torch.utils.data import DataLoader
import hydra
from inspect import isfunction
from torch.utils.data import Dataset

from transformers import AutoModel

import sys
sys.path.insert(0, '/workspace/CLAP/src')

from infer import infer_text

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from typing_extensions import TypeGuard
from math import atan, cos, pi, sin, sqrt
import torch.nn.functional as F
from einops import rearrange, repeat

# Load environment variables from `.env`.
dotenv.load_dotenv(override=True)
log = utils.get_logger(__name__)

autoencoder = AutoModel.from_pretrained(
    "archinetai/dmae1d-ATC32-v3", trust_remote_code=True
).to("cuda:0")
autoencoder.requires_grad_(False)

T = TypeVar("T")

def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None

def to_batch(
    batch_size: int,
    device: torch.device,
    x: Optional[float] = None,
    xs: Optional[Tensor] = None,
) -> Tensor:
    assert exists(x) ^ exists(xs), "Either x or xs must be provided"
    # If x provided use the same for all batch items
    if exists(x):
        xs = torch.full(size=(batch_size,), fill_value=x).to(device)
    assert exists(xs)
    return xs

def pad_dims(x: Tensor, ndim: int) -> Tensor:
    # Pads additional ndims to the right of the tensor
    return x.view(*x.shape, *((1,) * ndim))


def clip(x: Tensor, dynamic_threshold: float = 0.0):
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)
        # Clamp to a min of 1.0
        scale.clamp_(min=1.0)
        # Clamp all values and scale
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        x = x.clamp(-scale, scale) / scale
        return x


def extend_dim(x: Tensor, dim: int):
    # e.g. if dim = 4: shape [b] => [b, 1, 1, 1],
    return x.view(*x.shape + (1,) * (dim - x.ndim))

class KarrasSchedule(Schedule):
    """https://arxiv.org/abs/2206.00364 equation 5"""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device: Any) -> Tensor:
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max ** rho_inv
            + (steps / (num_steps - 1))
            * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.rho
        sigmas = F.pad(sigmas, pad=(0, 1), value=0.0)
        return sigmas

    
class KDiffusion(Diffusion):
    """Elucidated Diffusion (Karras et al. 2022): https://arxiv.org/abs/2206.00364"""

    alias = "k"

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        sigma_data: float,  # data distribution standard deviation
        dynamic_threshold: float = 0.0,
    ):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas: Tensor) -> Tuple[Tensor, ...]:
        sigma_data = self.sigma_data
        c_noise = torch.log(sigmas) * 0.25
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        # Predict network output and add skip connection
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred

        # Clips in [-1,1] range, with dynamic thresholding if provided
        return clip(x_denoised, dynamic_threshold=self.dynamic_threshold)

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        # Computes weight depending on data distribution
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x: Tensor, noise: Tensor = None, **kwargs) -> Tensor:
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = x + sigmas_padded * noise

        # Compute denoised values
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)

        # Compute weighted loss
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        losses = losses * self.loss_weight(sigmas)
        loss = losses.mean()
        return loss

class ADPM2Sampler(Sampler):
    """https://www.desmos.com/calculator/jbxjlqd9mb"""

    diffusion_types = [KDiffusion]

    def __init__(self, net: nn.Module, rho: float = 1.0):
        super().__init__()
        self.net = net
        self.rho = rho

    def get_sigmas(self, sigma: float, sigma_next: float) -> Tuple[float, float, float]:
        r = self.rho
        sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
        sigma_down = sqrt(sigma_next ** 2 - sigma_up ** 2)
        sigma_mid = ((sigma ** (1 / r) + sigma_down ** (1 / r)) / 2) ** r
        return sigma_up, sigma_down, sigma_mid

    def step(self, x: Tensor, fn: Callable, sigma: float, sigma_next: float) -> Tensor:
        # Sigma steps
        sigma_up, sigma_down, sigma_mid = self.get_sigmas(sigma, sigma_next)
        # Derivative at sigma (∂x/∂sigma)
        d = (x - fn(x, sigma=sigma)) / sigma
        # Denoise to midpoint
        x_mid = x + d * (sigma_mid - sigma)
        # Derivative at sigma_mid (∂x_mid/∂sigma_mid)
        d_mid = (x_mid - fn(x_mid, sigma=sigma_mid)) / sigma_mid
        # Denoise to next
        x = x + d_mid * (sigma_down - sigma)
        # Add randomness
        x_next = x + torch.randn_like(x) * sigma_up
        return x_next

    def forward(
        self, noise: Tensor, fn: Callable, sigmas: Tensor, num_steps: int
    ) -> Tensor:
        x = sigmas[0] * noise
        # Denoise to sample
        for i in range(num_steps - 1):
            x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
        return x

    def inpaint(
        self,
        source: Tensor,
        mask: Tensor,
        fn: Callable,
        sigmas: Tensor,
        num_steps: int,
        num_resamples: int,
    ) -> Tensor:
        x = sigmas[0] * torch.randn_like(source)

        for i in range(num_steps - 1):
            # Noise source to current noise level
            source_noisy = source + sigmas[i] * torch.randn_like(source)
            for r in range(num_resamples):
                # Merge noisy source and current then denoise
                x = source_noisy * mask + x * ~mask
                x = self.step(x, fn=fn, sigma=sigmas[i], sigma_next=sigmas[i + 1])  # type: ignore # noqa
                # Renoise if not last resample step
                if r < num_resamples - 1:
                    sigma = sqrt(sigmas[i] ** 2 - sigmas[i + 1] ** 2)
                    x = x + sigma * torch.randn_like(x)

        return source * mask + x * ~mask
    
class DiffusionSampler(nn.Module):
    def __init__(
        self,
        diffusion: Diffusion,
        *,
        sampler: Sampler,
        sigma_schedule: Schedule,
        num_steps: Optional[int] = 100,
        clamp: bool = True,
    ):
        super().__init__()
        self.denoise_fn = diffusion.denoise_fn
        self.sampler = sampler
        self.sigma_schedule = sigma_schedule
        self.num_steps = num_steps
        self.clamp = clamp

        # Check sampler is compatible with diffusion type
        sampler_class = sampler.__class__.__name__
        diffusion_class = diffusion.__class__.__name__
        message = f"{sampler_class} incompatible with {diffusion_class}"
        assert diffusion.alias in [t.alias for t in sampler.diffusion_types], message

    @torch.no_grad()
    def forward(
        self, noise: Tensor, num_steps: Optional[int] = None, **kwargs
    ) -> Tensor:
        device = noise.device
        num_steps = self.num_steps  # type: ignore
        assert exists(num_steps), "Parameter `num_steps` must be provided"
        # Compute sigmas using schedule
        sigmas = self.sigma_schedule(num_steps, device)
        # Append additional kwargs to denoise function (used e.g. for conditional unet)
        fn = lambda *a, **ka: self.denoise_fn(*a, **{**ka, **kwargs})  # noqa
        # Sample using sampler
        x = self.sampler(noise, fn=fn, sigmas=sigmas, num_steps=num_steps)
        x = x.clamp(-1.0, 1.0) if self.clamp else x
        return x

class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(
        self, num_samples: int, device: torch.device = torch.device("cpu")
    ) -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()




sampling_rate = 48000
wav_length = 2**20
channels = 32
path = "/workspace/data"
strategy = "deepspeed_stage_2_offload"
num_workers = 8
every_n_steps = 250

def fast_scandir(path: str, exts: List[str], recursive: bool = False):
    # Scan files recursively faster than glob
    # From github.com/drscotthawley/aeiou/blob/main/aeiou/core.py
    subfolders, files = [], []

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(path):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in exts:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, exts, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)  # type: ignore

    return subfolders, files

class EmbeddingDataset(Dataset):
    def __init__(self, path):
        self.path = path
        _, files = fast_scandir(self.path, [".pt"], recursive=True)
        self.files = files
    
    def __getitem__(self, idx):
        item = torch.load(f"{self.path}/{idx}.pt")
        return item
    
    def __len__(self):
        return len(self.files)

@hydra.main()
def main(config) -> None:
    log.info("Disabling python warnings!")
    warnings.filterwarnings("ignore")

    # Apply seed for reproducibility
    pl.seed_everything(42)
    
    
    alltransformer = AllTransform(random_crop_size=wav_length, stereo=True, source_rate=44100, target_rate=sampling_rate)
    def transform(wave):
        x = alltransformer(wave)
        with torch.no_grad():
            embed = infer_audio([x])
            return {"waveform": x, "embedding": embed}
        
    # Initialize datamodule
    print(f"Instantiating datamodule.")
    datamodule = Datamodule(
        val_split=0.05,
        batch_size=16,
        num_workers=num_workers,
        pin_memory=True,
        # dataset=WAVDataset(
        #     recursive=True,
        #     sample_rate=44100,
        #     transforms=transform,
        #     path=path,
        #     check_silence=False
        # )
        dataset=EmbeddingDataset(path="/workspace/embeddings")
    )

    # Initialize model
    print(f"Instantiating model.")
    model = Model(
        lr=1e-4,
        lr_beta1=0.95,
        lr_beta2=0.999,
        lr_eps=1e-6,
        lr_weight_decay=1e-3,
        ema_beta=0.995,
        ema_power=0.7,
        model=DiffusionModel(
            net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
            in_channels=channels, # U-Net: number of input/output (audio) channels
            channels=[128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
            factors=[1, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
            items=[2, 2, 2, 4, 8, 8], # U-Net: number of repeating items at each layer
            attentions=[0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
            attention_heads=12, # U-Net: number of attention heads per attention item
            attention_features=64, # U-Net: number of attention features per attention item
            diffusion_t=KDiffusion, # The diffusion method used
            sampler_t=ADPM2Sampler, # The diffusion sampler used
            use_embedding_cfg=True, # U-Net: enables classifier free guidance
            embedding_max_length=64, # U-Net: text embedding maximum length (default for T5-base)
            embedding_features=512, # U-Net: text embedding features (default for T5-base)
            cross_attentions=[0, 0, 1, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
            diffusion_sigma_distribution=LogNormalDistribution(mean=-3, std=1),
            diffusion_sigma_data=0.1,
            diffusion_dynamic_threshold=0.0
        )
    )

    # Initialize all callbacks (e.g. checkpoints, early stopping)
    callbacks = []
    callbacks.append(RichProgressBar())
    callbacks.append(RichModelSummary(max_depth=2))
    callbacks.append(ModelCheckpoint(
        verbose=True,
        save_last=True,
        monitor='valid_loss',
        mode='min',
        dirpath='/workspace',
        filename='{epoch:02d}-{valid_loss:.3f}',
        every_n_train_steps=every_n_steps
    ))
    callbacks.append(SampleLogger(
        num_items=1,
        channels=2,
        sampling_rate=sampling_rate,
        length=wav_length,
        use_ema_model=True,
        num_steps=200
    ))

    # Initialize loggers (e.g. wandb)
    loggers = []
    loggers.append(
        WandbLogger(
            project="audio-diffusion",
            entity="tvergho1",
            job_type="train",
            group="",
            save_dir="/logs"
        )
    )

    print(f"Instantiating trainer.")
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        precision=32,
        min_epochs=0,
        max_epochs=-1,
        enable_model_summary=False,
        log_every_n_steps=1,
        check_val_every_n_epoch=None,
        val_check_interval=every_n_steps,
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        gpus=1,
        num_sanity_val_steps=0,
        accumulate_grad_batches=2
    )

    # Send some parameters from config to all lightning loggers
    print("Logging hyperparameters!")
    utils.log_hyperparameters(
        config={
            "model": "diffusion",
            "datamodule": "wav",
            "trainer": "Trainer"
        },
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    print("Fitting!")
    trainer.fit(model=model, datamodule=datamodule)
    # Make sure everything closed properly
    print("Finalizing!")
    utils.finish(
        config={},
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )
    print(f"Best model ckpt at {trainer.log_dir}")

class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        ema_beta: float,
        ema_power: float,
        model: nn.Module,
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay
        self.model = model
        # self.autoencoder = AutoModel.from_pretrained(
        #     "archinetai/dmae1d-ATC32-v3", trust_remote_code=True
        # )
        # self.autoencoder.requires_grad_(False)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

#     @torch.no_grad()
#     def encode_latent(self, x: Tensor) -> Tensor:
#         z = self.autoencoder.encode(x)
#         return z
    
#     @torch.no_grad()
#     def decode_latent(self, x: Tensor, num_steps:int=20) -> Tensor:
#         z = self.autoencoder.decode(x, num_steps=num_steps)
#         return z
    
    def training_step(self, batch, batch_idx):
        waveforms = batch
        # x = self.encode_latent(waveforms['waveform'])
        x = waveforms['waveform']
        embed = waveforms['embedding']
        loss = self.model(x, embedding_mask_proba=0.1, embedding=embed)
        self.log("train_loss", loss, rank_zero_only=True)
        
        # Update EMA model and log decay
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms = batch
        embed = waveforms['embedding']
        # x = self.encode_latent(waveforms['waveform'])
        x = waveforms['waveform']
        loss = self.model(x, embedding_mask_proba=0.1, embedding=embed)
        self.log("valid_loss", loss, rank_zero_only=True)
        return loss

class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        *,
        val_split: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        print("length", len(self.dataset))
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Any = None
        self.data_val: Any = None

    def setup(self, stage: Any = None) -> None:
        split = [1.0 - self.val_split, self.val_split]
        self.data_train, self.data_val = fractional_random_split(self.dataset, split)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, list):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    print("WandbLogger not found.")
    return None

def log_wandb_audio_batch(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
    logger.log(
        {
            f"sample_{idx}_{id}": wandb.Audio(
                samples[idx],
                caption=caption,
                sample_rate=sampling_rate,
            )
            for idx in range(num_items)
        }
    )

class SampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        use_ema_model: bool,
        num_steps: int = 20
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.use_ema_model = use_ema_model
        self.num_steps = num_steps

        self.log_next = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment

        diffusion_model = pl_module.model
        
        embed = infer_text("Classical piano")
        embed = embed[None, :, :]
        embed = embed.to("cuda:0")

        # Get start diffusion noise
        noise = torch.randn(
            (self.num_items, self.channels, self.length), device=pl_module.device
        )
        # I know, ugly.
        z = autoencoder.encode(noise)
        z = torch.tanh(z)
        
        sampler = DiffusionSampler(
            diffusion=diffusion_model.diffusion,
            num_steps=self.num_steps, # Suggested range 2-100, higher better quality but takes longer
            sampler=diffusion_model.sampler,
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0)
        )
        z_samples = sampler(z, embedding_mask_proba=0.1, embedding=embed)
        z_samples = torch.tanh(z_samples)
        samples = autoencoder.decode(z_samples, num_steps=self.num_steps)

        log_wandb_audio_batch(
            logger=wandb_logger,
            id="sample",
            samples=samples,
            sampling_rate=self.sampling_rate,
            caption=f"Sampled in {self.num_steps} steps",
        )

        if is_train:
            pl_module.train()    

if __name__ == "__main__":
    main()
