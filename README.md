# Versatile Audio Generation with Diffusion and GANs

## Introduction

This repository summarizes the results of training and working with four different models in the audio generation space. This document will highlight their challenges and capabilities. The models were trained on a corpus of piano music to generate waveform (.wav) files. The chosen model for the Technigala demo is [Musika](https://github.com/marcoppasini/musika), which combines an adversarial autoencoder with a GAN using a latent coordinate system. I also discuss the other three models (or more accurately, their training frameworks): [audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch), Huggingface’s [diffusers](https://github.com/huggingface/diffusers) library, and [RAVE](https://github.com/acids-ircam/RAVE/tree/master/rave) (Realtime Variational Autoencoder). The first two models are diffusion models, which leverage denoising techniques to generate high-quality audio waveforms by sampling from a latent space. In contrast, RAVE introduces a novel approach to training variational autoencoders, resulting in enhanced compression capabilities and improved sample quality.

This project is a **work in progress**. None of these models (with the exception of fine-tuned Musika) could be fully trained to satisfactory results. Ultimately, more time or computational resources will be needed to yield results closer to SOTA model benchmarks in the audio generation domain. I would like to continue developing these models further throughout the next few months to achieve more promising results.

## Objective

The original goal for this project was to create a versatile, effective, and lightweight text-to-audio generation model for pop music. This project was inspired by [MusicLM](https://google-research.github.io/seanet/musiclm/examples/) – a SOTA Google text-to-audio model released during the class – and [AudioLDM](https://arxiv.org/pdf/2301.12503.pdf), which enables quick audio sample generation of significant duration on a single consumer GPU. At first, I intended to replicate MusicLM or Jukebox with a smaller dataset, but quickly realized that the computational requirements far exceeded any resources I had access to – [MuLan](https://arxiv.org/pdf/2208.12415.pdf), the base model for both MusicLM and [AudioLM](https://arxiv.org/abs/2209.03143), was trained on 370K hours of audio recordings. Jukebox required several A100 GPUs to get even a few seconds of audio. 

I was therefore inspired by the diffusion model approach and decided to explore other methods that could potentially achieve high-quality audio generation with more manageable computational requirements. By focusing on diffusion models, GANs, and novel approaches like RAVE, I aimed to create a more accessible and efficient audio generation model that could be trained and utilized by a broader audience.

The objective of this project shifted towards finding a suitable model that could generate high-quality audio samples (conditionally or unconditionally) while maintaining the balance between computational efficiency and performance. Moreover, I shifted to using piano music for training (instead of my original intention of generating pop music), as a proof of concept due to the lesser complexity of modeling a single-instrument track.

## Background and Results

Audio generation has been an active area of research in recent years, with significant advancements being made in generating high-quality, realistic audio samples. Early attempts to generate audio were primarily based on rule-based systems, followed by the introduction of statistical models such as Hidden Markov Models (HMMs).

Generative Adversarial Networks (GANs) and diffusion models have recently emerged as promising techniques for audio generation. GANs consist of a generator and a discriminator network that work together in a competitive fashion, leading to the generation of high-quality, realistic audio samples. Diffusion models, on the other hand, use denoising techniques to generate audio samples by iteratively refining a noisy input until it closely resembles the target distribution.

### Diffusion Models

The most promising model in the audio diffusion space is [AudioLDM](https://github.com/haoheliu/AudioLDM). They have a package that's installable from `pip` and a [Huggingface Spaces](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation) demo of relatively high-quality generation of long audio samples (up to around 60 seconds), with fidelity to the original prompt. The generation is fast and can occur on a single GPU. I considered attempting to adapt this model for my final project, but the training code hasn't been provided by the authors (waiting until after their paper acceptance). Though the rest of the model code has been open-sourced, and given more time I think I could figure out a training strategy, I chose to attempt other models with more robust open-source support. The major innovation advanced by the AudioLDM paper, however, was using [CLAP](https://github.com/LAION-AI/CLAP) (Contrastive Language-Audio Pretraining) to associate embeddings with audio samples in a pre-trained text-audio latent space without relying on costly manual labeling of samples to train text-conditional models. I attempted to employ this approach for the Moûsai/ArchiSound model, as I document below.

The model I spent the most time with was the [Moûsai model](https://arxiv.org/pdf/2301.11757v2.pdf). The training repository is available [here](https://github.com/archinetai/audio-diffusion-pytorch-trainer) and model code is available [here](https://github.com/archinetai/audio-diffusion-pytorch). The main issue was the generation of samples with significant background noise and only vague resemblance to the original piano samples. More training didn't resolve this problem, even after tens of thousands of training steps. This may be attributed to the choice of architecture, hyperparameters, or the training strategy employed. In particular, most examples of successful training with this model that I found online did not employ the autoencoder, but in doing so either made sacrifices in sample length or quality or utilized computational resources beyond what I have access to. The model code also changed fairly drastically between versions 0 and 1 of the package, which made it more difficult to train than expected (as class names, interfaces, etc. changed) and meant that some hyperparameters could have possibly been thrown off in the diffusion or sampling process. The unique innovation of this model was the use of a diffusion magnitude autoencoder (DMAE) instead of a typical mel-spectrogram VAE, which enables a much more significant degree of compression (32-64x) directly from the original audio samples using diffusion.

The other diffusion model that I worked with was Huggingface's [diffusers](https://github.com/huggingface/diffusers) library, which is a more general-purpose diffusion framework that can be adapted for various applications, including audio generation (the bulk of the library is dedicated to image generation with diffusion models). While working with this model, I faced similar issues as with the Moûsai model – the generated samples had substantial background noise and only faint resemblance to the original piano samples. The challenges in training this model might be due to a suboptimal selection of hyperparameters, the architecture, or issues with the training process itself. Despite the promise of the diffusion-based approach, it became clear that obtaining satisfactory results with the computational resources and time available would be difficult.

The original source code for the audio diffusion component I worked with is available [here](https://github.com/teticio/audio-diffusion). From my understanding, this is the architecture that most closely resembles the AudioLDM approach. Waveforms are converted to 512x512 mel-spectrograms, which are further compressed into latent space by a pre-trained VAE. A [2-dimensional UNet](https://huggingface.co/docs/diffusers/api/models#diffusers.UNet2DModel) model is then trained on the compressed latent representations, and a vocoder is used to convert the decoded samples back to waveforms. The `diffusers` library uses an iterative Griffin-Lim approach instead of a pre-trianed vocoder like [HiFi-GAN](https://github.com/jik876/hifi-gan).

### RAVE

[RAVE](https://arxiv.org/abs/2111.05011) (Realtime Variational Autoencoder) is a novel approach for training variational autoencoders, aimed at improving sample quality and compression capabilities. Some of the pretrained models exhibited the ability to compress a waveform up to 2048x in latent space while maintaining fidelity to the original sample after being decoded. However, I ultimately didn't have enough GPU hours to complete training of the model (though some samples are available in the `samples` directory and through the W&B link below). Further investigation and experimentation with different hyperparameters, architectures, or training strategies could potentially lead to better results. However, given the time constraints, it was not possible to fully explore and optimize the RAVE model during the course of this project.

I am optimistic that given the powerful compression and representation capabilities demonstrated by the model, it can be used as a universal input into other diffusion models – such as the ones listed above – that rely on a trained latent space. Since the RAVE model is specifically trained during the second stage to produce random samples that stylistically resemble the original training set.

### GANs and Transformers

Among the four models, [Musika](https://github.com/marcoppasini/musika) demonstrated the most promising results. It combines an adversarial autoencoder with a GAN using a latent coordinate system, which helps generate high-quality audio samples with relatively lower computational requirements. The final model could produce a 2 minute audio track in under 10 seconds! The model was quick to train on a single GPU, and produced audio samples *much* faster than any of the other models. This speed increase is likely due to Musika being a GAN, which only requires one forward pass as opposed to the iterative denoising process required by diffusion. Similar improvements in speed and quality can be observed with [SOTA text-to-image generation models](https://mingukkang.github.io/GigaGAN/).

The fine-tuned Musika model produced audio samples that closely resembled the original piano pieces, with minimal background noise and artifacts. Although the model is not yet perfect, it shows great potential for further improvements and has been chosen as the primary model for the Technigala demo.

The Google-developed transformer models ([MusicLM](https://arxiv.org/abs/2301.11325) and [AudioLM](https://arxiv.org/abs/2209.03143)) also seem as though they can produce good results, but they seem to require far more computing resources to train that I have available (likely due to being transformer-based). Similarly, Jukebox apparently requires an A100 GPU to just run and even then can only generate a few seconds of audio at a time. The advantage with the diffusion model approach is that due to the compression provided by the autoencoder, training and inference can theoretically occur on a single consumer GPU. Though the official training code for these models aren't available online, [PyTorch implementations](https://github.com/lucidrains/musiclm-pytorch) are available.

## Dataset and Resources

I trained each model on a corpus of piano music. Specifically, I use the freely available [MAESTRO V3 dataset](https://magenta.tensorflow.org/datasets/maestro#v300) – a compilation of recordings from an e-piano competition – and the [ADL Piano MIDI database](https://github.com/lucasnfe/adl-piano-midi). The latter was converted from .wav files to MIDI files using the script `convert_midi_to_wav.py` in the `/src` directory, using the `midi2audio` package.

Other resources used include:
- Heavy use of Huggingface's [transformers](https://github.com/huggingface/transformers) and [diffusers](https://github.com/huggingface/diffusers) libraries.
- [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/) for training, running, and loading models.
- [Wandb](https://wandb.ai/) for logging and visualizing training metrics.
- [Kaggle](https://www.kaggle.com/) and [Google Colab](https://colab.research.google.com/) for training models on free GPU hours.
- [Google Cloud Storage](https://cloud.google.com/storage) for storing and sharing large files, along with Google Cloud's free credits.

## Code and Training Strategy

The training for RAVE, the `diffusers` model, and Musika was largely conducted using free GPU hours provided by Kaggle notebooks and Google Colab. The training code for Musika is provided in the `/src` directory. It is designed to be run start-to-finish in a Kaggle environment using two T4 GPUs, and has not been tested elsewhere. The dataset (pre-encoded Musika encodings on the MAESTRO datset) for Musika is available [here](https://www.kaggle.com/datasets/trevorvireo3221/musika-encodings). 

The RAVE trainer can also be run from a Google Colab, which is available below. Note that the RAVE trainer is designed to be run with pre-processed RAVE files in a mounted Google Drive directory.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A376LMprvKI3PUiz0mAjpTEipeL0H2B2#scrollTo=mF4R7GAWVOZF)

The `diffusers` model can also be trained through a Colab notebook, which is available below. It is accompanied by the dataset used for training, which was generated using the `audio_to_images.py` script from the [audiodiffusion repo](https://github.com/teticio/audio-diffusion/tree/main). It is available on Huggingface [here](https://huggingface.co/datasets/tvergho/maestro).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/175vjhfPWsY7cVY5Pg1X5NIkTwuluBalh?usp=sharing)

The Moûsai model could not fit on free Colab/Kaggle 16GB GPUs for training, so it was trained using an RTX 3090 GOU with 24GB of VRAM on [Runpod](https://runpod.io/). This was trained using the script `train_mousai.py` in the `/src` directory. The notebook `setup.ipynb` was used to set up the environment on each instance prior to training.

## Discussion and Summary

This project proved an in-depth, challenging exploration of training audio generation models and the current state of audio generation research. The diffusion models in particular were difficult to optimize and train successfully. There is a need for better training strategies and more accessible models that can be deployed by a wider range of researchers and practitioners.

Going forward, I would like to continue exploring and refining these models, particularly focusing on the diffusion models and RAVE, to achieve more promising results. By fine-tuning the models, experimenting with different hyperparameters, and incorporating novel approaches like CLAP, it may be possible to develop a more versatile, efficient, and high-quality text-to-audio generation model that anyone can use.

In the long term, the goal is to create a lightweight, effective text-to-audio generation model for various music genres that can be trained and utilized by a broad audience. This will not only contribute to the ongoing research in the field of audio generation, but also enable new creative applications and possibilities in music and beyond.

### Next Steps
- [ ] Finish training the RAVE model (both stages), to verify my theory that it can be used as a more efficient univeral autoencoder for diffusion models.
- [ ] Successfully train a text-conditional diffusion model on a limited piano dataset.
- [ ] Successfully train a multi-genre model, or audio model that extends beyond just music samples.
- [ ] Make the generation/inference process runnable on a single CPU/GPU.

## Training Runs and Model Weights

The best place to view output from a sample training run of the *Moûsai/ArchiSound* model is [this Weights and Biases dashboard](https://wandb.ai/tvergho1/audio-diffusion/reports/Training-Report-Archisound-Mousai--VmlldzozNzE4NzI3?accessToken=smieeukxyk6p3waw9cjt2d02z8q827bmc182joebysbwjn14q6sq3jsm67ethfv6), which corresponds to the most complete training run of the conditional text model on the MAESTRO v3 dataset. An aggregated compilation of all the training runs can be viewed [here](https://wandb.ai/tvergho1/audio-diffusion). 

A compiliation of all the training runs for the RAVE model can be accessed [here](https://wandb.ai/tvergho1/rave) on W&B. The best training run is summarized [here](https://wandb.ai/tvergho1/rave/reports/Training-Report-RAVE--VmlldzozNzE4OTQ5?accessToken=lv8i5rtsmlcgjmh4mfyo4a56cjx01ybo6tx8gv6jlukjebmqj0vh78a7j2psge1e). The training of this model is still in progress – I haven't even gotten to the second stage (tuning the discriminator).

The most recent version of the trained *diffusers* model was trained using a Huggingface library and so can be [downloaded from Huggingface](https://huggingface.co/tvergho/audio-diffusion-512). The final fine-tuned Musika model is available through [this Google Drive link](https://drive.google.com/drive/u/0/folders/1WzANXTaDXb33wbfCdKTuEpSyGicdJ8Gm).

## Comparison of Text-To-Audio Models

| Model Name       | Model Type | Input                           | Output                    | Training             | Components                                      |
|------------------|------------|---------------------------------|---------------------------|----------------------|-------------------------------------------------|
| [Moûsai](https://arxiv.org/abs/2301.11757)           | Diffusion  | Text                            | Music (any genre)         | Text + audio pairs   | Vocoder, Spectrogram autoencoder, UNet latent diffusion, T5 text embeddings |
| [AudioLDM](https://arxiv.org/pdf/2301.12503.pdf)         | Diffusion  | Text                            | Music (any genre), sound effects | Audio only | CLAP, UNet latent diffusion, Variational autoencoder, Vocoder |
| [noise2music](https://google-research.github.io/noise2music/noise2music.pdf)      | Diffusion  | Text                            | Music (any genre)         | Text + audio         | UNet diffusion, Waveform generator/cascader, Spectrogram generator/vocoder |
| [Make-An-Audio](https://text-to-audio.github.io/paper.pdf)    | Diffusion  | Text + optional audio (or image/video) | Any sound | Audio (with generated captions) | CLAP or T5, Spectrogram autoencoder, Latent diffusion UNet text-conditional DDPM, Vocoder (HiFi-GAN) |
| [AudioLM](https://arxiv.org/abs/2209.03143)          | Transformer | Audio                          | Piano/speech              | Text + audio         | Soundstream (acoustic tokenizer), w2v-BERT (semantic tokenizer), Coarse transformer, Fine transformer, T5 conditioner (optional) |
| [MusicLM](https://arxiv.org/abs/2301.11325)          | Transformer | Text (and optional melody)     | Music (any genre)         | Text + audio (only for MuLan) | MuLan, Soundstream, w2v-BERT, Coarse acoustic model, Fine acoustic model |
| [Jukebox](https://jukebox.openai.com/)          | Transformer | Text + lyrics                  | Music                     | Text + metadata (artist, genre, lyrics) | VQ-VAE, Transformer, Upsampler |
