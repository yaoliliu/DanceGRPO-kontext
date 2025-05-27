## DanceGRPO
**DanceGRPO is the first unified RL-based framework for visual generation.**

We develop [DanceGRPO](https://arxiv.org/abs/2505.07818) based on FastVideo, a scalable and efficient framework for video and image generation.

## Key Features

DanceGRPO has the following features:
- Support Stable Diffusion
- Support FLUX
- Support HunyuanVideo (todo)


## Getting Started
### Downloading checkpoints
1. Download the Stable Diffusion v1.4 checkpoints from [here](https://huggingface.co/CompVis/stable-diffusion-v1-4) to "./data/stable-diffusion-v1-4".
2. Download the FLUX checkpoints from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev) to "./data/flux".
3. Download the HPS-v2.1 checkpoint (HPS_v2.1_compressed.pt) from [here](https://huggingface.co/xswu/HPSv2/tree/main) to "./hps_ckpt".
4. Download the CLIP H-14 checkpoint (open_clip_pytorch_model.bin) from [here](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main) to "./hps_ckpt".

### Installation
```bash
./env_setup.sh fastvideo
```
### Training
```bash
# for Stable Diffusion, with 8 H800s
bash scripts/finetune/finetune_sd_grpo.sh   
```
```bash
# for FLUX, preprocessing with 8 H800s
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
# for FLUX, training with 16 H800s
bash scripts/finetune/finetune_flux_grpo.sh   
```

### Rewards
We give the (moving average) reward curves of Stable Diffusion (left or upper) and FLUX (right or lower). We can complete the FLUX training (200 iterations) within 12 hours with 16 H800s.

<img src=assets/rewards/opensource_sd.png width="49%">
<img src=assets/rewards/opensource_flux.png width="49%">

We provide more visualization examples (base, 80 iters rlhf, 160 iters rlhf) in "./assets/flux_visualization". The visualization scripts can be find in "./scripts/visualization/vis_flux.py". We always use larger resolutions and more sampling steps than RLHF training for visualization.

More discussion on FLUX can be found in "./fastvideo/README.md".

## Acknowledgement
We learned and reused code from the following projects:
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [diffusers](https://github.com/huggingface/diffusers)
- [DDPO-Pytorch](https://github.com/kvablack/ddpo-pytorch)


## Citation
If you use DanceGRPO for your research, please cite our paper:

```bibtex
@article{xue2025dancegrpo,
  title={DanceGRPO: Unleashing GRPO on Visual Generation},
  author={Xue, Zeyue and Wu, Jie and Gao, Yu and Kong, Fangyuan and Zhu, Lingting and Chen, Mengzhao and Liu, Zhiheng and Liu, Wei and Guo, Qiushan and Huang, Weilin and others},
  journal={arXiv preprint arXiv:2505.07818},
  year={2025}
}
```
