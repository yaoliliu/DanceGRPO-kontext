## DanceGRPO
**DanceGRPO is the first unified RL-based framework for visual generation.**

This is the official implementation for [paper](https://arxiv.org/abs/2505.07818), DanceGRPO: Unleashing GRPO on Visual Generation.
We develop [DanceGRPO](https://arxiv.org/abs/2505.07818) based on FastVideo, a scalable and efficient framework for video and image generation.

## Key Features

DanceGRPO has the following features:
- Support Stable Diffusion
- Support FLUX
- Support HunyuanVideo (todo)


## Getting Started
### Downloading checkpoints
You should use ```"mkdir"``` for these folders first.
1. Download the Stable Diffusion v1.4 checkpoints from [here](https://huggingface.co/CompVis/stable-diffusion-v1-4) to ```"./data/stable-diffusion-v1-4"```.
2. Download the FLUX checkpoints from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev) to ```"./data/flux"```.
3. Download the HPS-v2.1 checkpoint (HPS_v2.1_compressed.pt) from [here](https://huggingface.co/xswu/HPSv2/tree/main) to ```"./hps_ckpt"```.
4. Download the CLIP H-14 checkpoint (open_clip_pytorch_model.bin) from [here](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main) to ```"./hps_ckpt"```.

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

For open-source version, we use the prompts in [HPD](https://huggingface.co/datasets/ymhao/HPDv2/tree/main) dataset for training, as shown in ```"./prompts.txt"```.

### Rewards
We give the (moving average) reward curves (also the results in `reward.txt` or `hps_reward.txt`) of Stable Diffusion (left or upper) and FLUX (right or lower). We can complete the FLUX training (200 iterations) within 12 hours with 16 H800s.

<img src=assets/rewards/opensource_sd.png width="49%">
<img src=assets/rewards/opensource_flux.png width="49%">

We provide more visualization examples (base, 80 iters rlhf, 160 iters rlhf) in ```"./assets/flux_visualization"```. We always use larger resolutions and more sampling steps than RLHF training for visualization, because we use lower resolutions and less sampling steps for speeding up the RLHF training.

Here is the visualization script `"./scripts/visualization/vis_flux.py"` for FLUX. First, run `rm -rf ./data/flux/transformer/*` to clear the directory, then copy the files from a trained checkpoint (e.g., `checkpoint-160-0`) into `./data/flux/transformer`. After that, you can run the visualization. If it's trained for 160 iterations, the results are already provided in my repo.  

We don't recommend using 8 H800s for the FLUX training script, because we find a global prompt batch size of 8 is not enough.

More discussion on FLUX can be found in ```"./fastvideo/README.md"```.

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
