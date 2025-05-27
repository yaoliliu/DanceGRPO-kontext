# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import argparse
from email.policy import strict
import logging
import math
import os
import shutil
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper, broadcast
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.utils.validation import log_validation
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
#from torch.distributed.fsdp.state_dict 
import json
from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers import FlowMatchEulerDiscreteScheduler
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformer3DModel
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from safetensors.torch import save_file, load_file
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
    resume_lora_optimizer,
)
from fastvideo.utils.logging_ import main_print
from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline
from fastvideo.models.reward_model_altclip import AltClip
#from fastvideo.models.reward_model.viclip import ViCLIP
#from fastvideo.models.reward_model.simple_tokenizer import SimpleTokenizer as _Tokenizer
#from fastvideo.models.reward_model.inference import (get_clip, 
#                                                     get_text_feat_dict, 
#                                                     get_vid_feat, 
#                                                     frames2tensor,
#                                                     _frame_from_video)
import cv2
import scenedetect
from diffusers.video_processor import VideoProcessor

# 创建一个视频场景检测器
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector
from fastvideo.utils.load import load_vae
#from fastvideo.ppo_utils import flux_step

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")
import time
from collections import deque
import random
import numpy as np
from einops import rearrange
from moviepy.editor import ImageSequenceClip
from diffusers.utils import export_to_video
import torch.distributed as dist
from torch.nn import functional as F
from transformers import AutoProcessor
#from Mantis.mantis.models.idefics2 import Idefics2ForSequenceClassification
import av
from typing import List
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from decord import cpu, VideoReader, bridge
import io
#from PIL import Image
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)

def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break
    # print(type(frame_id_list[0]))
    video_data = decord_vr.get_batch(frame_id_list)
    # print(type(video_data[0]))
    video_data = video_data.permute(3, 0, 1, 2)
    # print(video_data.shape)
    return video_data

def inference(tokenizer, model, video_path, query, temperature=0.1):
    TORCH_TYPE = torch.bfloat16

    video_data = open(video_path, 'rb').read()              
    strategy = 'chat'
    video = load_video(video_data, strategy=strategy)
    
    history = []

    # 确保使用正确的token大小写，例如小写"yes"
    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]  # 获取"yes"的token ID

    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 1,  # 只生成一个token
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else 128002,
        "output_scores": True,
        "return_dict_in_generate": True,
        "temperature": temperature,
        "do_sample": False,  # 保持与原始代码一致，但概率计算仍受temperature影响
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        # 获取第一个生成步骤的logits
        logits = outputs.scores[0][0]  # [vocab_size]
        # 应用softmax计算概率
        probs = torch.softmax(logits, dim=-1)
        yes_prob = probs[yes_token_id].item()
    
    return yes_prob

def score(args,device,tokenizer, model, video_path, prompt) -> float:
    queries = [question.replace('[[prompt]]', prompt) for question in questions]
    yes_probs = []   
        # batch_reward = torch.tensor(np.mean(batch_reward * weight)).to(device)
    if args.use_tuwen:
        queries = [queries[0]]
    elif args.use_aes:
        queries = [queries[3]]
    elif args.use_aes_tuwen:
        queries = [queries[0], queries[3]]
            
    for query in tqdm(queries, desc='scoring video'):
        yes_prob= inference(tokenizer, model, video_path, query)
        yes_probs.append(yes_prob)
    # 将yes的概率转换为得分：2*p -1（范围从-1到1）
    batch_reward = np.array(yes_probs)
    if args.use_all_qs:
        batch_reward = torch.tensor(np.mean(batch_reward * weight)).to(device)
    elif args.use_aes_tuwen:
        batch_reward = torch.tensor(batch_reward[0]*0.5+batch_reward[1]).to(device)
    else:
        batch_reward = torch.tensor(batch_reward[0]).to(device)
    # 加权平均
    #weighted_scores = scores * weight
    return batch_reward

def _read_video_pyav(
    container: av.container.Container,
    indices: List[int],
):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

##pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), ddpo=True, sde_solver=True
def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    ddpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean - log_term * delta_t

    if ddpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        

    if ddpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = (
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob_first_frame = (
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))

        # mean along all but batch dimension
        log_prob_first_frame = log_prob_first_frame.mean(dim=tuple(range(1, log_prob_first_frame.ndim)))

        return prev_sample, pred_original_sample, log_prob, log_prob_first_frame
    else:
        return prev_sample_mean,pred_original_sample



def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"

STATS = {
    "mean": torch.Tensor(
        [
            -0.06730895953510081,
            -0.038011381506090416,
            -0.07477820912866141,
            -0.05565264470995561,
            0.012767231469026969,
            -0.04703542746246419,
            0.043896967884726704,
            -0.09346305707025976,
            -0.09918314763016893,
            -0.008729793427399178,
            -0.011931556316503654,
            -0.0321993391887285,
        ]
    ),
    "std": torch.Tensor(
        [
            0.9263795028493863,
            0.9248894543193766,
            0.9393059390890617,
            0.959253732819592,
            0.8244560132752793,
            0.917259975397747,
            0.9294154431013696,
            1.3720942357788521,
            0.881393668867029,
            0.9168315692124348,
            0.9185249279345552,
            0.9274757570805041,
        ]
    ),
}

def dit_latents_to_vae_latents(latents: torch.Tensor) -> torch.Tensor:
    """Unnormalize latents output by Mochi's DiT to be compatible with VAE.
    Run this on sampled latents before calling the VAE decoder.

    Args:
        latents (torch.Tensor): [B, C_z, T_z, H_z, W_z], float

    Returns:
        torch.Tensor: [B, C_z, T_z, H_z, W_z], float
    """
    latents_mean = (
        torch.tensor(STATS["mean"])
        .view(1, 12, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        torch.tensor(STATS["std"])
        .view(1, 12, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = (
        latents * latents_std + latents_mean
    )
    return latents


def save_video(final_frames, output_path, fps=30):
    assert final_frames.ndim == 4 and final_frames.shape[3] == 3, f"invalid shape: {final_frames} (need t h w c)"
    if final_frames.dtype != np.uint8:
        final_frames = (final_frames * 255).astype(np.uint8)
    ImageSequenceClip(list(final_frames), fps=fps).write_videofile(output_path)

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states,
        encoder_attention_mask,
        ddpo_sample,
        empty_cond_hidden_states,
        empty_cond_attention_mask,
    ):
    if ddpo_sample:
        all_latents = [z]
        all_log_probs = []
        all_log_prob_first_frame = []
        for i in progress_bar:  # Add progress bar
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            #dsigma = sigma_schedule[i + 1] - sigma
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            #with torch.no_grad():
            transformer.eval()
            with torch.autocast("cuda", torch.bfloat16):
                model_pred= transformer(
                    hidden_states=z,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    guidance=torch.tensor(
                        [6018.0],
                        device=z.device,
                        dtype=torch.bfloat16
                    ),
                    encoder_attention_mask=encoder_attention_mask,  # B, L
                    return_dict=False,
                )[0]
            #pred  =  uncond_pred.to(torch.float32) + args.cfg_infer * (model_pred.to(torch.float32) - uncond_pred.to(torch.float32))
            pred = model_pred
            #z = z + dsigma * pred
            z, pred_original, log_prob, log_prob_first_frame = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, ddpo=True, sde_solver=True)
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
            all_log_prob_first_frame.append(log_prob_first_frame)
        latents = pred_original.to(torch.float32)/0.476986
        #latents = dit_latents_to_vae_latents(pred_original)
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        all_log_prob_first_frame = torch.stack(all_log_prob_first_frame, dim=1)  # (batch_size, num_steps, 1)
        return z, latents, all_latents, all_log_probs, all_log_prob_first_frame

        
def ddpo_one_step(
            args,
            latents,
            pre_latents,
            encoder_hidden_states,
            encoder_attention_mask,
            empty_cond_hidden_states,
            empty_cond_attention_mask,
            transformer,
            timesteps,
            i,
            sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    with torch.autocast("cuda", torch.bfloat16):
        transformer.train()
        model_pred= transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            guidance=torch.tensor(
                [6018.0],
                device=latents.device,
                dtype=torch.bfloat16
            ),
            encoder_attention_mask=encoder_attention_mask,  # B, L
            return_dict=False,
        )[0]
    #pred  =  uncond_pred.to(torch.float32) + args.cfg_infer * (model_pred.to(torch.float32) - uncond_pred.to(torch.float32))
    #z = z + dsigma * pred
    pred = model_pred
    z, pred_original, log_prob, log_prob_first_frame = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), ddpo=True, sde_solver=True)
    return log_prob, log_prob_first_frame

from scenedetect.detectors import ContentDetector, HistogramDetector

def detect_video_cuts(video_path, threshold=50.0):
    """
    检测视频中是否存在帧跳变。
    
    参数:
    video_path: 视频文件的路径
    threshold: 跳变检测阈值，默认20.0
    
    返回:
    True 如果检测到跳变；False 如果未检测到
    """
    cap = cv2.VideoCapture(video_path)
    ret, last = cap.read()
    if not ret:
        print("无法读取视频")
        return False

    # 转为灰度图
    last_gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, last_gray)
        score = np.mean(diff)
        
        if score > threshold:
            cap.release()
            print("find cuts, reward=0")
            return True

        last_gray = gray.copy()

    cap.release()
    return False


def video_first_frame_to_pil(video_path):
    # 使用OpenCV打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return None

    # 读取视频的第一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频的第一帧")
        cap.release()
        return None

    # 将OpenCV的BGR格式转换为RGB格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 将numpy数组转换为PIL图像
    pil_image = Image.fromarray(frame_rgb)

    # 释放视频文件
    cap.release()

    return pil_image


def sample_reference_model(
    args,
    step,
    device, 
    transformer,
    #reference_transformer, 
    vae,
    encoder_hidden_states, 
    encoder_attention_mask,
    empty_cond_hidden_states,
    empty_cond_attention_mask,
    reward_model,
    tokenizer,
    inferencer,
    preprocess_dgn5b,
    caption,
    preprocess_val,
    noise_scheduler,
    eval_mode=False,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(42+dist.get_rank()+step)

    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    #sigma_schedule = linear_quadratic_schedule(sample_steps, 0.025)
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1  # 可以根据你的显存情况调整这个值
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    all_latents = []
    all_log_probs = []
    all_rewards = []  # 用于存储所有批次的奖励
    all_log_prob_first_frame = []
    all_video_rewards = []
    if args.use_same_noise:
        input_latents = torch.randn(
                (1, IN_CHANNELS, latent_t, latent_h, latent_w),  #（c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )
    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_encoder_attention_mask = encoder_attention_mask[batch_idx]
        #if dist.get_rank() == 0:
        #    breakpoint()
        #dist.barrier()
        #batch_empty_cond_hidden_states = empty_cond_hidden_states[batch_idx]
        #batch_empty_cond_attention_mask = empty_cond_attention_mask[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        if not args.use_same_noise:
            input_latents = torch.randn(
                (len(batch_idx), IN_CHANNELS, latent_t, latent_h, latent_w),  #（c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )
        ddpo_sample=True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs, batch_log_prob_first_frame = run_sample_step(
                args,
                input_latents.clone(),
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_encoder_attention_mask,
                ddpo_sample,
                empty_cond_hidden_states,
                empty_cond_attention_mask,
            )
        
        # 累积所有批次的latents和log_probs
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        all_log_prob_first_frame.append(batch_log_prob_first_frame)
        vae.enable_tiling()
        
        video_processor = VideoProcessor(
            vae_scale_factor=8)
        
        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video = vae.decode(latents, return_dict=False)[0]
                videos = video_processor.postprocess_video(
                video)

                video_first_frame = vae.decode(latents[:,:,0].unsqueeze(2), return_dict=False)[0]
                videos_first_frame = video_processor.postprocess_video(
                        video_first_frame
                )
                #videos = (decode_latents_tiled_full(vae.decoder, latents) * 0.5 + 0.5).clamp(0, 1)
                #videos = (vae.decoder(latents[:,:,0].unsqueeze(2))* 0.5 + 0.5).clamp(0, 1)
        rank = int(os.environ["RANK"])
                
            
        from diffusers.utils import export_to_video
        
        export_to_video(videos[0], f"./videos/mochi_{rank}_{index}.mp4", fps=args.fps)
        export_to_video(videos_first_frame[0], f"./videos/mochi_{rank}_{index}_first.mp4", fps=args.fps)
                
        #videos = torch.from_numpy((videos * 0.5 + 0.5)).clamp(0, 1).to(device)       
        
        #save_video(
        #    videos[0].float().cpu().numpy(),
        #    f"./videos/mochi_{rank}_{index}.mp4",
        #    fps=args.fps,
        #)
        if args.use_hpsv2:
            with torch.no_grad():
                image_path = video_first_frame_to_pil(f"./videos/mochi_{rank}_{index}_first.mp4")
                image = preprocess_val(image_path).unsqueeze(0).to(device=device, non_blocking=True)
                # Process the prompt
                text = tokenizer([batch_caption[0]]).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.amp.autocast('cuda'):
                    outputs = reward_model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image)
                all_rewards.append(hps_score)

        if args.use_videoalign:
            with torch.no_grad():
                try:
                    #print("starting video align")
                    video_rewards = inferencer.reward(
                        [f"/opt/tiger/Fastvideo/videos/mochi_{rank}_{index}.mp4"],
                        [batch_caption[0]],
                        use_norm=True,
                    )
                    if args.use_videoalign_mq:
                        video_reward = torch.tensor(video_rewards[0]['MQ']).to(device)
                    elif args.use_videoalign_vq:
                        video_reward = torch.tensor(video_rewards[0]['VQ']).to(device)
                    elif args.use_videoalign_overall:
                        video_reward = torch.tensor(video_rewards[0]['Overall']).to(device)
                    elif args.use_videoalign_vq_mq:
                        video_reward = torch.tensor(video_rewards[0]['VQ']+video_rewards[0]['MQ']).to(device)
                    all_video_rewards.append(video_reward.unsqueeze(0))
                except Exception as e:
                    video_reward = torch.tensor(-1.0).to(device)
                    all_video_rewards.append(video_reward.unsqueeze(0))


        if args.use_altclip:
            with torch.no_grad():
                cap = cv2.VideoCapture(f"./videos/mochi_{rank}_{index}.mp4")
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 将像素值范围从 [0,255] 转换到 [0,1]
                frame = frame.astype(np.float32) / 255.0

                # 将 NumPy 数组转换为 PyTorch 张量
                image = torch.tensor(frame).permute(2, 0, 1).to(device)
                results = []
                cap.release()
                n_frames = 1  # 获取视频帧数
                # 针对每个视频计算奖励：每两帧采样一次
                for i in range(len(videos)):
                    rewards = []
                    # 遍历每两帧采样一次，步长设为 2
                    # 注意：使用 range(0, n_frames, 2) 时最后一个索引为 n_frames-1 或更小，不会越界
                    #for j in range(0, n_frames):
                    reward = reward_model.compute_reward(batch_caption[i], image, device)
                    rewards.append(reward)
                    # 计算所有采样帧奖励的平均值作为该视频的奖励
                    avg_reward = torch.mean(torch.stack(rewards), dim=0)
                    results.append(avg_reward)
                # 将每个视频的奖励拼接成一个 batch 的 tensor
                batch_reward = torch.cat(results, dim=0)
                if detect_video_cuts(f"./videos/mochi_{rank}_{index}.mp4") and args.use_videocuts:
                    all_rewards.append(batch_reward-0.07)
                else:
                    all_rewards.append(batch_reward)  # 添加每个批次的奖励
        
        
        
        if args.use_visionreward:
            with torch.no_grad():
                batch_reward = score(args, device, tokenizer, reward_model, f"./videos/mochi_{rank}_{index}.mp4", batch_caption[0])
                all_rewards.append(batch_reward.unsqueeze(0))

        if args.use_clipscore:
            with torch.no_grad():
                image_path = video_first_frame_to_pil(f"./videos/mochi_{rank}_{index}_first.mp4")
                image = preprocess_dgn5b(image_path).unsqueeze(0).to(device=device, non_blocking=True)
                # Process the prompt
                text = tokenizer([batch_caption[0]], context_length=inferencer.context_length).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.amp.autocast('cuda'):
                    image_features = inferencer.encode_image(image)
                    text_features = inferencer.encode_text(text)
                    image_features = F.normalize(image_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)
                    hps_score = image_features @ text_features.T 
                all_video_rewards.append(hps_score.squeeze(0))


    # 将所有批次的latents、log_probs和rewards堆叠起来
    #if dist.get_rank()==0:
    #    breakpoint()
    #dist.barrier()

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_log_prob_first_frame = torch.cat(all_log_prob_first_frame, dim=0)
    all_rewards = torch.cat(all_rewards, dim=0)
    all_video_rewards = torch.cat(all_video_rewards, dim=0)
    #if dist.get_rank() == 0:
     #   breakpoint()
    #dist.barrier()
    #if not eval_mode:

    
    return videos, z, all_rewards, all_latents, all_log_probs, sigma_schedule, all_log_prob_first_frame, all_video_rewards

def pad_to_square(tensor, padding_value=0):
    ndim = tensor.ndim
    tensor = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
    batch_size, channels, height, width = tensor.size()
    pad_h = max(height, width) - height
    pad_w = max(height, width) - width

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_tensor = torch.nn.functional.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=padding_value)
    if ndim == 3:
        padded_tensor = padded_tensor.squeeze(0)
    return padded_tensor



def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size,),
            device="cpu",
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu", generator=generator)
    return u


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_one_step(
    args,
    device,
    transformer,
    #reference_transformer,
    vae,
    reward_model,
    tokenizer,
    inferencer,
    preprocess_dgn5b,
    empty_cond_hidden_states,
    empty_cond_attention_mask,
    model_type,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    noise_random_generator,
    gradient_accumulation_steps,
    sp_size,
    precondition_outputs,
    max_grad_norm,
    weighting_scheme,
    logit_mean,
    logit_std,
    mode_scale,
    step,
    preprocess_val,
):
    total_loss = 0.0
    optimizer.zero_grad()
    (
        #latents,
        encoder_hidden_states,
        #latents_attention_mask,
        encoder_attention_mask,
        caption,
    ) = next(loader)
    #device = latents.device
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            # 在dim=0维度重复每个元素args.num_generations次
            # 例如形状 [a,b] -> [a*args.num_generations, b]
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        encoder_attention_mask = repeat_tensor(encoder_attention_mask)

        # 处理字符串类型（每个元素重复args.num_generations次）
        if isinstance(caption, str):
            # 单个字符串转换为列表后重复
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            # 列表中每个元素重复args.num_generations次
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    empty_cond_hidden_states = empty_cond_hidden_states.unsqueeze(0)
    empty_cond_attention_mask = empty_cond_attention_mask.unsqueeze(0)
    videos, latents, reward, all_latents, all_log_probs, sigma_schedule, all_log_prob_first_frame, all_video_rewards = sample_reference_model(
            args,
            step,
            device, 
            transformer,
            #reference_transformer, 
            vae,
            encoder_hidden_states, 
            encoder_attention_mask, 
            empty_cond_hidden_states,
            empty_cond_attention_mask,
            reward_model,
            tokenizer,
            inferencer,
            preprocess_dgn5b,
            caption,
            preprocess_val,
            noise_scheduler,
            eval_mode=False,
        )
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)
    if args.ignore_last:
        samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "all_log_prob_first_frame": all_log_prob_first_frame[:, :-1],
        "rewards": reward.to(torch.float32),
        "video_rewards": all_video_rewards.to(torch.float32),
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "empty_cond_hidden_states": empty_cond_hidden_states.repeat(batch_size, 1, 1),
        "empty_cond_attention_mask": empty_cond_attention_mask.repeat(batch_size, 1),
    }
    else:  
        samples = {
            "timesteps": timesteps.detach().clone(),
            "latents": all_latents[
                :, :-1
            ],  # each entry is the latent before timestep t
            "next_latents": all_latents[
                :, 1:
            ],  # each entry is the latent after timestep t
            "log_probs": all_log_probs,
            "all_log_prob_first_frame": all_log_prob_first_frame,
            "rewards": reward.to(torch.float32),
            "video_rewards": all_video_rewards.to(torch.float32),
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "empty_cond_hidden_states": empty_cond_hidden_states.repeat(batch_size, 1, 1),
            "empty_cond_attention_mask": empty_cond_attention_mask.repeat(batch_size, 1),
        }
    gathered_reward = gather_tensor(samples["rewards"])
    gathered_video_reward = gather_tensor(samples["video_rewards"])
    if dist.get_rank()==0:
        print("gathered_reward", gathered_reward)
        print("gathered_video_reward", gathered_video_reward)
        with open('./reward.txt', 'a') as f:  # 'a'模式表示追加到文件末尾
            f.write(f"{gathered_reward.mean().item()}\n")
        with open('./video_reward.txt', 'a') as f:  # 'a'模式表示追加到文件末尾
            f.write(f"{gathered_video_reward.mean().item()}\n")

    train_timesteps_ablation = int(len(samples["timesteps"][0])*args.timestep_fraction_ablation)
    if args.first_ablation:
        samples["timesteps"] = samples["timesteps"][:, :train_timesteps_ablation]
        samples["latents"] = samples["latents"][:, :train_timesteps_ablation]
        samples["next_latents"] = samples["next_latents"][:, :train_timesteps_ablation]
        samples["log_probs"] = samples["log_probs"][:, :train_timesteps_ablation]
        samples["all_log_prob_first_frame"] = samples["all_log_prob_first_frame"][:, :train_timesteps_ablation]
    elif args.second_ablation:
        samples["timesteps"] = samples["timesteps"][:, -train_timesteps_ablation:]
        samples["latents"] = samples["latents"][:, -train_timesteps_ablation:]
        samples["next_latents"] = samples["next_latents"][:, -train_timesteps_ablation:]
        samples["log_probs"] = samples["log_probs"][:, -train_timesteps_ablation:]
        samples["all_log_prob_first_frame"] = samples["all_log_prob_first_frame"][:, -train_timesteps_ablation:]
    #计算advantage
    if args.use_group:
        n = len(samples["rewards"]) // (args.num_generations)
        advantages = torch.zeros_like(samples["rewards"])
        video_advantages = torch.zeros_like(samples["video_rewards"])

        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["video_rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            video_advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        
        samples["video_advantages"] = video_advantages
        
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        
        samples["advantages"] = advantages
    else:
        advantages = (samples["rewards"] - gathered_reward.mean())/(gathered_reward.std()+1e-8)
        samples["advantages"] = advantages

        video_advantages = (samples["video_rewards"] - gathered_video_reward.mean())/(gathered_video_reward.std()+1e-8)
        samples["video_advantages"] = video_advantages
    
    # shuffle samples along batch dimension
    #perm_batch = torch.randperm(batch_size).to(device) 
    #samples = {k: v[perm_batch] for k, v in samples.items()}

    # shuffle along time dimension independently for each sample
    perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0]))
            for _ in range(batch_size)
        ]
    ).to(device) 
    for key in ["timesteps", "latents", "next_latents", "log_probs", "all_log_prob_first_frame"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device) [:, None],
            perms,
        ]
    samples_batched = {
        k: v.unsqueeze(1)
        for k, v in samples.items()
    }
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]

    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
    if args.second_ablation:
        sigma_schedule = sigma_schedule[:-1][-train_timesteps_ablation-1:]
    for i,sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            clip_range = 1e-4
            adv_clip_max = 5.0
            #index = (sample["timesteps"][:,_] == t).nonzero().item()
            #step_out = self.run_step_ddpo(input, neg_input, pre_latents = pre_latents, yes = True, debug = False)
            new_log_probs, new_log_probs_first_frame = ddpo_one_step(
                args,
                sample["latents"][:,_],
                sample["next_latents"][:,_],
                sample["encoder_hidden_states"],
                sample["encoder_attention_mask"],
                sample["empty_cond_hidden_states"],
                sample["empty_cond_attention_mask"],
                transformer,
                sample["timesteps"][:,_],
                perms[i][_],
                sigma_schedule,
            )

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            video_advantages = torch.clamp(
                sample["video_advantages"],
                -adv_clip_max,
                adv_clip_max,
            )
            
            if args.discount_factor < 1:
                advantages_discount = args.discount_factor ** (args.sampling_steps - 1 - perms[i][_])
                advantages = advantages_discount * advantages

            ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])
            if args.use_first_frame_as_ratio:
                ratio_first_frame = torch.exp(new_log_probs_first_frame - sample["all_log_prob_first_frame"][:,_])
            else:
                ratio_first_frame = ratio
            unclipped_loss_first_frame = -advantages * ratio_first_frame
            clipped_loss_first_frame = -advantages * torch.clamp(
                ratio_first_frame,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            
            loss_first_frame = torch.mean(torch.maximum(unclipped_loss_first_frame, clipped_loss_first_frame))
            
            unclipped_loss = -video_advantages * ratio
            clipped_loss = -video_advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            loss_video = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
            
            
            grpo_ratio = torch.exp(sample["log_probs"][:,_] - new_log_probs)
            grpo_loss = torch.mean(grpo_ratio - torch.log(grpo_ratio) - 1)
            loss = (args.video_beta*loss_video + args.first_frame_beta*loss_first_frame + args.grpo_beta * grpo_loss)/(train_timesteps*args.gradient_accumulation_steps)
            #print(ratio)
            #if dist.get_rank()==0:
            #    breakpoint()
            #dist.barrier()
            #assert sample["latents"][:,_] == all_latents[i, perms[i][_]].unsqueeze(0)
            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
            if (_+1)%train_timesteps==0 and (i+1)%args.gradient_accumulation_steps==0:
                grad_norm = transformer.clip_grad_norm_(max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        if dist.get_rank()%8==0:
            print("first frame loss/video loss", loss_first_frame.item(), loss_video.item())
            print("reward/video reward", reward[i].item(), all_video_rewards[i].item())
            print("first frame ratio/video ratio", ratio_first_frame, ratio)
            print("advantage/video advantage", advantages.item(), video_advantages.item())
            print("final loss", loss.item())
        dist.barrier()
    return total_loss, grad_norm.item()


def sync_reference_model_weights(args, transformer, reference_transformer):
    """
    同步 transformer 的权重到 reference_transformer。
    
    参数:
        transformer: 主模型
        reference_transformer: 参考模型
    """
    # 提取 transformer 的权重
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(transformer, StateDictType.FULL_STATE_DICT, cfg):
        state = transformer.state_dict()
        #transformer_state_dict = fsdp_state_dict(transformer)

    reference_model = load_transformer(
        args.model_type, 
        args.dit_model_name_or_path, 
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )

    #for key in state:
    #    state[key] = state[key].to(reference_model.state_dict()[key].dtype)
    #    state[key] = state[key].to(reference_model.state_dict()[key].device)
    
    for key,value in state.items():
        if key in reference_model.state_dict():
            try:
                reference_model.state_dict()[key].copy_(value)
            except Exception as e:
                print(f"Error copying key {key}: {e}")
                if dist.get_rank() == 0:
                    breakpoint()
                dist.barrier()
    
    #reference_model.load_state_dict(state)

    # 2. 准备 FSDP 参数
    #    如果你已有类似 get_dit_fsdp_kwargs 的函数，可以也使用它或稍作修改：
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        reference_model,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    
    # 3. 使用 FSDP 包装参考模型
    reference_model = FSDP(reference_model, **fsdp_kwargs)
    return reference_model

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    if args.use_visionreward:
        MODEL_PATH = "./VisionReward-Video"
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        TORCH_TYPE = torch.bfloat16 

        processor = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            # padding_side="left"
        )

        reward_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True
        ).eval().to(DEVICE)


    if args.use_altclip:
        reward_model = AltClip(device=device, weight_path = args.weight_path, bf16=True).to(device)
        processor = None
        
    preprocess_val = None
    if args.use_hpsv2:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from typing import Union
        import huggingface_hub
        from hpsv2.utils import root_path, hps_version_map
        def initialize_model():
            model_dict = {}
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                'ViT-H-14',
                './hps_ckpt/open_clip_pytorch_model.bin',
                precision='amp',
                device=device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            model_dict['model'] = model
            model_dict['preprocess_val'] = preprocess_val
            return model_dict
        model_dict = initialize_model()
        model = model_dict['model']
        preprocess_val = model_dict['preprocess_val']
        #cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
        cp = "./hps_ckpt/HPS_v2.1_compressed.pt"

        checkpoint = torch.load(cp, map_location=f'cuda:{device}')
        model.load_state_dict(checkpoint['state_dict'])
        processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(device)
        reward_model.eval()
    inferencer = None
    if args.use_videoalign:
        from fastvideo.models.videoalign.inference import VideoVLMRewardInference
        load_from_pretrained = "./checkpoints"
        dtype = torch.bfloat16
        inferencer = VideoVLMRewardInference(load_from_pretrained, device=f'cuda:{device}', dtype=dtype)

    preprocess_dgn5b = None
    if args.use_clipscore:
        from open_clip import create_model_from_pretrained, get_tokenizer 
        from typing import Union
        import huggingface_hub
        model, preprocess_dgn5b = create_model_from_pretrained(
            model_name = 'ViT-H-14-378-quickgelu',
            pretrained = './DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin'
        )
        tokenizer = get_tokenizer('ViT-H-14')
        inferencer = model.to(device)
        inferencer.eval()

        
    #reward_model = 
    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    
    main_print(f"--> loading model from {args.model_type}")
    
    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )

    if args.use_lora:
        assert args.model_type != "hunyuan", "LoRA is only supported for huggingface model. Please use hunyuan_hf for lora finetuning"
        if args.model_type == "mochi":
            pipe = MochiPipeline
        elif args.model_type == "hunyuan_hf":
            pipe = HunyuanVideoPipeline
        transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)

    if args.resume_from_lora_checkpoint:
        lora_state_dict = pipe.lora_state_dict(
            args.resume_from_lora_checkpoint)
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(
            transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer,
                                                      transformer_state_dict,
                                                      adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys",
                                      None)
            if unexpected_keys:
                main_print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. ")

    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
    )

    if args.use_lora:
        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        transformer._no_split_modules = [
            no_split_module.__name__ for no_split_module in no_split_modules
        ]
        fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](transformer)

    transformer = FSDP(transformer, **fsdp_kwargs,)
    #reference_transformer = load_reference_model(args)
    main_print(f"--> model loaded")

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )

    # Set model as trainable.
    transformer.train()
    #reference_transformer.eval()
    #@reward_model.eval()

    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": False,
        "max_image_seq_len": 4096,
        "max_shift": 1.15,
        "num_train_timesteps": 1000,
        "shift": 7.0,
        "shift_terminal": None,
        "use_beta_sigmas": False,
        "use_dynamic_shifting": False,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False
    }
    noise_scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    if args.resume_from_lora_checkpoint:
        transformer, optimizer, init_steps = resume_lora_optimizer(
            transformer, args.resume_from_lora_checkpoint, optimizer
        )
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = (
        LengthGroupedSampler(
            args.train_batch_size,
            rank=rank,
            world_size=world_size,
            lengths=train_dataset.lengths,
            group_frame=args.group_frame,
            group_resolution=args.group_resolution,
        )
        if (args.group_frame or args.group_resolution)
        else DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader)
        / args.gradient_accumulation_steps
        * args.sp_size
        / args.train_sp_batch_size
    )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    vae, autocast_type, fps = load_vae(args.model_type, args.vae_model_path)
    #vae.enable_tiling()

    if rank <= 0:
        project = args.tracker_project_name or "fastvideo"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)
    empty_cond_hidden_states = torch.load(
        "./data/empty/prompt_embed/0.pt", map_location=torch.device(f'cuda:{device}'),weights_only=True
    )
    empty_cond_attention_mask = torch.load(
        "./data/empty/prompt_attention_mask/0.pt", map_location=torch.device(f'cuda:{device}'),weights_only=True
    )

    # todo future
    #for i in range(init_steps):
    #    next(loader)
    for epoch in range(1000):
        train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
        sampler = (
            LengthGroupedSampler(
                args.train_batch_size,
                rank=rank,
                world_size=world_size,
                lengths=train_dataset.lengths,
                group_frame=args.group_frame,
                group_resolution=args.group_resolution,
            )
            if (args.group_frame or args.group_resolution)
            else DistributedSampler(
                train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed+epoch
            )
        )
    
        train_dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            collate_fn=latent_collate_function,
            pin_memory=True,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            drop_last=True,
        )
        loader = sp_parallel_dataloader_wrapper(
            train_dataloader,
            device,
            args.train_batch_size,
            args.sp_size,
            args.train_sp_batch_size,
        )
        print("re-initialize dataloader!")

        
        for step in range(init_steps+1, args.max_train_steps+1):
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                if args.use_lora:
                    save_lora_checkpoint(transformer, optimizer, rank, args.output_dir,
                                         step, pipe, epoch)
                else:
                    save_checkpoint(transformer, rank, args.output_dir,
                                    step, epoch)

                dist.barrier()
            loss, grad_norm = train_one_step(
                args,
                device, 
                transformer,
                #reference_transformer,
                vae,
                reward_model,
                processor,
                inferencer,
                preprocess_dgn5b,
                empty_cond_hidden_states,
                empty_cond_attention_mask,
                args.model_type,
                optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                noise_random_generator,
                args.gradient_accumulation_steps,
                args.sp_size,
                args.precondition_outputs,
                args.max_grad_norm,
                args.weighting_scheme,
                args.logit_mean,
                args.logit_std,
                args.mode_scale,
                step,
                preprocess_val,
            )
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )
    
                
            if args.log_validation and step % args.validation_steps == 0:
                log_validation(args, transformer, device, torch.bfloat16, step)

    if args.use_lora:
        save_lora_checkpoint(transformer, optimizer, rank, args.output_dir,
                             args.max_train_steps, pipe)
    else:
        save_checkpoint(transformer, rank, args.output_dir,
                        args.max_train_steps)


    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="hunyuan_hf", help="The type of model to train."
    )
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t", type=int, default=28, help="Number of latent timesteps."
    )
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--uncond_prompt_dir", type=str)
    parser.add_argument(
        "--validation_sampling_steps",
        type=str,
        default="64",
        help="use ',' to split multi sampling steps",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=str,
        default="4.5",
        help="use ',' to split multi scale",
    )
    parser.add_argument("--validation_steps", type=int, default=50)
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=256, help="Alpha parameter for LoRA."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="LoRA rank parameter. "
    )
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"],
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,   
        help="Reward model path",
    )
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--sync_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,   
        help="eval steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,   
        help="fps of stored video",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--grpo_beta",
        type=float,
        default=0.05,   
        help="grpo beta",
    )
    parser.add_argument(
        "--video_beta",
        type=float,
        default=1.0,   
        help="video beta",
    )
    parser.add_argument(
        "--first_frame_beta",
        type=float,
        default=0.0,   
        help="video beta",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="loss coef",
    )
    parser.add_argument(
        "--gradient_accumulation_out_steps",
        type=int,
        default=2,   
        help="grad steps out",
    )
    parser.add_argument(
        "--use_all_qs",
        action="store_true",
        default=False,
        help="whether use 29 questions",
    )
    parser.add_argument(
        "--use_tuwen",
        action="store_true",
        default=False,  
        help="whether use tuwen questions",
    )
    parser.add_argument(
        "--use_aes",
        action="store_true",
        default=False,
        help="whether use aes questions",
    )
    parser.add_argument(
        "--use_aes_tuwen",
        action="store_true",
        default=False,
        help="whether use aes questions",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--use_first_frame_as_ratio",
        action="store_true",
        default=False,
        help="whether to use first frame as ratio",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_altclip",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--use_visionreward",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--use_hpsv2",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--use_videoalign",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--use_same_noise",
        action="store_true",
        default=False,
        help="whether to use same noise",
    )
    parser.add_argument(
        "--use_videoalign_mq",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--use_videoalign_vq",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--use_videoalign_vq_mq",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--use_videocuts",
        action="store_true",
        default=False,
        help="whether to use video cuts",
    )
    parser.add_argument(
        "--use_videoalign_overall",
        action="store_true",
        default=False,
        help="whether to use video cuts",
    )
    parser.add_argument(
        "--first_ablation",
        action="store_true",
        default=False,
        help="first_ablation",
    )
    parser.add_argument(
        "--second_ablation",
        action="store_true",
        default=False,
        help="second_ablation",
    )
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether to ignore last step",
    )
    parser.add_argument(
        "--discount_factor",
        type = float,
        default=1.0,
        help="dis factor",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep_fraction",
    )
    parser.add_argument(
        "--timestep_fraction_ablation",
        type = float,
        default=1.0,
        help="timestep_fraction_ablation",
    )
    parser.add_argument(
        "--cfg_infer",
        type = float,
        default=1.0,
        help="cfg",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="cfg",
    )
    parser.add_argument(
        "--use_clipscore",
        action="store_true",
        default=False,
        help="whether to use group",
    )





    args = parser.parse_args()
    main(args)
