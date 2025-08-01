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

from typing import Any, Tuple, Optional, Union, List

import torch
import torch.distributed as dist
from torch import Tensor

from fastvideo.utils.parallel_states import nccl_info
from fastvideo.utils.logging_ import main_print

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


from transformers import CLIPTokenizer, T5Tokenizer, CLIPTextModel, T5EncoderModel


def broadcast(input_: torch.Tensor):
    src = nccl_info.group_id * nccl_info.sp_size
    dist.broadcast(input_, src=src, group=nccl_info.group)


def _all_to_all_4D(input: torch.tensor,
                   scatter_idx: int = 2,
                   gather_idx: int = 1,
                   group=None) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (input.reshape(bs, shard_seqlen, seq_world_size, shard_hc,
                                 hs).transpose(0, 2).contiguous())

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(
            bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (input.reshape(
            bs, seq_world_size, shard_seqlen, shard_hc,
            hs).transpose(0, 3).transpose(0, 1).contiguous().reshape(
                seq_world_size, shard_hc, shard_seqlen, bs, hs))

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(
            bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError(
            "scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _all_to_all_4D(input, scatter_idx, gather_idx, group=group)

    @staticmethod
    def backward(ctx: Any,
                 *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(ctx.group, *grad_output, ctx.gather_idx,
                                ctx.scatter_idx),
            None,
            None,
        )


def all_to_all_4D(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return SeqAllToAll4D.apply(nccl_info.group, input_, scatter_dim,
                               gather_dim)


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous()
        for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input_, ctx.world_size, process_group,
                             scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )
        
class _AllToAllVariable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor_list, process_group):
        ctx.process_group = process_group
        ctx.input_shapes = [t.shape for t in input_tensor_list]
        ctx.world_size = dist.get_world_size(process_group)

        output_tensor_list = [
            torch.empty_like(t) for t in input_tensor_list
        ]

        dist.all_to_all(output_tensor_list, input_tensor_list, group=process_group)

        return tuple(output_tensor_list)  # autograd expects tuple

    @staticmethod
    def backward(ctx, *grad_outputs):
        process_group = ctx.process_group
        world_size = ctx.world_size

        grad_input_list = [
            torch.empty(ctx.input_shapes[i], device=grad_outputs[0].device, dtype=grad_outputs[0].dtype)
            for i in range(world_size)
        ]

        dist.all_to_all(grad_input_list, list(grad_outputs), group=process_group)

        return grad_input_list, None  # second input (group) doesn't require grad



def all_to_all(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, nccl_info.group, scatter_dim, gather_dim)

def all_to_all_variable(
    input_tensor_list: List[torch.Tensor],
):
    """
    Performs an all-to-all operation on a list of input tensors.

    Args:
        input_tensor_list (List[torch.Tensor]): List of input tensors to be communicated.

    Returns:
        Tuple[torch.Tensor]: Tuple of output tensors after all-to-all operation.
    """
    return _AllToAllVariable.apply(input_tensor_list, nccl_info.group)


class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        world_size = nccl_info.sp_size
        group = nccl_info.group
        input_size = list(input_.size())

        ctx.input_size = input_size[dim]

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = nccl_info.sp_size
        rank = nccl_info.rank_within_group
        dim = ctx.dim
        input_size = ctx.input_size

        sizes = [input_size] * world_size

        grad_input_list = torch.split(grad_output, sizes, dim=dim)
        grad_input = grad_input_list[rank]

        return grad_input, None


def all_gather(input_: torch.Tensor, dim: int = 1):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim)


def prepare_sequence_parallel_data(
    encoder_hidden_states, pooled_prompt_embeds, text_ids, supervise_latents, control_latents, referenced_latents, caption
):
    if nccl_info.sp_size == 1:
        return (
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids, 
            supervise_latents,
            control_latents,
            referenced_latents,
            caption,
        )

    def prepare(
        encoder_hidden_states, pooled_prompt_embeds, text_ids, supervise_latents, control_latents, referenced_latents, caption
    ):
        #hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=0)
        encoder_hidden_states = all_to_all(
            encoder_hidden_states, scatter_dim=1, gather_dim=0
        )
        #attention_mask = all_to_all(attention_mask, scatter_dim=1, gather_dim=0)
        pooled_prompt_embeds = all_to_all(
            pooled_prompt_embeds, scatter_dim=1, gather_dim=0
        )
        # TODO: 我们必须确保每次每个process取出的supervice_latents, control_latents, referenced_latents形状是相同的，否则无法支持这种将batch堆叠到同一个tensor的处理方法
        supervise_latents = all_to_all_variable(supervise_latents.tensor_split(dist.get_world_size(nccl_info.group), dim=1))
        control_latents = all_to_all_variable(control_latents.tensor_split(dist.get_world_size(nccl_info.group), dim=1))
        referenced_latents = all_to_all_variable(referenced_latents.tensor_split(dist.get_world_size(nccl_info.group), dim=1))
        
        text_ids = all_to_all(text_ids, scatter_dim=1, gather_dim=0)
        return (
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids, 
            supervise_latents,
            control_latents,
            referenced_latents,
            caption,
        )

    sp_size = nccl_info.sp_size
    #frame = hidden_states.shape[2]
    #assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    (
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids, 
        caption,
    ) = prepare(
        #hidden_states,
        encoder_hidden_states.repeat(1, sp_size, 1),
        pooled_prompt_embeds.repeat(1, sp_size, 1, 1),
        supervise_latents.repeat(1, sp_size, 1, 1),
        control_latents.repeat(1, sp_size, 1, 1),
        referenced_latents.repeat(1, sp_size, 1, 1),
        text_ids.repeat(1, sp_size),
        caption,
    )

    return encoder_hidden_states, pooled_prompt_embeds, supervise_latents, control_latents, referenced_latents, text_ids, caption

def handle_batch(batch: DataLoaderBatchDTO):
    prompts = batch.get_caption_list()
    latents = batch.latents
    control_tensor = batch.control_tensor
    referenced_tensor = batch.referenced_tensor
    return prompts, latents, control_tensor, referenced_tensor

def _get_t5_prompt_embeds(
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        t5_tokenizer: T5Tokenizer = None,
        t5: T5EncoderModel = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = t5_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = t5_tokenizer.batch_decode(untruncated_ids[:, t5_tokenizer.model_max_length - 1 : -1])
            main_print(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = t5(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = t5.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_clip_prompt_embeds
def _get_clip_prompt_embeds(
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_tokenizer: CLIPTokenizer = None,
        clip: CLIPTextModel = None,
    ):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = clip_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = clip_tokenizer.batch_decode(untruncated_ids[:, clip_tokenizer.model_max_length - 1 : -1])
            main_print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {77} tokens: {removed_text}"
            )
        prompt_embeds = clip(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=clip.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds
    


def sp_parallel_dataloader_wrapper(
    dataloader, device, train_batch_size, sp_size, train_sp_batch_size, clip_tokenizer, t5_tokenizer, clip, t5, vae
):
    while True:
        for data_item in dataloader:
            # encoder_hidden_states, pooled_prompt_embeds, text_ids, caption = data_item
            prompts, supervise_latents, control_tensor, referenced_tensor = handle_batch(data_item)
            #latents = latents.to(device)
            encoder_hidden_states = _get_t5_prompt_embeds(prompt=prompts, max_sequence_length=t5_tokenizer.model_max_length, device=device, dtype=t5.dtype, t5_tokenizer=t5_tokenizer, t5=t5)
            pooled_prompt_embeds = _get_clip_prompt_embeds(prompt=prompts, device=device, clip_tokenizer=clip_tokenizer, clip=clip)
            text_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=device, dtype=t5.dtype)
            control_latents = vae.encode(control_tensor.to(device)).latent_dist.mode()
            referenced_latents = vae.encode(referenced_tensor.to(device)).latent_dist.mode()
            #frame = latents.shape[2]
            frame = 19
            if frame == 1:
                yield encoder_hidden_states, pooled_prompt_embeds, text_ids, supervise_latents, control_latents, referenced_latents, caption
            else:
                encoder_hidden_states, pooled_prompt_embeds, text_ids, supervise_latents, control_latents, referenced_latents, caption = prepare_sequence_parallel_data(
                    encoder_hidden_states, pooled_prompt_embeds, text_ids, supervise_latents, control_latents, referenced_latents, caption
                )
                assert (
                    train_batch_size * sp_size >= train_sp_batch_size
                ), "train_batch_size * sp_size should be greater than train_sp_batch_size"
                for iter in range(train_batch_size * sp_size // train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size
                    encoder_hidden_states = encoder_hidden_states[st_idx:ed_idx]
                    pooled_prompt_embeds = pooled_prompt_embeds[st_idx:ed_idx]
                    text_ids = text_ids[st_idx:ed_idx]
                    supervise_latents = supervise_latents[st_idx:ed_idx]
                    control_latents = control_latents[st_idx:ed_idx]
                    referenced_latents = referenced_latents[st_idx:ed_idx]
                    yield (
                            encoder_hidden_states,
                            pooled_prompt_embeds,
                            text_ids, 
                            supervise_latents,
                            control_latents,
                            referenced_latents,
                            caption,
                    )

