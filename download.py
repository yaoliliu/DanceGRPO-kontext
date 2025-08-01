from huggingface_hub import snapshot_download

local_dir = "data/flux-kontext"  # 自定义目录
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-Kontext-dev",
    local_dir=local_dir,
    local_dir_use_symlinks=False  # 推荐设为 False，实际复制文件
)

# import torch
# from transformers import AutoTokenizer, AutoModel
# path = "OpenGVLab/InternVL3-8B"
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     use_flash_attn=True,
#     trust_remote_code=True).eval().cuda()
