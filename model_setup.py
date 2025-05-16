import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "HiDream-E1"))
from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
from peft import LoraConfig
from huggingface_hub import hf_hub_download
from diffusers import HiDreamImageTransformer2DModel
from safetensors.torch import load_file

def setup_models():
    """
    Set up and initialize all required models for HiDream-E1.
    Returns:
        tuple: (edit_pipe, gen_pipe, transformer, reload_keys)
    """
    # Set to True to enable instruction refinement and transformer model
    ENABLE_REFINE = True

    # Load models
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
    )

    # Configure transformer model if refinement is enabled
    transformer = None
    reload_keys = None
    if ENABLE_REFINE:
        transformer = HiDreamImageTransformer2DModel.from_pretrained("HiDream-ai/HiDream-I1-Full", subfolder="transformer")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["to_k", "to_q", "to_v", "to_out", "to_k_t", "to_q_t", "to_v_t", "to_out_t", "w1", "w2", "w3", "final_layer.linear"],
            init_lora_weights="gaussian",
        )
        transformer.add_adapter(lora_config)
        transformer.max_seq = 4608
        lora_ckpt_path = hf_hub_download(repo_id="HiDream-ai/HiDream-E1-Full", filename="HiDream-E1-Full.safetensors")
        lora_ckpt = load_file(lora_ckpt_path, device="cuda")
        src_state_dict = transformer.state_dict()
        reload_keys = [k for k in lora_ckpt if "lora" not in k]
        reload_keys = {
            "editing": {k: v for k, v in lora_ckpt.items() if k in reload_keys},
            "refine": {k: v for k, v in src_state_dict.items() if k in reload_keys},
        }
        info = transformer.load_state_dict(lora_ckpt, strict=False)
        assert len(info.unexpected_keys) == 0

    # Initialize editing pipeline
    if ENABLE_REFINE:
        edit_pipe = HiDreamImageEditingPipeline.from_pretrained(
            "HiDream-ai/HiDream-I1-Full",
            tokenizer_4=tokenizer_4,
            text_encoder_4=text_encoder_4,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
        )
    else:
        edit_pipe = HiDreamImageEditingPipeline.from_pretrained(
            "HiDream-ai/HiDream-E1-Full",
            tokenizer_4=tokenizer_4,
            text_encoder_4=text_encoder_4,
            torch_dtype=torch.bfloat16,
        )

    # Initialize generation pipeline
    gen_pipe = HiDreamImageEditingPipeline.from_pretrained(
        "HiDream-ai/HiDream-I1-Dev",
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
    )

    # Move pipelines to GPU
    edit_pipe = edit_pipe.to("cuda", torch.bfloat16)
    gen_pipe = gen_pipe.to("cuda", torch.bfloat16)
    
    return edit_pipe, gen_pipe, transformer, reload_keys 