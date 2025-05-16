import torch
from PIL import Image
from typing import Optional, Tuple, Union

def process_image_edit(
    pipe,
    image: Image.Image,
    prompt: str,
    reload_keys: Optional[dict] = None,
    guidance_scale: float = 5.0,
    image_guidance_scale: float = 4.0,
    num_inference_steps: int = 28,
    refine_strength: float = 0.3,
    seed: int = 3
) -> Image.Image:
    """
    Process an image editing request using the HiDream pipeline.
    
    Args:
        pipe: The HiDream pipeline
        image: Input image to edit
        prompt: Editing instruction
        reload_keys: Keys for model reloading
        guidance_scale: Text guidance scale
        image_guidance_scale: Image guidance scale
        num_inference_steps: Number of inference steps
        refine_strength: Strength of refinement (0-1)
        seed: Random seed for generation
        
    Returns:
        PIL.Image: Edited image
    """
    # Store original dimensions
    original_width, original_height = image.size
    
    # Resize to model's required size
    processed_image = image.resize((768, 768))
    
    # Generate edited image
    edited_image = pipe(
        prompt=prompt,
        negative_prompt="low resolution, blur",
        image=processed_image,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        refine_strength=refine_strength,
        reload_keys=reload_keys,
    ).images[0]
    
    # Resize back to original dimensions
    edited_image = edited_image.resize((original_width, original_height))
    
    return edited_image

def process_image_generation(
    pipe,
    prompt: str,
    size: Tuple[int, int] = (768, 768),
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 42
) -> Image.Image:
    """
    Generate a new image using the HiDream pipeline.
    
    Args:
        pipe: The HiDream pipeline
        prompt: Generation instruction
        size: Output image size (width, height)
        guidance_scale: Text guidance scale
        num_inference_steps: Number of inference steps
        seed: Random seed for generation
        
    Returns:
        PIL.Image: Generated image
    """
    # Generate image
    generated_image = pipe(
        prompt=prompt,
        negative_prompt="low resolution, blur, bad quality",
        height=size[1],
        width=size[0],
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    
    return generated_image

def is_image_generation_prompt(prompt: str) -> bool:
    """
    Determine if the prompt is asking for image generation rather than editing.
    
    Args:
        prompt: User's input prompt
        
    Returns:
        bool: True if the prompt is for image generation
    """
    generation_keywords = [
        "create", "generate", "make", "draw", "design", "new image",
        "create an image", "generate a picture", "make a photo"
    ]
    
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in generation_keywords) 