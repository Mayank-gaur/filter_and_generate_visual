from diffusers import  ControlNetModel, StableDiffusionControlNetInpaintPipeline,StableDiffusionXLControlNetInpaintPipeline 
import torch
from diffusers.utils import load_image
from PIL import Image, ImageFilter
import os
from diffusers import (
    EulerAncestralDiscreteScheduler,
    )
from transformers import CLIPVisionModelWithProjection
import cv2
import numpy as  np
from filter import get_prod_masks



def add_padding(image, op_res):
    """
    Add padding to an image to reach a specified resolution.

    Parameters:
    - image (Image): The PIL Image object to be padded.
    - op_res (int): The desired resolution for both width and height after padding.
    """
     # Open the input image
    # image = Image.open(image_path)
    image = image
    # Get the width and height of the image
    w, h = image.size

    # Calculate padding along height and width
    padding_height = abs(op_res - h) // 2
    padding_width = abs(op_res - w) // 2

    # Create a new image with the desired dimensions
    new_width = w + 2 * padding_width
    new_height = h + 2 * padding_height
        # Check if the image is single-channel (grayscale)
    if image.mode == 'L':
        result = Image.new('L', (new_width, new_height), 0)  # Black background
    elif image.mode == 'F':
        result = Image.new('F', (new_width, new_height), 0)  # Black background
    else:
        result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))  # Black background
    # Paste the original image onto the new canvas
    result.paste(image, (padding_width, padding_height))

    # Save the padded image
    # result.save('padded_image.jpg')

    return result


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


# def generate_variations(img_path, product, save_path ):

    

def blur_image(input_image, blur_factor):
    """
    Applies a Gaussian blur to an image.

    Parameters:
    - input_image: The image to be blurred.
    - blur_factor: The intensity of the blur; must be a non-negative integer.

    The function handles images with or without an alpha channel and ensures
    the output is in the same color mode as the input, either 'L' or 'RGB'.

    Returns:
    - The blurred image.
    """
    # Ensure the blur factor is a non-negative integer
    blur_factor = max(int(blur_factor), 0)
    
    # Convert the image to the appropriate mode if necessary
    if input_image.mode not in ['L', 'RGB']:
        if 'A' in input_image.mode:
            # If the image has an alpha channel, separate it and merge after blurring
            alpha = input_image.split()[-1]
            input_image = input_image.convert('RGB')
            blurred_image = input_image.filter(ImageFilter.GaussianBlur(blur_factor))
            blurred_image.putalpha(alpha)
        else:
            # Convert the image to RGB before blurring
            input_image = input_image.convert('RGB')
            blurred_image = input_image.filter(ImageFilter.GaussianBlur(blur_factor))
    else:
        # Apply Gaussian Blur filter directly
        blurred_image = input_image.filter(ImageFilter.GaussianBlur(blur_factor))
        
    # Convert the blurred image back to single channel if it's not
    if blurred_image.mode != 'L':
        blurred_image = blurred_image.convert('L')
    return blurred_image


def change_bg ( inp_img, obj_mask, product, color):
    """
    Changes the background of an image using a mask and Stable Diffusion inpainting with controlnet.

    Parameters:
    - inp_img: The input image to modify.
    - obj_mask: The object mask defining areas to keep.
    - product: Unused parameter, can be removed or repurposed.
    - color: The color to avoid in negative prompts.

    This function initializes a Stable Diffusion pipeline with a specific seed and
    controlnet model, then generates a new image with the specified background prompts,
    avoiding certain features as defined in the negative prompt.

    Returns:
    - The image with the modified background.
    """

    bg_mask= obj_mask    
    generator = torch.Generator(device='cuda').manual_seed(0)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16")
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.enable_model_cpu_offload()
    # bg_mask = Image.fromarray(bg_mask)
    control_image = make_inpaint_condition(inp_img, bg_mask)
    gen_image = pipeline(prompt=f"pink wall, realistic texture, black shadows",
                    negative_prompt = 'any objects, any colors, any patterns, watermarks, text, big shadows, not {color},  badartist (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated ',
                    image=inp_img,
                    mask_image=bg_mask,
                    generator=generator,
                    strength = 0.7,
                    guidance_scale = 7.5,
                    num_inference_steps=80,
                    control_image=control_image).images[0]
    return gen_image


def change_bg_ip ( inp_img, obj_mask, color):
    """
    Changes an image's background using Stable Diffusion XL and inpainting.

    Parameters:
    - inp_img: Image to modify.
    - obj_mask: Mask for object boundaries.
    - color: Desired background color (RGB).

    Sets up a pipeline with control net and image encoder, uses an IP adapter for style guidance, and overlays the new background.

    Returns:
    - Modified image with new background.
    """
    bg_mask= obj_mask
    # define SDXL controlnet inpainting pipeline integrated with an ip adapter
    # 
    generator = torch.Generator(device='cuda').manual_seed(0)
    controlnet = ControlNetModel.from_pretrained("destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                torch_dtype=torch.float16,
                # cache_dir=CACHE_DIR
    ).to('cuda')
    pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", image_encoder=image_encoder
    )
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
    if color == (180, 180, 180):
        pipeline.set_ip_adapter_scale(0.4)
    else:
        pipeline.set_ip_adapter_scale(0.6)

    pipeline.to('cuda')
    # create style image for ip adapter.(a blank canvas with  color as input color ). generated image bg will be of similar color as style image
    bg_color = np.zeros((480, 640, 3), dtype=np.uint8)
    bg_color[:] = color  # RGB values for pink (BGR order)
    style_img = Image.fromarray(cv2.cvtColor(bg_color, cv2.COLOR_BGR2RGB))

    inp_img_blur = blur_image(inp_img, 1).convert('RGB')    # control_image.resize_(1, 3, 768, 768)
    # run generation
    gen_image = pipeline(prompt=f"highly textured wall, realistic texture of wall,(significant black shadows),",
                negative_prompt = 'any objects, any colors, any patterns, watermarks, text, big shadows,  badartist (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated ',
                    
                image=inp_img,
                ip_adapter_image=style_img,
                mask_image=bg_mask,
                generator=generator,
                strength = 0.85,
                num_inference_steps=80,
                control_image=inp_img_blur).images[0]
    # overllay generated image with orig image
    gen_image = pipeline.image_processor.apply_overlay(bg_mask, inp_img, gen_image)                             
    return gen_image



def change_bg_outpaint ( inp_img, obj_mask, color):
    """
    Outpaints the background of an image using Stable Diffusion XL with a control net.

    Parameters:
    - inp_img: The input image to be processed.
    - obj_mask: The mask defining the object's boundaries.
    - color: The target background color as an RGB tuple.

    The function sets up a Stable Diffusion XL pipeline with a control net and image encoder,
    then generates a new image with a plain background, avoiding certain features as defined
    in the negative prompt. It applies an overlay to blend the generated background with the
    original image.

    Returns:
    - The image with the outpainted background.
    """
    bg_mask= obj_mask
    # generate seed
    generator = torch.Generator(device='cuda')
    # define SDXL controlnet inpainting pipeline
    controlnet = ControlNetModel.from_pretrained(
        "alimama-creative/EcomXL_controlnet_inpaint",
        use_safetensors=True,
    )    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                torch_dtype=torch.float16,
                # cache_dir=CACHE_DIR
    ).to('cuda')
    pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", image_encoder=image_encoder
    )
    pipeline.to(device="cuda", dtype=torch.float16)
    # make condition image for controlnet
    control_image = make_inpaint_condition(inp_img, bg_mask)
    # run generation
    gen_image = pipeline(
        prompt=f"plain wall , shallow shadows of product, uniformly plain",
        negative_prompt = 'texture, stripped, multiple colors',
                
        image=inp_img,
        # ip_adapter_image=style_img,
        mask_image=bg_mask,
        generator=generator,
        strength = 0.9,
        # guidance_scale = 10,
        num_inference_steps=80,
        control_image=control_image).images[0]
    # overllay generated image with orig image
    gen_image = pipeline.image_processor.apply_overlay(bg_mask, inp_img, gen_image)                 
    return gen_image





# generate_img(r'/home/ubuntu/projects/python/mayank/photoshoot_visual/res/ Bottle /1.png', 'bottle', r'/home/ubuntu/projects/python/mayank/photoshoot_visual/res_arch/Bottle')
def main_change_bg():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "/home/ubuntu/projects/python/mayank/translate_object/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = "/home/ubuntu/projects/python/mayank/translate_object/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    import groundingdino
    dino_model = groundingdino.util.inference.load_model(model_path, weights_path)
        # init sam
    from filter import init_sam

    sam_predictor = init_sam("vit_h", "/home/ubuntu/projects/python/mayank/translate_object/models/sam_vit_h_4b8939.pth", DEVICE)
    change_bg(r'/home/ubuntu/projects/python/mayank/photoshoot_visual/res/ Sunglasses/0.png', 
            'bottle',
            dino_model,
            sam_predictor,
            'white',
            r'/home/ubuntu/projects/python/mayank/photoshoot_visual/res_archh/sunglasses')

# main_change_bg()