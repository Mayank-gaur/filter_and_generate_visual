# filter_and_generate_visual
This project offers an exciting and dynamic image processing pipeline, leveraging the cutting-edge capabilities of models like Stable Diffusion and ControlNet. By seamlessly integrating a range of versatile utility functions, this pipeline provides a delightful and comprehensive solution for several tasks. It allows you to filter objects based on text prompts, generate creative variants of those objects, and ultimately produce stunning photoshoot visuals.

## Installation
1. create environment using the yml file
  conda env create -f environment.yml
2. clone the repo
  git clone https://github.com/Mayank-gaur/filter_and_generate_visual.git
3. Setup DINO: Follow the steps here to install DINO: 
  https://github.com/IDEA-Research/DINO#installation
4. Setup SAM: Follow the steps here:
  https://github.com/facebookresearch/segment-anything#installation
5. In the projects folder, setup InstantMesh in the same newly created env.
    https://github.com/TencentARC/InstantMesh?tab=readme-ov-file#%EF%B8%8F-dependencies-and-installation
6. move contents of above to the projects folder. From the projects folder, run:
  mv InstantMesh/*  .
7. Thats it. Setup complete! Check the usage section here to run the project pipeline now.

## Core Functions
### add_padding
Adds padding to an image to ensure it reaches a specified resolution, centering the image and adding an even border on all sides.

### make_inpaint_condition
Prepares a masked image for inpainting by setting masked pixels to a specific value and converting the image to a tensor format suitable for model processing.

### blur_image
Applies a Gaussian blur to an image, handling both single-channel (grayscale) and multi-channel (RGB) images, ensuring the output remains true to the input's color mode.

### change_bg
Changes the background of an image using a mask and Stable Diffusion inpainting with ControlNet, generating a new image with specified background characteristics.

### change_bg_ip
Transforms an image's background using Stable Diffusion XL and inpainting. It utilizes an IP adapter for style guidance, blending the new background seamlessly with the original image.

### change_bg_outpaint
Expands the background of an image using Stable Diffusion XL with a control net, generating a new image with a plain background and blending it with the original.

### overlay_red_mask
Overlays a red mask on the input image where the mask indicates the presence of an object.

### get_prod_masks
Combines object detection and segmentation to produce masks for objects based on a text prompt using the GroundingDINO and SAM models.

### Main Class: filter_and_generate_visual
This class ties all core functions together, managing the entire image processing workflow. It initializes the necessary models and processes input images based on specified prompts to generate stunning visual outputs.

### Method: run
The run method orchestrates the processing of an input image based on a list of prompts. It performs the following steps:

Get Masks: Retrieves masks for products mentioned in the prompts.
Generate Variants: Iterates through the masks to create variants of the products.
Crop and Blend: Crops the image and blends it with the mask to isolate the product.
Add Padding: Ensures consistent resolution by adding padding to cropped images.
Save Variants: Stores the generated variants and their masks.InstantMesh is leveraged for this task.
Generate Visuals: Creates photoshoot visuals with different background colors. controlnet_inpaint_pipeline with ip adapter
and controlnet_inpaint_pipeline without ip adapter, both results are saved.
Run Inpainting: Applies inpainting techniques to enhance realism.
Save Results: Saves the final processed images, ready for use.

## Improvements
1. Modularize run function
2. Adjust IP adapter weightage  and prompts weihtage(guidance_scale) to generate more shadows
3. Support for passing style image option to user.


## Results
Input image
![Designer](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/ece94805-3be2-4b47-aa1d-bbcd96367241)

Output of few shoe, perfume and one of the sunglasses in the image is mentioned below

controlnet_inpaint_pipeline with ip adapter

![2grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/d4455615-65dc-4c56-b7ec-a5515c45a628)
![1grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/25b46efe-668f-41a6-aaa0-f30248f20798)
![0grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/ee294bec-9fcf-4b12-a594-b25dee03890d)
![5grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/c6af08ed-9073-4594-a044-3de38837a04f)

![4grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/9dfe2968-c9fe-431f-b678-c11e3d0cf37e)
![3grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/4278cc12-d141-4a34-ac18-f1ade027148c)
![2grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/3ee8a19c-1a79-44aa-8d28-ac82e6ac2d49)
![1grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/cc0a3159-092b-4c95-bd85-b9cf49e03ca2)
![0grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/289d74a1-9cf7-494d-aa4e-919b22984d42)
![5grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/d0044200-18b5-4d9e-a9df-ba920dc881ce)

![4grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/3b82fc55-0da5-4054-b2e9-8fd989db4e00)
![3grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/50c14a65-351a-4ac0-88d5-c73def431cce)
![2grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/60310999-f971-46d4-b56c-4217455bbea1)
![1grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/35e96fbd-40ad-4b79-bacc-701e38f7f258)
![0grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/de72091f-7468-4fbe-8fe5-a7c838b79065)
![5grey_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/712f4167-6bd4-4c37-8ce0-5f8e16e078a0)


controlnet_inpaint_pipeline without ip adapter
![0random_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/f5d36f2c-7590-424c-a3f6-0d7e8a5d566b)
![5random_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/42f0b3f3-8c00-4979-b499-cea20e89c3b7)
![3random_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/a470ba8e-1f6b-4a26-96cf-1eb7e2c3f3d9)
![1random_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/1fb6537f-afa5-4c5d-80b5-31cef7d0dcd2)

![2random_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/18e468cf-2c9e-455c-9ff8-e9afb68846d5)
![1random_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/7d446079-d65d-4fdc-b4cd-1c4f36c33637)
![0random_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/8825a6dd-16dc-40c4-871d-3ffa4a8e6d6f)
![5random_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/bf113801-71ca-4dfc-8874-dba2c124c554)
![4random_bg](https://github.com/Mayank-gaur/filter_and_generate_visual/assets/56195849/251bd526-6906-4f24-a4b2-2aa2beb397c9)


### Usage Example
Using this pipeline is straightforward. Simply create an instance of the filter_and_generate_visual class, call the run method with the necessary parameters, and watch the magic happen! Ensure the input image path, list of prompts, and save path are correctly specified for optimal results.

### Conclusion
This project is a celebration of creativity and technological advancement, offering a powerful and flexible image processing pipeline. With the ability to filter objects, generate variants, and create beautiful photoshoot visuals.
