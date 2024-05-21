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
Save Variants: Stores the generated variants and their masks. A 2d to 3d nerf based model is leveraged for this task.
Generate Visuals: Creates photoshoot visuals with different background colors.
Run Inpainting: Applies inpainting techniques to enhance realism.
Save Results: Saves the final processed images, ready for use.
### Usage Example
Using this pipeline is straightforward. Simply create an instance of the filter_and_generate_visual class, call the run method with the necessary parameters, and watch the magic happen! Ensure the input image path, list of prompts, and save path are correctly specified for optimal results.

### Conclusion
This project is a celebration of creativity and technological advancement, offering a powerful and flexible image processing pipeline. With the ability to filter objects, generate variants, and create beautiful photoshoot visuals.
