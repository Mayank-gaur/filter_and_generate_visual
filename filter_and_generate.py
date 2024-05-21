import os
from groundingdino.util.inference import load_model
from filter import init_sam,get_prod_masks, mask_bbox
from archh import add_padding, change_bg, change_bg_ip, change_bg_outpaint
import torch
import numpy as np
import cv2
from PIL import Image
import random

def replace_background_by_blending_3channel(foreground_image, mask_image,color = (255, 192, 203)):
    # Open the foreground image and mask
    foreground = foreground_image
    mask = mask_image

    # Blend the foreground image onto the pink background based on the mask
    new_image_data = []
    for fg_pixel, mask_pixel in zip(foreground.getdata(), mask.getdata()):
        if mask_pixel> 0:  # Non-zero mask value (foreground)
            new_image_data.append(fg_pixel)
        else:  # Zero mask value (background)
            new_image_data.append(color)  # Pink color

    new_image = Image.new(mode="RGB", size=foreground.size)
    new_image.putdata(new_image_data)

    # Save the resulting image
    return new_image


class filter_and_generate_visual:
    def __init__(self, in_dict):
        # init sam and dino
        model_path = "/home/ubuntu/projects/python/mayank/translate_object/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        weights_path = "/home/ubuntu/projects/python/mayank/translate_object/GroundingDINO/weights/groundingdino_swint_ogc.pth"

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dino_model = load_model(model_path, weights_path)
        # init sam
        self.sam_predictor = init_sam("vit_h", "/home/ubuntu/projects/python/mayank/translate_object/models/sam_vit_h_4b8939.pth", DEVICE)

    def run(self, img_path, prompt_list,save_path):
        # define n to  denote count
        n = 0
        for prompt in prompt_list:
            print(prompt)
            try:
                # 1. get masks of prod mentioned in prompt
                masks = get_prod_masks(img_path, prompt, self.dino_model, self.sam_predictor)
                
                # 2.iterate through every prompt, get its variants, generate photostudio visual
                for i  in range(len(masks)):
                    # 2.1. get roi of prod and its mask
                    mask = masks[i]
                    # convert to float array
                    mask = mask.astype(float)
                    mask = np.where(mask == 1.0, 255.0, 0.0)
                    # get roi
                    t, l, b, r = mask_bbox(mask, padding=[5,5,5,5])
                    # crop img and mask acc to roi
                    img = cv2.imread(img_path)
                    img = img[t:b, l:r]
                    mask = mask[t:b, l:r]
                    #  alphablend
                    mask = cv2.merge((mask,mask,mask))
                    blend = np.where(mask == 255.0, img, 0)
                    blend= cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
                    inp_img = Image.fromarray(blend)
                    # add padding to the product
                    inp_img = add_padding(inp_img, 512)
                    # save padded ROI
                    filter_save_path = os.path.join(save_path, 'filter', prompt.strip())
                    os.makedirs(filter_save_path, exist_ok=True)
                    # y = cv2.imwrite(os.path.join(save_path, prompt, str(i) + '.png'), blend)
                    inp_img.save(os.path.join(filter_save_path, str(i) + '.png'))
                    # generate variants
                    command=f"python run_variants.py configs/instant-nerf-large.yaml {os.path.join(filter_save_path, str(i) + '.png')} --output_path {os.path.join(save_path, 'variants', prompt.strip(), str(i))}  --seed 1234 --view 4"
                    print(command)
                    os.system( command)  
                        # from the variants mesh, isolate diff prods and get their mask
                    # extract variants and their msks by running sam and dino. 
                    variants_strip_path = os.path.join(save_path, 'variants', prompt.strip(), str(i), 'instant-nerf-large', 'images', str(i)+'.png')
                    masks_variants = get_prod_masks(variants_strip_path,prompt,  self.dino_model, self.sam_predictor)
                    
                    for j in range(len(masks_variants)):
                        try:
                            # append count by 1
                            n += 1
                            # get roi of current variant
                            var_mask = masks_variants[j]
                            var_mask = var_mask.astype(float)
                            var_mask = np.where(var_mask == 1.0, 255.0, 0.0)
                            # get roi
                            t, l, b, r = mask_bbox(var_mask, padding=[5,5,5,5])
                            # crop img and var_mask acc to roi
                            img = cv2.imread(variants_strip_path)
                            img = img[t:b, l:r]
                            var_mask = var_mask[t:b, l:r]
                            #  alphablend
                            var_mask_tmp = cv2.merge((var_mask,var_mask,var_mask))
                            blend = np.where(var_mask_tmp == 255.0, img, 0)
                            blend = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
                            # define variant input image
                            var_inp_img = Image.fromarray(blend)
                            var_inp_img = add_padding(var_inp_img, 512)
                            var_inp_img = var_inp_img.resize((512, 512))
                            var_mask = Image.fromarray(var_mask)
                            # make res 512, suitable for generation by adding padding
                            var_mask = add_padding(var_mask, 512)
                            var_mask = var_mask.resize((512, 512))
                            # get bg mask
                            bg_mask = Image.eval(var_mask, lambda a: 255 - a)
                            # make bg of  var_inp_img grey, to generate photovisual shots
                            light_grey_color = (180, 180, 180)
                            random_bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                            var_inp_img = replace_background_by_blending_3channel(var_inp_img, var_mask,light_grey_color)
                            # Run inpainting to make image realistic
                            #  generate photostudio image (grey bg)
                            pipeline_res = change_bg_ip(var_inp_img, bg_mask, light_grey_color )
                            # save it
                            os.makedirs(os.path.join(save_path, 'res_final', prompt, str(i)), exist_ok=True)
                            pipeline_res.save(os.path.join(save_path, 'res_final', prompt, str(i),  str(j) + 'grey_bg.png'))
                            print()
                            #  generate photostudio image (random bg)
                            pipeline_res = change_bg_outpaint(var_inp_img, bg_mask, random_bg_color )
                            # save it
                            os.makedirs(os.path.join(save_path, 'res_final', prompt, str(i)), exist_ok=True)
                            pipeline_res.save(os.path.join(save_path, 'res_final', prompt, str(i),  str(j) + 'random_bg.png'))
                        except Exception as e:
                            print(e)
                            continue
 
            except Exception as e:
                print(e)
                continue
        if n == 0:
            print('reject image')
obj = filter_and_generate_visual({})
prods= ["Shoe",
"Sneaker"," Bottle"," Cup"," Sandal"," Perfume"," Toy"," Sunglasses"," Car","Bottle"," Chair"," Can"," Cap"," Hat",
"Couch"," Wristwatch"," Glass"," Bag"," Handbag"," Baggage"," Suitcase"," Headphones"," Jar"," Vase "
]
obj.run(r'/home/ubuntu/projects/python/mayank/photoshoot_visual/shoe.png',prods, r'/home/ubuntu/projects/python/mayank/photoshoot_visual/res/shoef')


