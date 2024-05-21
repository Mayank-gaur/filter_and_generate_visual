import argparse
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import supervision as sv
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import os

def annotate_(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = _box_cxcywh_to_xyxy(boxes=boxes).numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame, xyxy

def _box_cxcywh_to_xyxy(boxes):
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (Tensor(N, 4)): boxes in (x1, y1, x2, y2) format.
    """
    # We need to change all 4 of them so some temporary variable is needed.
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    return boxes



def mask_bbox(mask, padding=[400, 50, 50, 50]):
    """
    args:
        mask -- np array (single channel)
    return:
        top, left, bottom, right -- int: coordinates of bbox
    """
    mask = mask/255.0
    mask = np.round(mask)
    coords = np.argwhere(mask == 1)
    top = int(coords[np.argmin(coords[:, 0]), 0]) - padding[0]
    top = max(0, top)
    bottom = coords[np.argmax(coords[:, 0]), 0] + padding[1]
    bottom = min(bottom, mask.shape[0])
    # adding padding across width only
    left = coords[np.argmin(coords[:, 1]), 1] - padding[2]
    left = max(0, left)
    right = coords[np.argmax(coords[:, 1]), 1] + padding[3]
    right = min(right, mask.shape[1])
    return top, left, bottom, right


def detect_object(image_path, text_prompt, model):
    
    image_source, image = load_image(image_path)
    
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=.65,
        text_threshold=.6,
    )
    print(boxes)
    print(logits)
    print(phrases)
    annotated_frame, xyxy = annotate_(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # cv2.imwrite(op_path, annotated_frame)

    
    return xyxy, image_source
    # You can further process or save the annotated_frame as needed


def init_sam(SAM_ENCODER_VERSION, SAM_CHECKPOINT_PATH,DEVICE):
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def overlay_red_mask(image, mask):
    """
    Overlay a red mask on the input image where the mask is 1.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): Binary mask (0 or 1) indicating object presence.

    Returns:
        numpy.ndarray: Output image with red highlights where the mask is 1.
    """
    for m in mask:
        # Create a red mask (all zeros except for red channel)
        red_mask = np.zeros_like(image)
        red_mask[:, :, 0] = 255  # Set red channel to 255 (full red)
        # mask  = mask[0]
        # Apply the mask to the red channel
        output_image = image.copy()
        output_image[m == 1] = red_mask[m == 1]

        return output_image    


def get_prod_masks(image_path, text_prompt,dino_model, sam_predictor):
    # detect mask
    xyxy, img = detect_object(image_path, text_prompt, dino_model)

    # get mask
    mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    xyxy= xyxy)

    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GroundingDINO Inference")
    parser.add_argument("--image_path", type=str, default="/home/ubuntu/projects/python/mayank/photoshoot_visual/translate_object/imgs/multiple_bags.png",required=False)
    parser.add_argument("--text_prompt", type=str, default="bag", required=False)
    parser.add_argument("--op_path", type=str, default="/home/ubuntu/projects/python/mayank/photoshoot_visual/res/tmp.png", required=False)
    args = parser.parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "/home/ubuntu/projects/python/mayank/translate_object/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = "/home/ubuntu/projects/python/mayank/translate_object/GroundingDINO/weights/groundingdino_swint_ogc.pth"


    dino_model = load_model(model_path, weights_path)
        # init sam
    sam_predictor = init_sam("vit_h", "/home/ubuntu/projects/python/mayank/translate_object/models/sam_vit_h_4b8939.pth", DEVICE)
    masks = get_prod_masks(args.image_path, args.text_prompt,dino_model,sam_predictor)
    # get red mask
    # mask_img = overlay_red_mask(img, mask)
    # save mask image to output path
    # cv2.imwrite(args.op_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))