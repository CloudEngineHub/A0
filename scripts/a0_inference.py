import time
import yaml
import argparse
import os
import copy, json


import torch
import cv2
import numpy as np
from PIL import Image as PImage
from PIL import Image, ImageDraw

from scripts.afford_model import create_model


from scripts.utils import draw_text_on_image, draw_arrows_on_image_cv2

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", action="store", type=int, help="Random seed", default=None, required=False,)
    parser.add_argument("--chunk_size",action="store",type=int,help="Action chunk size",default=64,required=False,)
    parser.add_argument("--use_depth_image",action="store_true",help="Whether to use depth images",default=False,required=False,)

    parser.add_argument("--config_path",type=str,default="configs/base.yaml",help="Path to the config file",)
    parser.add_argument("--pretrained_model_name_or_path",type=str,required=True,default=None,help="Name or path to the pretrained model",)
    parser.add_argument('--pretrained_vision_encoder_name_or_path',type=str,required=False,
                        default="google/siglip-so400m-patch14-384",help='Name or path to the pretrained vision encoder')
    parser.add_argument('--pretrained_text_encoder_name_or_path',type=str,required=False,
                        default="Qwen/Qwen2.5-7B",help='Name or path to the pretrained text encoder. choice=[Qwen/Qwen2.5-7B,google/t5-v1_1-xxl]')
    parser.add_argument('--text_encoder',type=str,required=False,
                        default="Qwen2.5-7B",help='Name or path to the text encoder. choice=[Qwen2.5-7B,t5-v1_1-xxl]')
    
    parser.add_argument("--instruction", type=str, required=True, help="Instruction for the robotic manipulation task")

    parser.add_argument('--lang_embeddings_path', type=str, required=False, default=None,
                        help='Path to the pre-encoded language instruction embeddings')
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image")

    args = parser.parse_args()
    return args


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        # result.paste(pil_img, (0, (width - height) // 2))
        result.paste(pil_img, (0, 0))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        # result.paste(pil_img, ((height - width) // 2, 0))
        result.paste(pil_img, (0, 0))
        return result


# Initialize the model
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config

    model = create_model(
        args=args.config,
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_text_encoder_name_or_path=args.pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=args.pretrained_vision_encoder_name_or_path,
        text_encoder=args.text_encoder,
    )

    return model


# RDT inference
def inference_fn(args):
    instruction = args.instruction # instruction
    image_path = args.image_path  
    assert os.path.exists(image_path), image_path

    # Visual image saving path generation
    dir_name, file_name = os.path.split(image_path)
    file_base, file_ext = os.path.splitext(file_name)
    # Generate a new file name
    new_file_name = f"{file_base}_{instruction.replace(' ','_')}{file_ext}"
    # Combine new paths
    new_image_path = os.path.join(dir_name, new_file_name)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # model inference
    # Load rdt model
    policy = make_policy(args)
    if args.lang_embeddings_path is not None:
        lang_embeddings = torch.load(args.lang_embeddings_path)
    else:
        lang_embeddings = policy.encode_instruction(instruction)

    time1 = time.time()
    # fetch images in sequence [front, right, left]
    image_arrs = [
        None,
        # None,
        image,
        # None,
    ]

    images2 = [PImage.fromarray(arr) if arr is not None else None for arr in image_arrs]


    # actions shaped as [1, 64, 14] in format [left, right]
    actions = policy.step(
        images=images2,
        text_embeds=lang_embeddings,
    )

    print(f"Model inference time: {time.time() - time1} s")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    normalized_points = actions
    # normalized_points = normalized_points.view(-1, 2)
    points = copy.deepcopy(actions)
    # points = points.to(torch.float64)
    print("normalized four waypoints", points)
    points = points.view(-1, 2)
    h, w, _ = image.shape
 
    points[:, 0] *= w
    points[:, 1] *= h

    print("points", points.shape, points)
   
    m = points.to(dtype=torch.float32).cpu().numpy().astype(int)
    with open(new_image_path.replace(".png",".json"), 'w') as f:
        json.dump(m.tolist(), f) 
    
    points = points.to(torch.int)

    image = draw_text_on_image(image, instruction)
    draw_arrows_on_image_cv2(image, points, save_path=new_image_path)
    print(f"Result image saved to {new_image_path}")
    return normalized_points, points


args = get_arguments()


inference_fn(args)
