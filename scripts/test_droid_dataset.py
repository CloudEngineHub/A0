import time
import yaml
import argparse
import os

import torch

import cv2
import numpy as np
from PIL import Image as PImage
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.afford_model import create_model
from data.waypoint_droid_cotracker_dataset import DroidWaypointDatasetCotracker
from data.waypoint_droid_dataset import DroidWaypointDataset


from scripts.utils import draw_arrows_on_image, draw_text_on_image, draw_arrows_on_image_cv2

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action='store', type=int, 
                        help='Random seed', default=None, required=False)

    parser.add_argument('--use_depth_image', action='store_true', 
                        help='Whether to use depth images',
                        default=False, required=False)
    
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')

    parser.add_argument('--pretrained_vision_encoder_name_or_path',type=str,required=False,
                        default="google/siglip-so400m-patch14-384",help='Name or path to the pretrained vision encoder')
    parser.add_argument('--pretrained_text_encoder_name_or_path',type=str,required=False,
                        default="Qwen/Qwen2.5-7B",help='Name or path to the pretrained text encoder. choice=[Qwen/Qwen2.5-7B,google/t5-v1_1-xxl]')
    parser.add_argument('--text_encoder',type=str,required=False,
                        default="Qwen2.5-7B",help='Name or path to the text encoder. choice=[Qwen2.5-7B,t5-v1_1-xxl]')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=False,
                        default=None, help='Name or path to the pretrained model')
    
    # parser.add_argument('--lang_embeddings_path', type=str, required=True, 
    #                     help='Path to the pre-encoded language instruction embeddings')
    parser.add_argument('--dataset',type=int,default='1',help='assign the tow different datases of droid.')
    parser.add_argument('--poa',action='store_true',help='use position offset attention of embeddings of the image encoder')
    parser.add_argument('--first_frame',action='store_true',help='use the first frame rather previous frame of the dataset for image encoder')
    parser.add_argument('--image_save_path', type=str, default=None,help='Path to save the images')

    # set default to False of first_frame
    
    args = parser.parse_args()
    return args


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
def inference_fn(args,dataset,split = 'test'):

    output_dir = args.image_save_path
    if output_dir:
        output_subdir = os.path.join(  output_dir,'droid')
    if args.dataset ==2:
        if output_dir:
            output_subdir = os.path.join(  output_dir,'droid_new')

    num = len(dataset)
    idxx = 0

    mse_all = []
    mae_all = []
    mae_first_all = []
    # mse_cls_all = {}
    mae_cls_all = {}
    for idx in tqdm(range(idxx,idxx+num,1)): 
        sample = dataset[idx]

        images = sample['cam_high']
        # depths = sample['cam_right_wrist']
        text = sample['meta']['instruction']

        episode_id = sample['meta']['episode_id']
        sub_class = sample['meta']['sub_class']
        json_file_path = sample['meta']['json_file_path']
        start = sample['meta']['validRange']['start']
        # print('episode_id',episode_id,'sub class',sub_class)
        # print(json_file_path)

        # states = sample['state']
        action_gt = sample['actions']
        points_gt = action_gt[:,:]

        # print('instruction:',text)
        lang_embeddings = policy.encode_instruction(text)
        
        time1 = time.time()     

        # fetch images in sequence [front, right, left]
        image_arrs = [
            images[0,:,:,:],
            # None,
            # None,
            
            images[1,:,:,:],
            # None,
            # None,
        ]
        
        images2 = [PImage.fromarray(arr) if arr is not None else None
                    for arr in image_arrs]
        
        actions = policy.step(
            images=images2,
            text_embeds=lang_embeddings,
            poa=args.poa
        )
        # print(f"inference_actions: {actions.squeeze()}")
        

        # print(f"Model inference time: {time.time() - time1} s")
        

        image =images[1,:,:,:]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        points_gt = torch.tensor(points_gt)

        points = actions[0,:,:]

        h,w,_ = image.shape
        points_gt[:,0] *=  w
        points_gt[:,1] *=  h
        
        points[:,0] *= w
        points[:,1] *= h

        points_gt = points_gt.to(torch.float)
        points = points.to(torch.float)

        points =  points.view(-1, 2)
        points_gt =  points_gt.view(-1, 2)

        mse_loss = F.mse_loss(points.cuda(), points_gt.cuda())

        mae_loss = F.l1_loss(points.cuda(), points_gt.cuda())
        mae_first_loss = F.l1_loss(points[0,:].cuda(), points_gt[0,:].cuda())

        mse_all.append(mse_loss.cpu().numpy())
        mae_all.append(mae_loss.cpu().numpy())
        mae_first_all.append(mae_first_loss.cpu().numpy())
        # cls mae
        if sub_class not in mae_cls_all:
            mae_cls_all[sub_class] = []
        mae_cls_all[sub_class].append(mae_loss.cpu().numpy())

        if args.image_save_path:
            points_gt = points_gt.to(torch.int)
            points = points.to(torch.int)

            draw_arrows_on_image(image,points_gt)
            image = draw_text_on_image(image,text,color=(0,0,255),thickness=1,font_scale=0.4)
            s_path = os.path.join(output_dir,output_subdir,split,f"episode_{episode_id}_{start}.jpg")
            draw_arrows_on_image_cv2(image, points, save_path=s_path)
            # print(f"Image with arrows saved to {s_path}")

    
    a = np.mean(mse_all)
    b = np.mean(mae_all)
    c = np.mean(mae_first_all)
    print(dataset)
    print('------------------------------')
    print(f"mae: {b:.2f}")
    print(f"mae first point: {c:.2f}")
    print(f"mse: {a:.2f}")


args = get_arguments()
if args.seed is not None:
    set_seed(args.seed)

policy = make_policy(args)

dataset = DroidWaypointDatasetCotracker(split='test',first_frame=args.first_frame)
inference_fn(args,dataset)
dataset = DroidWaypointDataset(split='test',first_frame=args.first_frame)
args.dataset=2
inference_fn(args,dataset)
