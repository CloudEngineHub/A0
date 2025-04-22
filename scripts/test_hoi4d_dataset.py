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

from scripts.afford_model import create_model
from data.waypoint_hoi4d_dataset import HOI4DWaypointDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
def inference_fn(args,split='test'):

    output_dir = args.image_save_path
    if output_dir:
        output_subdir = os.path.join(   args.image_save_path,'hoi4d')
    
    # Load rdt model
    policy = make_policy(args)

    dataset = HOI4DWaypointDataset(split=split,first_frame=args.first_frame)
    num = len(dataset)
    idxx = 0

    mse_all = []
    mae_all = []
    mae_first_all = []
    mse_cls_all = {}
    mae_cls_all = {}
    for idx in tqdm(range(idxx,idxx+num,1)): 
        sample = dataset[idx]

        meta = sample['meta']

        images = sample['cam_high']

        image_dir = meta['image_dir']
        text = sample['meta']['instruction']
        objectt = sample['meta']['label_name']


        # states = sample['state']
        action_gt = sample['actions']
        # points_gt = action_gt[0:4,103:105]
        points_gt = action_gt[:,:]
        lang_embeddings = policy.encode_instruction(text)

        
        time1 = time.time()     

        # fetch images in sequence [front, right, left]
        image_arrs = [
            images[0,:,:,:],
            # depths[0,:,:,:],
            # None,
            
            images[1,:,:,:],
            # depths[1,:,:,:],
            # None,
        ]
        
        images2 = [PImage.fromarray(arr) if arr is not None else None
                    for arr in image_arrs]
        
        # actions shaped as [1, 64, 14] in format [left, right]
        actions = policy.step(
            images=images2,
            text_embeds=lang_embeddings,
            poa=args.poa
        )
        # print(f"inference_actions: {actions.squeeze()}")
        
        # print(f"Model inference time: {time.time() - time1} s")
        
        # image = np.array(image)
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

        if objectt not in mse_cls_all:
            mse_cls_all[objectt] = []
        if objectt not in mae_cls_all:
            mae_cls_all[objectt] = []
        mse_cls_all[objectt].append(mse_loss.cpu().numpy())
        mae_cls_all[objectt].append(mae_loss.cpu().numpy())

        if output_dir:
            points_gt = points_gt.to(torch.int)
            points = points.to(torch.int)

            draw_arrows_on_image(image,points_gt)
            image = draw_text_on_image(image,text)
            s_path = os.path.join(output_dir,output_subdir,split,f"{split}_id_{idx}_{image_dir.replace('/','_')}.jpg")
            draw_arrows_on_image_cv2(image, points, save_path=s_path)

    
    a = np.mean(mse_all)
    b = np.mean(mae_all)
    c = np.mean(mae_first_all)
    print('------------------------------')
    print('mae',b)
    print('mae first point',c)
    print('mse',a)
    
    print('------------------------------')
    mean_values = {key: sum(values) / len(values) if values else 0 for key, values in mae_cls_all.items()}

    print("Per-class MAE:")
    for cls, mean in mean_values.items():
        print(f"{cls}: {mean:.4f}")

    
args = get_arguments()
if args.seed is not None:
    set_seed(args.seed)

inference_fn(args)
