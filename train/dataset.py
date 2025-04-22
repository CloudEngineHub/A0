import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import transformers

from data.filelock import FileLock
from data.waypoint_hoi4d_frame_dataset import HOI4DWaypointFrameDataset

from data.waypoint_droid_cotracker_dataset import DroidWaypointDatasetCotracker
from data.waypoint_droid_dataset import DroidWaypointDataset
from data.waypoint_maniskill_dataset import ManiskillWaypointDataset
from data.waypoint_hoi4d_dataset import HOI4DWaypointDataset


from train.image_corrupt import image_corrupt


def get_clean_item(chunk_dir):
    """
    Get indexes of clean items in a chunk.
    """
    dirty_bit = read_dirty_bit(chunk_dir)
    return np.where(1 - dirty_bit)[0].tolist()


def save_dirty_bit(chunk_dir, dirty_bit):
    """
    Save the dirty bit to the chunk directory.
    """
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_write_lock()
            with open(file_path, 'wb') as file:
                file.write(dirty_bit.tobytes())
            lock.release_lock()
            return
        except KeyboardInterrupt:
            lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            lock.release_lock()
            continue
    raise RuntimeError("Failed to save dirty bit.")


def read_dirty_bit(chunk_dir):
    """
    Read the dirty bit from the chunk directory.
    """
    # If error occurs, retry
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_read_lock()
            with open(file_path, 'rb') as file:
                dirty_bit = np.frombuffer(file.read(), dtype=np.uint8).copy()
            lock.release_lock()
            assert len(dirty_bit) > 0
            return dirty_bit
        except KeyboardInterrupt:
            lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            lock.release_lock()
            continue
    raise RuntimeError("Failed to read dirty bit.")


def flip_image_with_waypoints(image_list, waypoints):
    """
    Perform horizontal and vertical flip augmentations on a list of input OpenCV images
    and update the normalized waypoints accordingly.
    
    Args:
        image_list (list of numpy.ndarray): A list of input images in OpenCV format (numpy arrays).
        waypoints (numpy.ndarray): Waypoints with shape (N, 2), where N is the number of points.
                                   Values are normalized to (x, y) in [0, 1].
        
    Returns:
        images_augment (list of numpy.ndarray): The augmented images.
        updated_waypoints (numpy.ndarray): The updated waypoints.
    """
    assert isinstance(image_list[0][0], np.ndarray), "The input images must be in numpy.ndarray format."
    assert waypoints.shape[1] == 2, "The waypoints must have a shape of (N, 2)."
    
    images_augment = []

    # Randomly choose a flip type
    flip_type = random.choice(["horizontal", "vertical", "both", "none", "none"])

    if flip_type == "horizontal":
        for image,flag in image_list:
            # Perform horizontal flip
            flipped_image = cv2.flip(image, 1)
            images_augment.append((flipped_image,flag))
        # Update x-coordinates of waypoints
        waypoints[:, 0] = 1 - waypoints[:, 0]
    
    elif flip_type == "vertical":
        for image,flag in image_list:
            # Perform vertical flip
            flipped_image = cv2.flip(image, 0)
            images_augment.append((flipped_image,flag))
        # Update y-coordinates of waypoints
        waypoints[:, 1] = 1 - waypoints[:, 1]
    
    elif flip_type == "both":
        for image,flag in image_list:
            # Perform both horizontal and vertical flip
            flipped_image = cv2.flip(image, -1)
            images_augment.append((flipped_image,flag))
        # Update both x and y coordinates of waypoints
        waypoints[:, 0] = 1 - waypoints[:, 0]
        waypoints[:, 1] = 1 - waypoints[:, 1]
    
    else:
        # No flip applied, return the original images
        images_augment = image_list.copy()

    return images_augment, waypoints


def random_crop_images_with_waypoints(image_list, waypoints, crop_ratio=0.9):
    """
    Perform random cropping on a list of input images and update the normalized waypoints.
    
    Args:
        image_list (list of numpy.ndarray): A list of input images in OpenCV format (numpy arrays).
        waypoints (numpy.ndarray): Waypoints with shape (N, 2), where N is the number of points.
                                   Values are normalized to (x, y) in [0, 1].
        crop_size (tuple): The size of the crop (crop_height, crop_width).
        
    Returns:
        cropped_images (list of numpy.ndarray): The cropped images.
        updated_waypoints (numpy.ndarray): The updated waypoints.
    """

    h, w, _ = image_list[0][0].shape  # Original image height, width
    crop_h, crop_w = int(h * crop_ratio),int(w * crop_ratio)  # Crop height and width

    # Ensure the crop size is valid
    if crop_h > h or crop_w > w:
        raise ValueError("Crop size must be smaller than the image dimensions.")

    # Randomly choose a center point for cropping
    center_x = random.randint(crop_w // 2, w - crop_w // 2)
    center_y = random.randint(crop_h // 2, h - crop_h // 2)

    # Compute cropping box
    left = center_x - crop_w // 2
    top = center_y - crop_h // 2
    right = left + crop_w
    bottom = top + crop_h

    # Scale waypoints to pixel coordinates
    scaled_waypoints = waypoints * [w, h]

    # Adjust waypoints to the new cropped coordinate system
    scaled_waypoints[:, 0] -= left
    scaled_waypoints[:, 1] -= top

    # Discard waypoints outside the cropped area
    valid = (scaled_waypoints[:, 0] >= 0) & (scaled_waypoints[:, 0] < crop_w) & \
            (scaled_waypoints[:, 1] >= 0) & (scaled_waypoints[:, 1] < crop_h)
    if not all(valid):
        return image_list, waypoints

    # Normalize the updated waypoints back to [0, 1] in the cropped image
    normalized_waypoints = scaled_waypoints / [crop_w, crop_h]

    cropped_images = []
    for i, (image,flag) in enumerate(image_list):
        # Crop the image
        cropped_image = image[top:bottom, left:right]
        cropped_images.append((cropped_image,flag))

    return cropped_images, normalized_waypoints

def rotate_images_with_waypoints(image_list, waypoints, angle_range, background_color=(0, 0, 0)):
    """
    Perform image rotation on a list of input images and update normalized waypoints accordingly.
    
    Args:
        image_list (list of numpy.ndarray): List of input images in OpenCV format (H, W, C).
        waypoints (numpy.ndarray): Normalized waypoints with shape (N, 2) in [0, 1] range.
        angle (float): Rotation angle in degrees (positive for counter-clockwise).
        background_color (tuple): Background fill color (B, G, R).
        
    Returns:
        rotated_images (list of numpy.ndarray): Rotated images.
        updated_waypoints (numpy.ndarray): Updated normalized waypoints.
    """
    # Validate inputs
    assert len(image_list) > 0, "Image list cannot be empty"
    assert waypoints.shape[1] == 2, "Waypoints must have shape (N, 2)"
    # assert isinstance(image_list[0], np.ndarray), "Images must be numpy arrays"
    
    # Get image dimensions from the first image
    h, w = image_list[0][0].shape[:2]
    center = (w // 2, h // 2)
    
    angle = random.uniform(angle_range[0],angle_range[1])
    # Create rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate all images
    rotated_images = []
    for img,flag in image_list:
        rotated = cv2.warpAffine(img, rot_mat, (w, h), borderValue=background_color)
        rotated_images.append((rotated,flag))
    
    # Prepare homogeneous coordinates for waypoints transformation
    ones = np.ones((waypoints.shape[0], 1))
    scaled_waypoints = waypoints * [w, h]  # Convert to pixel coordinates
    homogenous_points = np.hstack([scaled_waypoints, ones])
    
    # Apply rotation transformation
    rotated_coords = homogenous_points @ rot_mat.T  # Matrix multiplication
    
    # Normalize coordinates back to [0, 1] range
    updated_waypoints = rotated_coords / [w, h]
    
    return rotated_images, updated_waypoints



class VLAConsumerDataset(Dataset):
    """A vision-languange-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """
    
    def __init__(
        self, 
        config,
        tokenizer,
        image_processor,
        num_cameras,
        img_history_size,
        image_size=None,
        auto_adjust_image_brightness=False,
        crop_rotate_aug = False,
        image_aug=False,
        dataset_type='pretrain',
        cond_mask_prob=0.1,
        cam_ext_mask_prob=-1.0,
        state_noise_snr=None,
        use_hdf5=False,
        use_precomp_lang_embed=False,
        train_datasets="all",
        dataset_split = 'train'
    ):
        super(VLAConsumerDataset, self).__init__()
        
        # Load the control frequency for each dataset
        # with open("configs/dataset_control_freq.json", 'r') as fp:
            # self.control_freq = json.load(fp)
        # Load the dataset names
        # dataset_names_cfg = 'configs/pretrain_datasets.json' \
        #     if dataset_type == 'pretrain' else 'configs/finetune_datasets.json'
        # with open(dataset_names_cfg, 'r') as file:
        #     DATASET_NAMES = json.load(file)
        # # Create the mapping between dataset name and id
        # self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        # self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}
        self.crop_rotate_aug = crop_rotate_aug
        
        self.image_processor = image_processor
        


        self.tokenizer_max_length = config["tokenizer_max_length"]
        self.image_aspect_ratio = config["image_aspect_ratio"]
        self.state_noise_snr = state_noise_snr
        self.num_cameras = num_cameras
        self.img_history_size = img_history_size
        self.cond_mask_prob = cond_mask_prob
        self.cam_ext_mask_prob = cam_ext_mask_prob
        self.use_hdf5 = use_hdf5
        self.hdf5_dataset = None
        self.split = dataset_split
        if use_hdf5:
            # self.hdf5_dataset = HDF5VLADataset()

            
            # real_dataset = AffordVLREALADataset(split=dataset_split)

            hoi4d_dataset = HOI4DWaypointFrameDataset(split=dataset_split,first_frame=True)
            hoi4d_dataset_new = HOI4DWaypointDataset(split=dataset_split,first_frame=True)
            droid_dataset = DroidWaypointDatasetCotracker(split=dataset_split,first_frame=True)
            droid_auto_dataset = DroidWaypointDataset(split=dataset_split,first_frame=True)
            sim_dataset = ManiskillWaypointDataset(split=dataset_split,first_frame=True)
            all_datasets = []
            if train_datasets == "all":
                all_datasets = [hoi4d_dataset,hoi4d_dataset_new,droid_dataset,sim_dataset,droid_auto_dataset]
            elif train_datasets == "hoi4d":
                all_datasets = [hoi4d_dataset,hoi4d_dataset_new]
            elif train_datasets == "droid":
                all_datasets = [droid_dataset,droid_auto_dataset]
            elif train_datasets == "maniskill":
                all_datasets = [sim_dataset]

            self.concatDataset = ConcatDataset(all_datasets)

        self.use_precomp_lang_embed = use_precomp_lang_embed
        if use_precomp_lang_embed:
            self.empty_lang_embed = torch.load("data/empty_lang_embed.pt")
        
        # Load dataset stat
        # with open("configs/dataset_stat.json", 'r') as f:
        #     dataset_stat = json.load(f)
        # self.dataset_stat = dataset_stat
        
        self.tokenizer = tokenizer
        # add special tokens
        # self.tokenizer.add_tokens(
        #     [
        #         DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN,DEFAULT_OFFSET_START_TOKEN,
        #         DEFAULT_OFFSET_END_TOKEN,DEFAULT_TEXT_START_TOKEN,DEFAULT_TEXT_END_TOKEN,
        #     ],
        #     special_tokens=True,
        # )
    
        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        self.image_aug = image_aug
        
        self.last_content = None
        self.last_meta = None
    
    # def get_dataset_name2id(self):
    #     return self.dataset_name2id
    
    # def get_dataset_id2name(self):
    #     return self.dataset_id2name
        
    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)


    def __len__(self) -> int:
        # return self.num_chunks * self.chunk_size
        return len(self.concatDataset)

    
    def __getitem__(self, index):
        # For robustness, we will try to load the data until we succeed
        while True:
            data_dict = None
            try:
                if self.use_hdf5:

                    res = self.concatDataset[index]
                    
                    content = res['meta']
                    instruction = content['instruction']
                    assert len(instruction) > 0, "instruction cannot be empty."

                    actions = res['actions']
                    
                    
                    # state_elem_mask = res['state_indicator']
                    image_metas = [
                        res['cam_high'], res['cam_high_mask'],
                        # res['cam_right_wrist'], res['cam_right_wrist_mask'],
                        # res['cam_left_wrist'], res['cam_left_wrist_mask'],
                    ]
                    # state_std = res['state_std']
                    # state_mean = res['state_mean']
                    # state_norm = res['state_norm']
                else:
                    raise Exception("Only support hdf5 keyword.")

                
                data_dict = {}
                data_dict['dataset_name'] = content['dataset_name']
                
                # We replace the invalid images with the background image
                # and also randomly mask images by the background image
                background_color = np.array([
                    int(x*255) for x in self.image_processor.image_mean
                ], dtype=np.uint8).reshape(1, 1, 3)
                background_image = np.ones((
                    self.image_processor.size["height"], 
                    self.image_processor.size["width"], 3), dtype=np.uint8
                ) * background_color
                
                image_metas = list(self.pairwise(image_metas))
                mask_probs = [self.cond_mask_prob] * self.num_cameras
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob
                rearranged_images = []
                for i in range(self.img_history_size):
                    for j in range(self.num_cameras):
                        images, image_mask = image_metas[j]
                        image, valid = images[i], image_mask[i]
                        if valid and (math.prod(image.shape) > 0) and \
                            (random.random() > mask_probs[j]):
                            rearranged_images.append((image, True))
                        else:
                            rearranged_images.append((background_image.copy(), False))
                

                if not content['data_augment'] and self.crop_rotate_aug:
                    # add flip augmentation
                    waypoints = actions[:,:]
                    # rearranged_images, waypoints = flip_image_with_waypoints(rearranged_images,)
                    # add random center crop augmentation

                    if random.random() < 0.15:
                        rearranged_images,waypoints = random_crop_images_with_waypoints(rearranged_images,waypoints,crop_ratio=0.85)
                    if random.random() < 0.15:
                        rearranged_images,waypoints = rotate_images_with_waypoints(rearranged_images,waypoints,(-30,30),(127,127,127))

                    actions[:,:] = waypoints
                data_dict["actions"] = actions

                preprocessed_images = []
                processor = self.image_processor
                num_valids = 0
                for image, valid in rearranged_images:
                    image = Image.fromarray(image)
                    if self.image_size is not None:
                        image = transforms.Resize(self.image_size)(image) # (1008, 336)
                    # assert image.height == 336, "We haven't prepare for training with images of different resolutions."
                    
                    if valid and self.auto_adjust_image_brightness:
                        pixel_values = list(image.getdata())
                        average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                        if average_brightness <= 0.15:
                            image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
                    # Only apply image augmentation to 20% of the images
                    if valid and self.image_aug and (random.random() > 0.8):
                        aug_type = random.choice([
                            "corrput_only", "color_only", "both"])
                        if aug_type != "corrput_only":
                            image = transforms.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                        if aug_type != "color_only":
                            image = image_corrupt(image)
                    
                    if self.image_aspect_ratio == 'pad':
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
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    preprocessed_images.append(image)
                data_dict["images"] = preprocessed_images
                # len ==2, [3, 384, 384]

                if self.use_precomp_lang_embed:
                    if content["instruction"][-1] == ".": 
                        content["instruction"] = content["instruction"][:-1]
                    data_dict["lang_embed"] = torch.load(content["instruction"]) \
                        if random.random() > self.cond_mask_prob else self.empty_lang_embed
                else:
                    instruction = content["instruction"] \
                        if random.random() > self.cond_mask_prob else "" # 有一定概率让instruction为空
                    tokenize_results = self.tokenizer(
                        instruction,
                        return_tensors="pt",
                        padding="longest",
                        truncation=False,
                    )
                    data_dict["input_ids"] = tokenize_results.input_ids[0].long() # 这一句是我加的，是否影响

                    assert len(data_dict["input_ids"]) <= self.tokenizer_max_length, \
                        f"Instruction length {len(data_dict['input_ids'])} exceeds the maximum length {self.tokenizer_max_length}."

                
                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v)

                for k, v in data_dict.items():
                    assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"
                        # data_dict[k] = torch.from_numpy(v)
        
                return data_dict
            except BaseException as e:
                # Print the error info
                if data_dict is not None:
                    print(f"Error catched when processing sample from {data_dict.get('dataset_name')}:", e)
                else:
                    print(f"Error catched when processing sample:", e)
                traceback.print_exc()
                # Try incresing the index
                index = (index + 1) % len(self)


class DataCollatorForVLAConsumerDataset(object):
    """Collate examples for supervised training."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            # "states": [],
            "actions": [],
            # "state_elem_mask": [],
            # "state_norm": [],
            "images": [],
            "data_indices": [],
            # "ctrl_freqs": []
        }
        input_ids = []
        lang_embeds = []
        lang_embed_lens = []
        
        for instance in instances:
            # Convert all the numpy arrays to tensor
            keys_to_check = [
                # 'states', 
                'actions',
                # 'state_elem_mask', 
                # 'state_norm',
            ]
            for key in keys_to_check:
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                else:
                    item = torch.from_numpy(instance[key])
                batch[key].append(item)

            if "input_ids" in instance:
                input_ids.append(instance["input_ids"])
            else:
                lang_embeds.append(instance["lang_embed"])
                lang_embed_lens.append(instance["lang_embed"].shape[0])
            
            batch["images"].append(torch.stack(instance["images"], dim=0))

        
        keys_to_stack = [

            'actions',

            "images"
        ]
        for key in keys_to_stack:

            batch[key] = torch.stack(batch[key], dim=0)
        
        # batch["ctrl_freqs"] = torch.tensor(batch["ctrl_freqs"])
    
        if len(input_ids) > 0:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            batch["input_ids"] = input_ids
            batch["lang_attn_mask"] = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            lang_embeds = torch.nn.utils.rnn.pad_sequence(
                lang_embeds,
                batch_first=True,
                padding_value=0)
            input_lang_attn_mask = torch.zeros(
                lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
            for i, l in enumerate(lang_embed_lens):
                input_lang_attn_mask[i, :l] = True
            batch["lang_embeds"] = lang_embeds
            batch["lang_attn_mask"] = input_lang_attn_mask
            
            
        return batch
