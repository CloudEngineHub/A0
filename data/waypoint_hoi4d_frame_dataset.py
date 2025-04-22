import os
import logging
import json
import yaml
import random

import copy

import numpy as np
import cv2
import csv



from torch.utils.data import Dataset

DATA_AUGMENT = False


def interpolate_and_add(dtraj2d):
    """
    Parameters:
        dtraj2d (numpy.ndarray): A (4,2) numpy array of points.

    Returns:
        numpy.ndarray: A (5,2) numpy array with the interpolated midpoint.
    """
    if dtraj2d.shape != (4, 2):
        raise ValueError("Input array must have shape (4,2)")
    
    # Compute the midpoint of the last two points
    midpoint = (dtraj2d[2] + dtraj2d[3]) / 2.0
    
    # Insert the midpoint at the 4th position
    dtraj2d_new = np.insert(dtraj2d, 3, midpoint, axis=0)
    
    return dtraj2d_new


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
        for image in image_list:
            # Perform horizontal flip
            flipped_image = cv2.flip(image, 1)
            images_augment.append(flipped_image)
        # Update x-coordinates of waypoints
        waypoints[:, 0] = 1 - waypoints[:, 0]
    
    elif flip_type == "vertical":
        for image in image_list:
            # Perform vertical flip
            flipped_image = cv2.flip(image, 0)
            images_augment.append(flipped_image)
        # Update y-coordinates of waypoints
        waypoints[:, 1] = 1 - waypoints[:, 1]
    
    elif flip_type == "both":
        for image in image_list:
            # Perform both horizontal and vertical flip
            flipped_image = cv2.flip(image, -1)
            images_augment.append(flipped_image)
        # Update both x and y coordinates of waypoints
        waypoints[:, 0] = 1 - waypoints[:, 0]
        waypoints[:, 1] = 1 - waypoints[:, 1]
    
    else:
        # No flip applied, return the original images
        images_augment = image_list.copy()

    return images_augment, waypoints


def random_crop_images_with_waypoints(image_list,depth_list, waypoints, crop_ratio=0.9):
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

    h, w, _ = image_list[0].shape  # Original image height, width
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
        return image_list,depth_list, waypoints

    # Normalize the updated waypoints back to [0, 1] in the cropped image
    normalized_waypoints = scaled_waypoints / [crop_w, crop_h]

    cropped_images = []
    for i, image in enumerate(image_list):
        # Crop the image
        cropped_image = image[top:bottom, left:right]
        cropped_images.append(cropped_image)

    cropped_depths = []
    for i, depth in enumerate(depth_list):
        cropped_depth = depth[top:bottom,left:right]
        cropped_depths.append(cropped_depth)

    return cropped_images,cropped_depths, normalized_waypoints

def rotate_images_with_waypoints(image_list,depth_list, waypoints, angle_range, background_color=(127, 127, 127)):
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
    h, w = image_list[0].shape[:2]
    center = (w // 2, h // 2)
    
    angle = random.uniform(angle_range[0],angle_range[1])
    # Create rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate all images
    rotated_images = []
    for img in image_list:
        rotated = cv2.warpAffine(img, rot_mat, (w, h), borderValue=background_color)
        rotated_images.append(rotated)
    
    rotated_depths = []
    for dpth in depth_list:
        rotated_d = cv2.warpAffine(dpth, rot_mat, (w, h), borderValue=(0,0,0))
        rotated_depths.append(rotated_d)
    
    # Prepare homogeneous coordinates for waypoints transformation
    ones = np.ones((waypoints.shape[0], 1))
    scaled_waypoints = waypoints * [w, h]  # Convert to pixel coordinates
    homogenous_points = np.hstack([scaled_waypoints, ones])
    
    # Apply rotation transformation
    rotated_coords = homogenous_points @ rot_mat.T  # Matrix multiplication
    
    # Normalize coordinates back to [0, 1] range
    updated_waypoints = rotated_coords / [w, h]
    
    return rotated_images,rotated_depths, updated_waypoints


class HOI4DWaypointFrameDataset(Dataset):
    """
    Starting from the original HOI4D dataset, we extracted waypoints for each image using event and object annotations from General Flow (https://arxiv.org/abs/2401.11439), 
    and then use molmo to anotate the start point.
    Finally, we manually selected the best-performing ones to form the dataset.
    """
    def __init__(self,split='train',first_frame=False):
        self.DATASET_NAME = "hoi4d_images"

        # Load the config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'base.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        self.split = split
        self.data_root = config['dataset']['hoi4d_frame_selection_path']
        self.points_path = os.path.join(self.data_root,"HOI4D_points")
        self.points_path_new = os.path.join(self.data_root,'HOI4D_points_molmo')

        self.json_path = os.path.join(os.path.dirname( self.points_path),'hoi4d_frame_metadata.json')
        self.csv_rectification_path = os.path.join(self.data_root, 'HOI4D_points_rectification.csv')

        self.origin_path = config['dataset']['hoi4d_rgb_path']
        # self.depth_path = '/mnt/data/Datasets/HOI4D_depth_video/'

        if not os.path.exists(self.points_path):
            raise FileNotFoundError(f"Points path not found: {self.data_root}")
        if not os.path.exists(self.origin_path):
            raise FileNotFoundError(f"Origin path not found: {self.origin_path}")

        with open(self.json_path, "r") as fp:
            metajson = json.load(fp)
            if split == 'train':
                self.metadata_origin = metajson['train'] + metajson['val']
            else:
                self.metadata_origin = metajson[split]


        self.metadata = self.load_and_filter_csv()
        logging.info("csv filter successfully!")

        self.data_num = len(self.metadata)
        logging.info("Totally {} samples in HOI4D_images {} set.".format(self.data_num, split))


    def load_and_filter_csv(self):
        filtered_metadata = []
        with open(self.csv_rectification_path, "r",encoding="utf-8-sig") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                
                # Check whether molmo, manual and base are 1
                if row['molmo'] == '1' or row['manual'] == '1' or row['base'] == '1':
                    # Obtain the corresponding metadata entry
                    metadata_entry = next((item for item in self.metadata_origin if item['id'] == int(row['id'])), None)

                    if metadata_entry:
                        if row['molmo'] == '1':
                            metadata_entry['source'] = 'molmo'
                        if row['manual'] == '1':
                            metadata_entry['source'] = 'manual'
                        if row['base'] == '1':
                            metadata_entry['source'] = 'base'

                        # rectify the instruction
                        meta = metadata_entry
                        action = metadata_entry['action']
                        object_name = metadata_entry['object']

                        if action =='pickup': action = 'pick up'
                        elif action =='putdown': action = 'put down'

                        object_name = 'the ' + object_name.lower()

                        instruction = action + ' ' + object_name
                        
                        if (meta['action']=='close' or meta['action']=='open') and meta['object']=='Storage Furniture' and \
                            "Body of the Cabinet" in meta["kpst_part"]:
                            if 'Drawer' in meta["kpst_part"]:
                                instruction = meta['action'] + " the drawer of the cabinet body in the storage furniture."
                            elif 'Door' in meta["kpst_part"]:
                                instruction = meta['action'] + " the door of the cabinet body in the storage furniture."
                        
                        ins = metadata_entry["instruction"]
                        if ins: instruction = ins

                        metadata_entry['instruction'] = instruction


                        # If the instruction in the CSV is not empty, update the instruction in the metadata
                        if row['instruction']:
                            metadata_entry['instruction'] = row['instruction']
                        filtered_metadata.append(metadata_entry)

        return filtered_metadata

    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def __len__(self):
        return self.data_num
    
    def get_item(self, index: int=None, state_only=False):
        return self.__getitem__(index)
    
    def __getitem__(self, idx: int=None):
        if idx is None:
            idx = random.randint(0, self.data_num-1)
        elif idx > self.data_num:
            raise ValueError(f"idx must be less than n_data={self.data_num}, but idx = {idx}")

        meta = self.metadata[idx]
  
        ins = meta["instruction"]
        if ins: instruction = ins

        data_fp_pure = '_'.join(self.metadata[idx]['index'].split(' '))
        img_idx = self.metadata[idx]['img']
        meta = self.metadata[idx]
        idd = meta['id']
        # traj_fp = os.path.join(self.data_root,'data', str(meta['id']),'kpst_traj.npy')

        point_source = meta['source']
        # read 2D waypoints trajectory
        # use molmo rectificatory
        if point_source == 'molmo':
            points_new_molmo = f'{idd}_points2d_molmo.npy'
            points_molmo_fp = os.path.join(self.points_path_new,self.split,points_new_molmo)
            if self.split == 'train' and not os.path.exists(points_molmo_fp):
                points_molmo_fp = os.path.join(self.points_path_new, 'val', points_new_molmo)
            if os.path.exists(points_molmo_fp):
                dtraj2d = np.load(points_molmo_fp)
        elif point_source == 'manual':
            points_new_manual = f'{idd}_points2d_manual.npy'
            points_manual_fp = os.path.join(self.points_path_new,self.split,points_new_manual)
            if self.split == 'train' and not os.path.exists(points_manual_fp):
                points_manual_fp = os.path.join(self.points_path_new, 'val', points_new_manual)
            if os.path.exists(points_manual_fp):
                dtraj2d = np.load(points_manual_fp)
        else:
            points_fp = os.path.join(self.points_path, self.split, data_fp_pure, str(img_idx).zfill(5), 'points2d.npy')
            if self.split == 'train' and not os.path.exists(points_fp):
                points_fp = os.path.join(self.points_path, 'val', data_fp_pure, str(img_idx).zfill(5), 'points2d.npy')
            if os.path.exists(points_fp):
                dtraj2d = np.load(points_fp) # np.ndarray shape [4,2]
                dtraj2d = dtraj2d[0,:,:] # [5,4,2]

        dtraj2d = dtraj2d.reshape(-1,2)
        # Five points were obtained through interpolation
        dtraj2d = interpolate_and_add(dtraj2d)
        
        imgs = []
        # depths = []
        for img_i in range(max(img_idx - self.IMG_HISORY_SIZE+1, 0), img_idx+1):

            rgb_fp = os.path.join(self.origin_path, data_fp_pure, 'align_rgb', str(img_i).zfill(5)+'.jpg')
            # depth_fp = os.path.join(self.depth_path, data_fp_pure, 'align_depth', str(img_i).zfill(5)+'.png')

            img = cv2.imread(rgb_fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            # depth = cv2.imread(depth_fp)
            # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            # depths.append(depth)

  
        H,W,_ = imgs[0].shape
        
        dtraj2d = dtraj2d.astype(np.float64)
        dtraj2d[:,0] /= float( W)  
        dtraj2d[:,1] /= float(H) 

        # add augmentation
        # if DATA_AUGMENT:
        #     if random.random() < 0.4:
        #     # imgs, dtraj2d = flip_image_with_waypoints(imgs,dtraj2d)
        #         
        #         imgs,depths, dtraj2d = random_crop_images_with_waypoints(imgs,depths,dtraj2d,crop_ratio=0.85)
        #     if random.random() < 0.4:
        #         imgs,depths,dtraj2d = rotate_images_with_waypoints(imgs,depths,dtraj2d,angle_range=(-30,30),background_color=(127,127,127))

        imgs = np.stack(imgs)
        # depths = np.stack(depths)

        # if image less than history_size
        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            # Pad the images using the first image
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)

        # if depths.shape[0] < self.IMG_HISORY_SIZE:
        #     depths = np.concatenate([
        #         np.tile(depths[:1], (self.IMG_HISORY_SIZE-depths.shape[0], 1, 1, 1)),
        #         depths
        #     ], axis=0)
 
        images = imgs


        cam_high_mask = np.array([True] * self.IMG_HISORY_SIZE)

        ONE_IMAGE = False
        if random.random() < 0.3:
            cam_high_mask = np.array([False,True])
            ONE_IMAGE = True

        actions = dtraj2d

        meta = {
            "dataset_name": self.DATASET_NAME,
            'data_path':data_fp_pure,
            'image_index':img_idx,

            "instruction": instruction,
            "step_id": img_idx,
            'id':self.metadata[idx]['id'],
            'object': self.metadata[idx]['object'],
            'data_augment':DATA_AUGMENT,
            'one_image':ONE_IMAGE
        }
        
        item ={
            'meta':meta,

            "actions": actions,

            'cam_high':images,
            "cam_high_mask": cam_high_mask,
        }

        return item



if __name__ == "__main__":
    dataset = HOI4DWaypointFrameDataset(split='train')
    print(len(dataset))
    # dataset.split_dataset()
