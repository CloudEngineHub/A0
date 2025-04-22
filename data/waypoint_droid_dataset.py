import os
import logging
import json
import yaml
import random

import numpy as np
import cv2

from torch.utils.data import Dataset

import pandas as pd


def filter_valid_episodes(split_file, csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file,on_bad_lines="skip")
    
    # Collect all episodes marked as valid (valid == 1)
    valid_episodes = set(df[df['valid'] == 1]['episode'].astype(str))
    
    # read train_test_split.json
    with open(split_file, "r") as f:
        split_data = json.load(f)
    
    # Filter out invalid episodes from both train and test lists
    def is_valid_episode(filename):
        episode_num = filename.split('_')[-1].split('.')[0]  
        return episode_num in valid_episodes
    
    split_data['train'] = [f for f in split_data['train'] if is_valid_episode(f)]
    split_data['test'] = [f for f in split_data['test'] if is_valid_episode(f)]
    
    # update JSON file
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=4)
    

class DroidWaypointDataset(Dataset):
    def __init__(self,split='train',sample_rate=5,first_frame=False):
        self.DATASET_NAME = "droid_molmo_sam2"

        # Load the config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'base.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        self.split = split
        self.sample_rate = sample_rate
        self.first_frame = first_frame
        if first_frame:
            print('use first frame rather than previous frame.')

        self.data_root= config['dataset']['droid_molmo_sam2_path']
        self.split_file = os.path.join(self.data_root, "train_test_split_droid.json")
        # self.verify_csv = "/mnt/data/xurongtao/datasets/droid_video_track_verify_clean-sheet1.csv"
 

        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root path not found: {self.data_root}")
        

        self.json_files = self.load_json(self.data_root)


        if not os.path.exists(self.split_file):
            self.split_dataset(train_ratio=0.9)

        # filter_valid_episodes(self.split_file, self.verify_csv)


        with open(self.split_file, "r") as f:
            split_data = json.load(f)
            self.json_files_split = split_data[split] 

        
        self.data_num = len(self.json_files_split)
        logging.info("Totally {} samples in DROID_molmo_sam2 {} set.".format(self.data_num, split))

    def load_json(self, file_path):
        valid_episodes = []
        for filename in os.listdir(file_path):

            if filename.endswith('.json') and filename.startswith('episode'):
                file_path_i = os.path.join(file_path, filename)

                with open(file_path_i, 'r') as file:
                    data = json.load(file)

                    if data.get('metadata', {}).get('is_valid', True):
                        valid_episodes.append(filename)
       
        return valid_episodes


    def split_dataset(self, train_ratio=0.8, shuffle=True):
        if shuffle:
            random.shuffle(self.json_files)  # Shuffle the dataset
        
        split_index = int(len(self.json_files) * train_ratio)
        train_files = self.json_files[:split_index]
        test_files = self.json_files[split_index:]

        split_data = {
            "train": train_files,
            "test": test_files
        }

        # Save the split result to a JSON file
        with open(self.split_file, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=4)

        print(f"Dataset split completed: {len(train_files)} training files, {len(test_files)} testing files")


    def __len__(self):
        return self.data_num

    def get_item(self, index: int=None, state_only=False):
        return self.__getitem__(index)
    
    def __getitem__(self, idx: int=None):
        if idx is None:
            idx = random.randint(0, self.data_num-1)
        elif idx > self.data_num:
            raise ValueError(f"idx must be less than n_data={self.data_num}, but idx = {idx}")
        json_file = self.json_files_split[idx]
        json_file = os.path.join(self.data_root, json_file)

        with open(json_file, 'r') as file:
            jsondata = json.load(file)
            metadata = jsondata.get('metadata', {})
            waypoints = jsondata['waypoints2d']
            image_paths = jsondata['cam_highs']
        valid_range = metadata['validRange']
        start = valid_range['start']
        end = valid_range['end']
        instruction = metadata['instructions']
        assert isinstance(instruction, str), "instructions should be a string"

        start_id = random.randint(start, end - self.CHUNK_SIZE)
        end_id = start_id + self.CHUNK_SIZE
        # dtraj2d = waypoints[start_id:end_id]
        # set sample rate
        num_points = self.CHUNK_SIZE
        k = random.choice([self.sample_rate,self.sample_rate-1])
        end_id = start_id + (num_points - 1) * k

        length = len(waypoints)
        if end_id > length:
            end_id = length
        dtraj2d = waypoints[start_id:end_id+1:k]
        ##
        
        dtraj2d = np.array(dtraj2d)
        dtraj2d = dtraj2d.reshape(-1,2)
        n = dtraj2d.shape[0] 
        if n < self.CHUNK_SIZE:
            last_point = dtraj2d[-1]
            num_points_to_add = num_points - dtraj2d.shape[0]
            points_to_add = np.tile(last_point, (num_points_to_add, 1))
            dtraj2d = np.vstack((dtraj2d, points_to_add))
            
        
        img_idx = start_id
        imgs = []
        for img_i in range(max(img_idx - self.IMG_HISORY_SIZE+1, 0), img_idx+1):
            image_path = image_paths[img_i]
            image_path = os.path.join(self.data_root, image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imgs.append(image)
        
        if self.first_frame and self.IMG_HISORY_SIZE>1 and len(imgs)>1:
            imgs.pop(0)
            image_path = image_paths[start]
            image_path = os.path.join(self.data_root, image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imgs.insert(0,image)



        dtraj2d = dtraj2d.astype(np.float64)


        imgs = np.stack(imgs)
        # if image less than history_size
        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            # Pad the images using the first image
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)

        # adapt data format
        images = imgs
        cam_high_mask = np.array([True] * self.IMG_HISORY_SIZE)
        # With a certain probability, a single image is used.
        ONE_IMAGE = False
        if random.random() < 0.4:
            cam_high_mask = np.array([False,True])
            ONE_IMAGE = True
        actions = dtraj2d

        meta = {
            "dataset_name": self.DATASET_NAME,
            "instruction": instruction,
            'data_augment':False,
            'sub_class':None,
            'json_file_path':json_file,
            'one_image':ONE_IMAGE
        }
        meta.update(metadata)
        
        item ={
            'meta':meta,
            "actions": actions,

            'cam_high':images,
            "cam_high_mask": cam_high_mask,
        }

        return item

if __name__ == "__main__":
    Dataset = DroidWaypointDataset(split='train')
    print(len(Dataset))
    # Dataset.split_dataset()