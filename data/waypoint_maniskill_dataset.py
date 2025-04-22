import os
import logging
import json
import yaml
import random

import numpy as np
import cv2

from torch.utils.data import Dataset



class ManiskillWaypointDataset(Dataset):
    def __init__(self,split='train',sample_rate=5,first_frame=False):
        self.DATASET_NAME = "maniskill"

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

        self.data_root=config['dataset']['maniskill_path']
        self.split_file = os.path.join(self.data_root, "train_test_split.json")
 

        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root path not found: {self.data_root}")
        

        self.json_files = self.load_json(self.data_root)

        if not os.path.exists(self.split_file):
            self.split_dataset()

        with open(self.split_file, "r") as f:
            split_data = json.load(f)
            self.json_files_split = split_data[split] 
        
        self.data_num = len(self.json_files_split)
        logging.info("Totally {} samples in ManiSkill {} set.".format(self.data_num, split))

    def load_json(self, file_path):
        valid_episodes = []

        for folder in os.listdir(file_path):
            folder_path = os.path.join(file_path, folder,'label')
            
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.json') and filename.startswith('out'):
                        file_path_i = os.path.join(folder_path, filename)
                        valid_episodes.append(file_path_i)
       
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
        sub_class = json_file.split('/')[-3]
        espisode_id = json_file.split('/')[-1].split('_')[-1][:-5]
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

        # print('waypoints',type(waypoints),waypoints)

        start_id = random.randint(start, end - self.CHUNK_SIZE)
        # if self.split == 'test':
        #     start_id = start # make sure the test is stable
        end_id = start_id + self.CHUNK_SIZE
        # dtraj2d = waypoints[start_id:end_id]
        # set sample rate
        num_points = self.CHUNK_SIZE 
        k = random.choice([self.sample_rate,self.sample_rate-1])
        end_id = start_id + (num_points - 1) * k
        
        length = len(waypoints) + start
        if end_id > length:
            end_id = length
        dtraj2d = waypoints[start_id-start:end_id+1-start:k]
        # add final point
        dtraj2d[-1] = waypoints[-1]

        dtraj2d = np.array(dtraj2d)
        dtraj2d = dtraj2d.reshape(-1,2)
        n = dtraj2d.shape[0] 
        if n < self.CHUNK_SIZE:
            last_point = dtraj2d[-1]
            num_points_to_add = num_points - dtraj2d.shape[0]
            points_to_add = np.tile(last_point, (num_points_to_add, 1))
            dtraj2d = np.vstack((dtraj2d, points_to_add))
            
        
        img_idx = start_id-start
        imgs = []
        for img_i in range(max(img_idx - self.IMG_HISORY_SIZE+1, 0), img_idx+1):
            image_path = image_paths[img_i]

            image_path = os.path.join(self.data_root, sub_class,image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imgs.append(image)
        
        # use the start first frame rather than previous frame
        if self.first_frame and self.IMG_HISORY_SIZE>1 and len(imgs)>1:
            imgs.pop(0)
            image_path = image_paths[0]
            image_path = os.path.join(self.data_root, sub_class,image_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imgs.insert(0,image)
        
        H,W,_ = imgs[0].shape
        dtraj2d = dtraj2d.astype(np.float64)
        dtraj2d[:,0] /= float( W)  # normalize to [0,1]
        dtraj2d[:,1] /= float(H) 

        imgs = np.stack(imgs)
        # if image less than history_size
        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            # Pad the images using the first image
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)

 
        images = imgs
        cam_high_mask = np.array([True] * self.IMG_HISORY_SIZE)
        
        ONE_IMAGE = False
        if random.random() < 0.4:
            cam_high_mask = np.array([False,True])
            ONE_IMAGE = True
        actions = dtraj2d

        meta = {
            "dataset_name": self.DATASET_NAME,
            "instruction": instruction,
            'data_augment':False,
            'sub_class':sub_class,
            'json_file_path':json_file,
            'one_image':ONE_IMAGE,
            'episode_id':int(espisode_id),
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
    Dataset = ManiskillWaypointDataset(split='train')
    Dataset.split_dataset()