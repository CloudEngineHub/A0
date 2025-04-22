import json
import random
import os
import logging
import yaml

import cv2
import numpy as np
from torch.utils.data import Dataset

def load_all_json_files(root_dir):
    """
    Recursively search for all .json files starting from the root_dir directory, read them, 
    and return a list containing all JSON data.
    """
    json_data_list = []

    # use os.walk
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith('.json') and not file_name.endswith('train_test_split.json'):
                file_path = os.path.join(root, file_name)
                json_data_list.append(file_path)
                # read JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    frame_data = data['frames']
                    for frame in frame_data:
                        if not frame['center'] :
                            json_data_list.remove(file_path)
                            break
    return json_data_list


class HOI4DWaypointDataset(Dataset):
    def __init__(self,split='train',first_frame=False,):
        self.DATASET_NAME = "hoi4d"
        # Load the config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'base.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        self.dataset_root = config['dataset']['hoi4d_metadata_path']
        self.image_dir_root = config['dataset']['hoi4d_rgb_path']
        self.split = split
        self.first_frame = first_frame
        if first_frame:
            print('use first frame rather than previous frame.')


        self.split_file = os.path.join(self.dataset_root, "train_test_split.json")

        # remove some actions in event
        self.remove_actions = ["rest","Reachout","Stop"]

        self.json_files = load_all_json_files(self.dataset_root)
        if not os.path.exists(self.split_file):
            self.split_dataset()

        with open(self.split_file, "r") as f:
            split_data = json.load(f)
            self.json_files_split = split_data[split] 

        # if self.split == 'train':
        #     try:
        #         self.json_files_split.remove(os.path.join(self.dataset_root,'ZY20210800001/H1/C20/N32/S292/s04/T3/metadata.json'))
        #         self.json_files_split.remove(os.path.join(self.dataset_root,'ZY20210800002/H2/C1/N48/S89/s04/T1/metadata.json'))
        #         self.json_files_split.remove(os.path.join(self.dataset_root,'ZY20210800002/H2/C2/N24/S2/s02/T1/metadata.json'))
        #     except Exception as e:
        #         print(f"Error removing files: {e}")
        # if self.split == 'test':
        #     self.json_files_split.remove(os.path.join(self.dataset_root,'ZY20210800003/H3/C3/N44/S284/s01/T2/metadata.json'))

        self.data_num = len(self.json_files_split)
        logging.info("Totally {} samples in HOI4D {} set.".format(self.data_num, split))

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
        json_file = self.json_files_split[idx]
        # print('json_file',json_file)
        json_file = os.path.join(self.dataset_root,json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            jsondata = json.load(f)
            imagedir = jsondata['image_dir']
            object_label = jsondata['label_name']
            task_name = jsondata['task_name']
            frames_list = jsondata['frames']
            events_frame = jsondata['events_frame']
        

        # Create a list of events not in remove_actions
        filtered_events = [event for event in events_frame if event["event"] not in self.remove_actions]
        num_events = len(filtered_events)
        while True:
            selected_event = random.choice(filtered_events)
            start_frame = selected_event["start_frame"]
            end_frame = selected_event["end_frame"]
            duration = end_frame - start_frame
            if duration -1 >= self.CHUNK_SIZE:
                break
        # if end_frame bigger than the last frame, then select the last frame
        if end_frame > len(frames_list):
            end_frame = len(frames_list)-1

        event = selected_event["event"]
        instruction = f'Task: {task_name}. {event} {object_label}.'

        if self.split == 'train':
            random_start_frame = random.randint(start_frame, end_frame - self.CHUNK_SIZE)
        elif self.split == 'test':
            random_start_frame = start_frame

        sample_points = list(map(int, np.linspace(random_start_frame, end_frame, self.CHUNK_SIZE)))
        waypoints = []
        for frame_idx in sample_points:
            try:
                frame_info = frames_list[frame_idx]
            except Exception as e:
                print(json_file)
                print('frame_idx:',frame_idx,'frame_list length:',len(frames_list),'start_frame',random_start_frame,'end_frame',end_frame)
                print(f"Unexpected error: {e}")

            pos = frame_info['center']
            waypoints.append(pos)
        # 
        waypoints = np.array(waypoints,dtype=np.float64)


        imgs = []
        img_idx = random_start_frame
        for img_i in range(max(img_idx - self.IMG_HISORY_SIZE+1, 0), img_idx+1):
            rgb_fp = os.path.join(self.image_dir_root,imagedir, f"{img_i:05d}.jpg")

            img = cv2.imread(rgb_fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
    
        # use the start first frame rather than previous frame
        if self.first_frame and self.IMG_HISORY_SIZE>1 and len(imgs)>1:
            imgs.pop(0)
            rgb_fp = os.path.join(self.image_dir_root,imagedir, f"{start_frame:05d}.jpg")
            image = cv2.imread(rgb_fp)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imgs.insert(0,image)

        H,W,_ = imgs[0].shape
        # normalize the x and y coordinates to [0,1]
        waypoints[:,0] /= float( W)
        waypoints[:,1] /= float(H) 

        imgs = np.stack(imgs)
        # if image less than history_size
        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            # Pad the images using the first image
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)

        # adapt the dat formate
        actions = waypoints
        cam_high_mask = np.array([True] * self.IMG_HISORY_SIZE)
        # With a certain probability, a single image is used
        if random.random() < 0.4:
            cam_high_mask = np.array([False,True])

        

        meta = {
            'data_augment': False,
            "dataset_name": self.DATASET_NAME,
            'image_dir':imagedir,
            'start_frame':img_idx,
            'label_name':object_label,
            'task_name':task_name,
            'event':event,
            'num_events': num_events,

            "instruction": instruction,


        }
        
        item ={
            'meta':meta,

            "actions": actions,

            'cam_high':imgs,
            "cam_high_mask": cam_high_mask,
        }

        return item



if __name__ == "__main__":

    dataset = HOI4DWaypointDataset(split='test')
    dataset.split_dataset()

