import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


from utilities.event_process import events_to_count_map_and_time_surface,events_to_scer,crop_events,events_to_scer_ver2,pure_events_to_voxel_grid


class Dataloader_test(Dataset):
    def __init__(self, args):
        super(Dataloader_test, self).__init__()
        self.args = args
        self.file_names = self.readFilePaths(suffix='.png')

    def __len__(self):
        return len(self.file_names)//self.args.test_range
    
    def __getitem__(self, idx):

        idx = idx * self.args.test_range
        """ -------------------- load all data -------------------- """
        blur,  all_events, timestamp_mid, prefix = load_data_single(self.args, self.file_names, idx, 'test')

        blur = np.transpose(blur, (2,0,1))  # [h,w,3] -> [3,h,w]
        #sharp_mid = np.transpose(sharp_mid, (2,0,1))
    
        
        """ -------------------- process event -------------------- """ 

        _,h,w = blur.shape
            
        if self.args.event_type == 'voxel_grid':
            new_event = events_to_scer(all_events, timestamp_mid, w, h)
        elif self.args.event_type == 'scer2':
            new_event = events_to_scer_ver2(all_events, timestamp_mid, w, h)
        elif self.args.event_type == 'pure_events_to_voxel_grid':
            new_event = pure_events_to_voxel_grid(all_events, 24, w, h)
        else:
            new_event = events_to_count_map_and_time_surface(all_events, 6, w, h)
        
        #smaller_event = events_to_count_map_and_time_surface(all_events, 6, w,h,timestamp_mid,True)

        blur = torch.from_numpy(blur).float() / 255.

        #sharp_mid = torch.from_numpy(sharp_mid).float() / 255.

        new_event = torch.from_numpy(new_event).float()

        #smaller_event = torch.from_numpy(smaller_event).float()

        return blur,new_event, prefix
    def readFilePaths(self, suffix='.png'):
        file_names = []
        path = os.path.join(self.args.dataset_path, self.args.dataset_name, 'test', 'blur')
        for pair in sorted(os.listdir(path)):
            if os.path.splitext(pair)[-1] == suffix:
                file_names.append(pair[:-4])
        return file_names

def load_data_single(args, file_names, idx, split='test',need_sharp_mid=False,gray=False):

    dir_blur = os.path.join(args.dataset_path, args.dataset_name, split, 'blur')
    if gray:
        blur = cv2.imread(os.path.join(dir_blur, file_names[idx] + '.png'), cv2.IMREAD_GRAYSCALE)
        blur = np.expand_dims(blur, axis=2)
        blur = np.repeat(blur, 3, axis=2)

    else:
        blur = cv2.imread(os.path.join(dir_blur, file_names[idx] + '.png'))
     

    if need_sharp_mid:
        dir_sharp_mid = os.path.join(args.dataset_path, args.dataset_name, split, 'sharp_mid')
        sharp_mid = cv2.imread(os.path.join(dir_sharp_mid, file_names[idx] + '.png'))
        

    dir_event = os.path.join(args.dataset_path, args.dataset_name, split, 'event')
    list_event = sorted([os.path.join(dir_event, file_names[idx] + '_' + str(i).zfill(2) + '.npz') for i in range(10)])  # remove the last one because it is out of the exposure time
    events_split = [np.load(path_event) for path_event in list_event]
    all_events = np.zeros((0, 4)).astype(np.float32)  # npz -> ndarray
    for event in events_split:
        ### IMPORTANT: dataset mistake x and y !!!!!!!!
        ###            Switch x and y here !!!!
        y = event['x'].astype(np.float32)            
        x = event['y'].astype(np.float32)          
        t = event['timestamp'].astype(np.float32)
        p = event['polarity'].astype(np.float32)

        this_events = np.concatenate((t,x,y,p),axis=1) # N,4
        all_events = np.concatenate((all_events, this_events), axis=0)
    timestamp_mid = (events_split[4]['timestamp'][-1] + events_split[5]['timestamp'][0]) / 2.0

    prefix = file_names[idx]

    if need_sharp_mid:
        return blur,  sharp_mid, all_events, timestamp_mid, prefix
    else:
        return blur,  all_events, timestamp_mid, prefix


