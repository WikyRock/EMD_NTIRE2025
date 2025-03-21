import numpy as np
import os
import cv2

def event_count_map(event, height=180, width=320):
    if len(event) == 0:
        return np.zeros((2, height, width), dtype=np.float32)
    t = event[:, 0].copy().astype(np.float64)
    x = event[:, 1].copy().astype(np.uint32)
    y = event[:, 2].copy().astype(np.uint32)
    p = event[:, 3].copy().astype(np.uint8)

    # normalize t
    t -= t[0]
    if t[-1] != 0:
        t /= t[-1]


    pos_event_x = x[p == 1]
    pos_event_y = y[p == 1]


    neg_event_x = x[p < 1]
    neg_event_y = y[p < 1]

    event_cnt_img_pos = np.zeros(height * width, dtype=np.float32)
    event_cnt_img_neg = np.zeros(height * width, dtype=np.float32)
    np.add.at(event_cnt_img_pos, pos_event_y * width + pos_event_x, 1)
    event_cnt_img_pos = event_cnt_img_pos.reshape([height, width])
    np.add.at(event_cnt_img_neg, neg_event_y * width + neg_event_x, 1)
    event_cnt_img_neg = event_cnt_img_neg.reshape([height, width])
    event_cnt_img = np.stack((event_cnt_img_pos, event_cnt_img_neg), 0)

    return event_cnt_img


def get_voxel_SCER(events, num_bins, width, height):
    
    if len(events) == 0:
        return np.zeros((num_bins, height, width), dtype=np.float32)
    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    E = np.zeros((num_bins, 2, height, width))
    
    

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0
        
        
    interval_time = deltaT / num_bins  
    new_t = events[:, 0] - first_stamp  
    idx = np.floor(new_t/interval_time).astype(np.int8)    
    idx[idx == num_bins] -= 1
        
    
    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)
    p = events[:, 3].astype(np.int32)
    p[p == 0] = -1  # polarity should be +1 / -1
    
    np.add.at(E, (idx, p, y, x), 1)
  
    
    voxel = E[:, 0, :, :] - E[:, 1, :, :]
    re_voxel = np.zeros_like(voxel)
    left_voxel = voxel[:num_bins//2, :, :]
    right_voxel = voxel[num_bins//2:, :, :]
    right_voxel_sum = np.cumsum(right_voxel, axis=0)
    left_voxel = left_voxel[::-1]
    left_voxel_sum = np.cumsum(left_voxel, axis=0)
    left_voxel_sum = left_voxel_sum[::-1]
    re_voxel[:num_bins//2, :, :] = -left_voxel_sum
    re_voxel[num_bins//2:, :, :] = right_voxel_sum
    
    
    return re_voxel
    
    

def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """
    if len(events) == 0:
        return np.zeros((num_bins, height, width), dtype=np.float32)
    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    ts = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    xs = events[:, 1].astype(np.int32)
    ys = events[:, 2].astype(np.int32)
    pols = events[:, 3].copy()
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int_)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


def event_to_cntts_img(event, height=180, width=320):
    if len(event) == 0:
        return np.zeros((4, height, width), dtype=np.float32)
    t = event[:, 0].copy().astype(np.float64)
    x = event[:, 1].copy().astype(np.uint32)
    y = event[:, 2].copy().astype(np.uint32)
    p = event[:, 3].copy().astype(np.uint8)

    # normalize t
    t -= t[0]
    if t[-1] != 0:
        t /= t[-1]

    pos_event_t = t[p == 1]
    pos_event_x = x[p == 1]
    pos_event_y = y[p == 1]

    neg_event_t = t[p < 1]
    neg_event_x = x[p < 1]
    neg_event_y = y[p < 1]

    event_cnt_img_pos = np.zeros(height * width, dtype=np.float32)
    event_cnt_img_neg = np.zeros(height * width, dtype=np.float32)
    np.add.at(event_cnt_img_pos, pos_event_y * width + pos_event_x, 1)
    event_cnt_img_pos = event_cnt_img_pos.reshape([height, width])
    np.add.at(event_cnt_img_neg, neg_event_y * width + neg_event_x, 1)
    event_cnt_img_neg = event_cnt_img_neg.reshape([height, width])
    event_cnt_img = np.stack((event_cnt_img_pos, event_cnt_img_neg), 0)

    event_time_img = np.zeros((2, height, width), dtype=np.float32)
    event_time_img[0, pos_event_y, pos_event_x] = pos_event_t
    event_time_img[1, neg_event_y, neg_event_x] = neg_event_t

    return np.concatenate((event_cnt_img, event_time_img), 0)

def event_to_cntts_img_2times(event, height=180, width=320):
    if len(event) == 0:
        return np.zeros((4, height, width), dtype=np.float32)
    t = event[:, 0].copy().astype(np.float64)
    x = event[:, 1].copy().astype(np.uint32)
    y = event[:, 2].copy().astype(np.uint32)
    p = event[:, 3].copy().astype(np.uint8)

    # normalize t
    t -= t[0]
    if t[-1] != 0:
        t /= t[-1]

    pos_event_t = t[p == 1]
    pos_event_x = x[p == 1]
    pos_event_y = y[p == 1]

    neg_event_t = t[p < 1]
    neg_event_x = x[p < 1]
    neg_event_y = y[p < 1]

    event_cnt_img_pos = np.zeros(height * width, dtype=np.float32)
    event_cnt_img_neg = np.zeros(height * width, dtype=np.float32)
    np.add.at(event_cnt_img_pos, pos_event_y * width + pos_event_x, 1)
    event_cnt_img_pos = event_cnt_img_pos.reshape([height, width])
    np.add.at(event_cnt_img_neg, neg_event_y * width + neg_event_x, 1)
    event_cnt_img_neg = event_cnt_img_neg.reshape([height, width])
    event_cnt_img = np.stack((event_cnt_img_pos, event_cnt_img_neg), 0)

    event_time_img = np.zeros((2, height, width), dtype=np.float32)
    event_time_img[0, pos_event_y, pos_event_x] = pos_event_t
    event_time_img[1, neg_event_y, neg_event_x] = neg_event_t

    event_time_img_old = np.zeros((2, height, width), dtype=np.float32)
    
    # 对于正极性事件，使用 np.minimum.at 来保留最早的时间
    np.minimum.at(event_time_img_old[0], (pos_event_y, pos_event_x), pos_event_t)
    # 对于负极性事件，使用 np.minimum.at 来保留最早的时间
    np.minimum.at(event_time_img_old[1], (neg_event_y, neg_event_x), neg_event_t)

    return np.concatenate((event_cnt_img, event_time_img,event_time_img_old), 0)

def events_to_scer(events, w, h):

    scer = []
    evt_lst = split_events_by_time(events, None, 6 )

    events = np.vstack((evt_lst[0], evt_lst[1], evt_lst[2])).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h) * -1)

    events = np.vstack(( evt_lst[1], evt_lst[2])).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h) * -1)
  
    events =evt_lst[2].astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h)* -1)
    
    events =evt_lst[3].astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    events = np.vstack((evt_lst[3], evt_lst[4])).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    events = np.vstack((evt_lst[3], evt_lst[4],evt_lst[5])).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    return np.concatenate(scer, 0)





def split_events_by_time(events, split_time_lst=None, split_num=6, start_ts=None, end_ts=None):
    """
    :param events: raw events [n, 4] t, x, y, z
    :param split_num: int
    """
    if len(events) == 0:
        start_ts = 0
        end_ts = 0
    else:
        start_ts = events[0, 0] if start_ts is None else float(start_ts)
        end_ts = events[-1, 0] if end_ts is None else float(end_ts)
    
    if split_time_lst is None:
        split_time_lst = [(end_ts - start_ts) * (i + 1) / split_num + start_ts for i in range(split_num - 1)]

    split_idx_lst = [0]
    for split_time in split_time_lst:
        split_idx = np.searchsorted(events[:, 0], split_time)
        split_idx_lst.append(split_idx)
    split_idx_lst.append(-1)

    events_lst = []
    for i in range(len(split_idx_lst) - 1):
        start_idx = split_idx_lst[i]
        end_idx = split_idx_lst[i + 1]
        events_split = events[start_idx:end_idx, :]
        events_lst.append(events_split)

    return events_lst

def reverse(events):
    """Reverse temporal direction of the event stream.
    Polarities of the events reversed.
                        (-)       (+) 
    --------|----------|---------|------------|----> time
        t_start        t_1       t_2        t_end

                        (+)       (-) 
    --------|----------|---------|------------|----> time
            0    (t_end-t_2) (t_end-t_1) (t_end-t_start)        
    """
    events_ = np.copy(events)
    if len(events_) == 0:
        return events_
    events_[:, 0] = events_[-1, 0] - events_[:, 0]
    events_[:, 3] = 1 - events_[:, 3]
    # Flip rows of the 'features' matrix, since it is sorted in oldest first.
    return np.copy(np.flipud(events_))

import torch
import cv2
class ImagePreviewer:
    """concatenate image arrays into one big image
    """
    def __init__(self, cols=-1, min_max=(0, 255)):
        self.cols = cols
        self.img_lst = []
        self.min_max = min_max

    def add_image(self, title, img):
        """
        title(str): the title of sub-image
        img(tensor/ndarray): the image array of shape [hw], [hw3], [hw1], [1hw], [3hw]
        """
        # convert image into ndarray
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        elif type(img) is not np.ndarray:
            raise ValueError(f"Unsupported image type of", type(img))

        img = img.clip(*self.min_max)
            
        if len(img.shape) == 2:     # [hw]
            img = img[:,:,None].repeat(3,2)
        elif len(img.shape) == 3 and img.shape[2] != 3:
            if img.shape[0] == 1:   # [1hw]
                img = img.transpose(1,2,0).repeat(3,2)
            elif img.shape[0] == 3: # [3hw]
                img = img.transpose(1,2,0)
            elif img.shape[2] == 1: # [hw1]
                img = img.repeat(3,2)
            else:
                raise ValueError(f"Unsupported image shape of", img.shape)

        h, w, _ = img.shape
        title_img = np.zeros((10, w, 3))
        title_img = cv2.putText(
            title_img,
            title,
            org=(0, 10),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(255, 255, 255),
            bottomLeftOrigin=False)
        self.img_lst.append(np.concatenate([title_img, img], axis=0))

    def make_img(self):
        if len(self.img_lst) == 0:
            return None
        else:
            for img in self.img_lst:
                assert img.shape == self.img_lst[0].shape

        if self.cols == -1:
            img_preview = np.concatenate(self.img_lst, 1)
        else:
            zero_num = - len(self.img_lst) % self.cols
            self.img_lst.extend([np.zeros_like(self.img_lst[0])] * zero_num)
            row_lst, cur_lst, i = [], [], 0
            while i < len(self.img_lst):
                cur_lst.append(self.img_lst[i])
                i += 1
                if i % self.cols == 0:
                    row_lst.append(np.concatenate(cur_lst, 1))
                    cur_lst = []
            img_preview = np.concatenate(row_lst, 0)

        return img_preview


'''
def wiky_make_event_preview(event_img):

    N,H,W = event_img.shape
    cnt_img = np.zeros((event_img.shape[1], event_img.shape[2]), dtype=np.float32)

    for i in range(N/4):
'''        

#made by wiky luo 2024-1-22 17:21:42
def make_event_and_image_together(event_img, img):
    def preview_for_polarity(event_img,img):
        preview = img.astype(np.uint8)
        b = preview[:, :, ...]
        r = preview[:, :, ...]
        b[event_img > 0] = [255,0,0]
        r[event_img < 0] = [0,0,255]
        return preview    
        
    cnt_img_lst = []
     
    for i in range(len(event_img) // 4):
        cnt_img_lst.append(event_img[4 * i] - event_img[4 * i + 1])
            
    return [preview_for_polarity(cnt_img_lst[i],img[i]) for i in range(len(cnt_img_lst))]
    

    
#import matplotlib.pyplot as plt
def make_event_preview(event_img, type='voxel', sum=True):
    def preview_for_polarity(img):
        preview = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        b = preview[:, :, 0]
        r = preview[:, :, 2]
        b[img > 0] = 255
        r[img < 0] = 255
        return preview

    #cmp = plt.get_cmap()

    if type == 'voxel':
        if sum is True:
            return preview_for_polarity(np.sum(event_img, axis=0))
        else:
            return [preview_for_polarity(img) for img in event_img]
    elif type == 'cntts':
        if sum is True:
            cnt_img = np.zeros((event_img.shape[1], event_img.shape[2]), dtype=np.float32)
            ts_img = np.zeros((event_img.shape[1], event_img.shape[2]), dtype=np.float32)
            for i in range(len(event_img) // 4):
                cnt_img += (event_img[4 * i] - event_img[4 * i + 1])
                ts_img = np.amax(np.stack((ts_img, event_img[4 * i + 2], event_img[4 * i + 3])), axis=0)
            return preview_for_polarity(cnt_img)#, (cmp(ts_img)[:, :, :3] * 255).astype(np.uint8)
        else:
            cnt_img_lst = []
            ts_img_lst = []
            for i in range(len(event_img) // 4):
                cnt_img_lst.append(event_img[4 * i] - event_img[4 * i + 1])
                ts_img_lst.append(np.amax(np.stack((event_img[4 * i + 2], event_img[4 * i + 3])), axis=0))

            return [preview_for_polarity(img) for img in cnt_img_lst]#, \
                #[(cmp(img)[:, :, :3] * 255).astype(np.uint8) for img in ts_img_lst]
    else:
        raise ValueError("type :{} is not supported.".format(type))