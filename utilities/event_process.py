import cv2
import numpy as np


def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    if events.shape[0] == 0:
        voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
        return voxel_grid

    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int32)
    ys = events[:, 2].astype(np.int32)
    pols = events[:, 3].astype(np.float32)
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int32)
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

def pure_events_to_voxel_grid(events, bins, w, h):

    event_t, event_x, event_y, event_p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    delta = (event_t[-1] - event_t[0]) / bins
    scer = []
    for i in range(bins):
        eidx = np.logical_and(event_t>=event_t[0]+delta*i, event_t<=event_t[0]+delta*(i+1))
        events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
        scer.append(events_to_voxel_grid(events, 1, w, h) )

    return np.squeeze(np.array(scer))

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




def events_to_count_map_and_time_surface(events, num_bins, width, height, timestamp_mid =0, cut_small=False):

    event_t, event_x, event_y, event_p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    delta = (event_t[-1] - event_t[0]) / num_bins

    if cut_small==False:

        evt_img_lst = []
        for i in range(num_bins):
            eidx = np.logical_and(event_t>=event_t[0]+delta*i, event_t<=event_t[0]+delta*(i+1))
            events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)

            evt_img_lst.append(event_to_cntts_img(events, height=height, width=width))
                    
        evt_img = np.concatenate(evt_img_lst)

    else:

        delta = (event_t[-1] - event_t[0]) / 20 

        small_delta = delta/num_bins

        evt_img_lst = []
        for i in range(num_bins//2):

            eidx = np.logical_and(event_t>=timestamp_mid+small_delta*(i-num_bins//2), event_t<=timestamp_mid)
            events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)

            evt_img_lst.append(event_to_cntts_img(events, height=height, width=width))

        for i in range(num_bins//2):

            eidx = np.logical_and(event_t>=timestamp_mid, event_t<=timestamp_mid+small_delta*(i+1))
            events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)

            evt_img_lst.append(event_to_cntts_img(events, height=height, width=width))

        evt_img = np.concatenate(evt_img_lst)

    return evt_img



def crop_events(events, y,x, crop_height, crop_width):
    mid =  events[(events[:, 2] >= y) & (events[:, 2] < y + crop_height) & (events[:, 1] >= x) & (events[:, 1] < x + crop_width)]

    mid[:, 1] -= x
    mid[:, 2] -= y


    return mid

# def crop_events(events, y,x, crop_height, crop_width):
#     event_t, event_x, event_y, event_p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

#     eidx1 = np.logical_and(event_x>=x, event_x<x+crop_width)
#     eidx2 = np.logical_and(event_y>=y, event_y<y+crop_height)
#     eidx = np.logical_and(eidx1, eidx2)
#     events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)

#     return events

def events_to_scer(events, timestamp_mid, w, h):
    event_t, event_x, event_y, event_p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    delta = (event_t[-1] - event_t[0]) / 6.

    scer = []
    eidx = np.logical_and(event_t>=event_t[0]+delta*0, event_t<=timestamp_mid)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h) * -1)

    eidx = np.logical_and(event_t>=event_t[0]+delta*1, event_t<=timestamp_mid)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h) * -1)

    eidx = np.logical_and(event_t>=event_t[0]+delta*2, event_t<=timestamp_mid)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h) * -1)

    eidx = np.logical_and(event_t>=timestamp_mid, event_t<=event_t[-1]-delta*2)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    eidx = np.logical_and(event_t>=timestamp_mid, event_t<=event_t[-1]-delta*1)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    eidx = np.logical_and(event_t>=timestamp_mid, event_t<=event_t[-1]-delta*0)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    return np.squeeze(np.array(scer))

def events_to_scer_ver2(events, timestamp_mid, w, h):
    event_t, event_x, event_y, event_p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    delta = (event_t[-1] - event_t[0]) / 6.

    delta1 = delta * 0.5
    delta2 = delta * 0.75
    delta3 = delta * 0.875


    scer = []
    eidx = np.logical_and(event_t>=event_t[0]+delta*0, event_t<=timestamp_mid)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h) * -1)

    eidx = np.logical_and(event_t>=event_t[0]+delta*1, event_t<=timestamp_mid)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h) * -1)

    eidx = np.logical_and(event_t>=event_t[0]+delta*2, event_t<=timestamp_mid)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h) * -1)

    eidx = np.logical_and(event_t>=event_t[0]+delta*2 + delta1, event_t<=timestamp_mid)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h) * -1)

    eidx = np.logical_and(event_t>=event_t[0]+delta*2 + delta2, event_t<=timestamp_mid)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h) * -1)

    eidx = np.logical_and(event_t>=event_t[0]+delta*2 + delta3, event_t<=timestamp_mid)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h) * -1)

    ##########################################################################################################

    eidx = np.logical_and(event_t>=timestamp_mid, event_t<=event_t[-1]-delta*2-delta3)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    eidx = np.logical_and(event_t>=timestamp_mid, event_t<=event_t[-1]-delta*2-delta2)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    eidx = np.logical_and(event_t>=timestamp_mid, event_t<=event_t[-1]-delta*2-delta1)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    eidx = np.logical_and(event_t>=timestamp_mid, event_t<=event_t[-1]-delta*2)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    eidx = np.logical_and(event_t>=timestamp_mid, event_t<=event_t[-1]-delta*1)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    eidx = np.logical_and(event_t>=timestamp_mid, event_t<=event_t[-1]-delta*0)
    events = np.stack((event_t[eidx], event_x[eidx], event_y[eidx], event_p[eidx]), axis=1).astype(np.float32)
    scer.append(events_to_voxel_grid(events, 1, w, h))

    return np.squeeze(np.array(scer))


def voxel2mask(voxel):
    mask_final = np.zeros_like(voxel[0, :, :])
    mask = (voxel != 0)
    for i in range(mask.shape[0]):
        mask_final = np.logical_or(mask_final, mask[i, :, :])
    # to uint8 image
    mask_img = mask_final * np.ones_like(mask_final) * 255
    mask_img = mask_img[..., np.newaxis] # H,W,C
    mask_img = np.uint8(mask_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_img_close = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    # print(mask_img_close.shape)
    mask_img_close = mask_img_close[np.newaxis,...] # H,W -> C,H,W  C=1
    return mask_img_close

def events_reversal(events, need_reversal=False):
    events = events.copy()

    deltaT = events[-1, 0] - events[0, 0]
    events[:, 0] = (events[:, 0] - events[0, 0]) / deltaT  # [0,1]
    events[:, 3][events[:, 3] == 0] = -1
    if need_reversal is False:
        return events
    else:
        # reverse timestamps
        events[:, 0] = 1 - events[:, 0]  # [1:0]
        # reverse porlarity
        events[:, 3] = events[:, 3] * -1
        events = np.flip(events, axis=0)
        return events