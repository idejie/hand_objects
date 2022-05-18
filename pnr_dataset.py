from __future__ import absolute_import, division, print_function

import json

import h5py
from torch.utils.data.dataset import Dataset


class PNRDataset(Dataset):
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.inference = mode == "test"
        self.frames_num = (8 * self.opt.fps)
        self.info = []
        self.region_feature = []
        self.init_info()

    def init_info(self):
        with open(self.opt.train_info, 'r') as f:
            self.info = json.load(f)
        # todo:modify
        with h5py.File(self.opt.region_feature, 'r') as f:
            self.region_feature = f['dataset']

    def get_frames(self, clip_uid, start, end, pre_frame, pnr_frame, post_frame):
        all_frame = self.region_feature[clip_uid]
        frames = []
        for i in range(start, end):
            frames.append(all_frame[str(i)])
        return frames

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        info = self.info[index]
        clip_uid = info['clip_uid']
        start_frame = info['clip_start_frame']
        end_frame = info['clip_start_frame']
        pre_frame = info['pnr_frame_num'] - start_frame
        pnr_frame = info['pnr_frame_num'] - start_frame
        post_frame = info['pnr_frame_num'] - start_frame
        action_id = info['action_id']
        verb_id = info['verb_id']
        noun_id = info['noun_id']
        active_object = info['active_object']
        active_region = info['active_region']

        frames = self.get_frames(clip_uid, start_frame, end_frame, pre_frame, pnr_frame, post_frame)
        return frames, (pre_frame, pnr_frame, post_frame), (action_id, verb_id, noun_id), (active_object, active_region)
