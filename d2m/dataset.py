import torch
import torch.utils.data
import torch.nn.functional as F
import math
from torch.utils.data import Dataset
import numpy as np
import io
import os
import json
import random
from pathlib import Path
from librosa.core import load
from librosa.util import normalize


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


class S25Dataset(Dataset):
    def __init__(self, audio_files, motion_files, video_files, genre_label, augment, config):
        self.config = config
        self.sampling_rate = config['sample_rate']
        self.segment_length = self.sampling_rate * config['segment_length']
        self.audio_files = files_to_list(audio_files)
        self.audio_files = [Path(audio_files).parent / x for x in self.audio_files]
        self.augment = True
        self.video_files = files_to_list(video_files)
        self.video_files = [Path(video_files).parent / x for x in self.video_files]
        self.motion_files = files_to_list(motion_files)
        self.motion_files = [Path(motion_files).parent / x for x in self.motion_files]
        self.use_genre = config['genre_config']['use_genre']
        if self.use_genre:
            self.genre = np.load(genre_label)

    def __len__(self): 
        return len(self.audio_files)
 
    def __getitem__(self, index):
        # Read audio
        audio_filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(audio_filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - audio.size(0)), "constant").data

        # Read video
        video_filename = self.video_files[index]
        video = self.load_img_to_torch(video_filename)
        video_max_len = self.config['video_context_length']
        if video.size(0) > video_max_len:
            video = video[:video_max_len]
        # Read motion
        motion_filename = self.motion_files[index]
        motion = np.load(motion_filename)
        motion_max_len = self.config['motion_context_length']
        if len(motion) > motion_max_len:
            need_to_delete_num = len(motion) - motion_max_len
            delete_rate = math.ceil(len(motion) / need_to_delete_num)
            motion_tmp = [
                num for i, num in enumerate(motion)
                if i % delete_rate != 0
            ]
            motion = np.array(motion_tmp)[:motion_max_len]
        motion = torch.from_numpy(motion).float()
        assert motion_max_len == motion.size(0), f"len motion: {len(motion)}"

        if self.use_genre:
            genre = self.genre[index]
            data = {
                    'music': audio.unsqueeze(0),
                    'motion': motion,
                    'video': video,
                    'genre': genre,
            }
        else:
            data = {
                    'music': audio.unsqueeze(0),
                    'motion': motion,
                    'video': video,
            }
        return data

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)            # mono: (sr*len,), binaural: (2, sr*len)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate

    def load_img_to_torch(self, full_path):
        data = np.load(full_path)
        return torch.from_numpy(data).float()