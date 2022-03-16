import os
import random
from abc import abstractmethod

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
import transformers
from base.base_dataset import get_video_len, video_reader


class TextVideoDataset(Dataset):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='decord'
                 ):
        self.dataset_name = dataset_name
        self.text_params = text_params
        self.video_params = video_params
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        if metadata_dir is not None:
            self.metadata_dir = os.path.expandvars(metadata_dir)
        else:
            self.metadata_dir = self.data_dir
        self.split = split
        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.video_reader = video_reader[reader]
        self.label_type = 'caption'
        self._load_metadata()
        if self.sliding_window_stride != -1:
            if self.split != 'test':
                raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            self._fix_temporal_samples()
        
        # build tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(text_params['model'],
                                                           TOKENIZERS_PARALLELISM=False)

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError("Metadata loading must be implemented by subclass")

    @abstractmethod
    def _get_video_path(self, sample):
        raise NotImplementedError("Get video path function must be implemented by subclass")

    def _get_caption(self, sample):
        raise NotImplementedError("Get caption function must be implemented by subclass")

    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata['vlen'] = self._get_video_lens()
        self.metadata['frame_intervals'] = self.metadata['vlen'].apply(
            lambda x: np.linspace(start=0, stop=x, num=min(x, self.video_params['num_frames']) + 1).astype(int))
        self.metadata['fix_start'] = self.metadata['frame_intervals'].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode('fix_start')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        # TODO-- tokenize in dataset 
        caption = self._get_caption(sample)
        # caption = self.tokenizer(self._get_caption(sample), return_tensors='pt', padding=True, truncation=True)
        # print(caption['input_ids'].shape, caption['attention_mask'].shape)

        video_loading = self.video_params.get('loading', 'strict')
        frame_sample = 'rand'
        fix_start = None
        if self.split == 'test':
            frame_sample = 'uniform'
        if self.sliding_window_stride != -1:
            fix_start = sample['fix_start']

        try:
            if os.path.isfile(video_fp):
                imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
                                               fix_start=fix_start)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        # (video, text, meta) 
        # return final, caption, meta_arr
        data = {'video': final, 'text': caption, 'meta': meta_arr}
        return data


class TextImageDataset(TextVideoDataset):

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        video_loading = self.video_params.get('loading', 'strict')

        try:
            img = Image.open(video_fp).convert("RGB")
        except:
            if video_loading == 'strict':
                raise ValueError(f'Image loading failed for {video_fp}, image loading for this dataset is strict.')
            else:
                img = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))

        # convert to tensor because video transforms don't, expand such that its a 1-frame video.
        img = transforms.ToTensor()(img).unsqueeze(0)
        if self.transforms is not None:
            img = self.transforms(img)
        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': img, 'text': caption, 'meta': meta_arr}
        return data



