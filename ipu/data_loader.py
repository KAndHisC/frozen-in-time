import poptorch
from poptorch import DataLoader
import torch

from ipu.MSRVTT_dataset import MSRVTT
from data_loader.transforms import init_transform_dict
from ipu import options

def dataset_loader(dataset_name,
                   text_params,
                   video_params,
                   data_dir,
                   metadata_dir=None,
                   split='train',
                   tsfms=None,
                   cut=None,
                   subsample=1,
                   sliding_window_stride=-1,
                   reader='decord'):
    kwargs = dict(
        dataset_name=dataset_name,
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample,
        sliding_window_stride=sliding_window_stride,
        reader=reader
    )

    # TODO: change to...
    #  dataset = globals()[dataset_name]
    #  ...is this safe / or just lazy?
    if dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset


class TextVideoDataLoader(DataLoader):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 tsfm_split=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='decord',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)

        if tsfm_split is None:
            tsfm_split = split
        tsfm = tsfm_dict[tsfm_split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)
        #        if split != 'train':
        #            shuffle = False
        self.batch_idx = 0
        self.n_samples = len(dataset)
        # mode = poptorch.DataLoaderMode.Async if async_dataloader else poptorch.DataLoaderMode.Sync
        if split == 'val':
           ipu_opts = options.get_inf_opts()
        else:
            ipu_opts = options.get_train_opts()
        super().__init__(ipu_opts,
                        dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=not(isinstance(dataset, torch.utils.data.IterableDataset)),
                        persistent_workers = True,
                        auto_distributed_partitioning = not isinstance(dataset, torch.utils.data.IterableDataset),
                        worker_init_fn=None,
                        # mode=poptorch.DataLoaderMode.Async,
                        async_options={'load_indefinitely': True})
        self.dataset_name = dataset_name
