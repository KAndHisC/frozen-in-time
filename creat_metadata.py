import pandas as pd
import os
import numpy as np
import argparse
import requests
import concurrent.futures
from tqdm import tqdm


def check_file_exist(file_path):
    try:
        if os.path.isfile(file_path) and os.path.getsize(file_path)>100000:
            # imgs, idxs = video_reader(video_fp)
            return True
        else:
            return False
    except Exception as e:
        # raise ValueError(f'Video loading failed for {save_fp}, video loading for this dataset is strict.') from e
        return False

def main(args):
    ### preproc
    video_dir = os.path.join(args.data_dir, 'videos')
    # print(video_dir)
    if not os.path.exists(os.path.join(video_dir, 'videos')):
        os.makedirs(os.path.join(video_dir, 'videos'))


    df = pd.read_csv(os.path.join(args.data_dir,'release', args.csv_file))

    df = df.dropna(axis=0,how='any')
    df['rel_fn'] = df.apply(lambda x: os.path.join(video_dir, x['page_dir'], str(x['videoid'])), axis=1)

    df['rel_fn'] = df['rel_fn'] + '.mp4'
    
    df['ok'] = df.apply(lambda x: check_file_exist(x['rel_fn']), axis=1) 
    df_clear = df.drop(df[df['ok']==False].index)
    df_clear = df_clear.drop(columns=['ok','rel_fn','contentUrl'])
    metadata_file = os.path.join(args.data_dir, 'metadata1',args.csv_file)
    df_clear.to_csv(metadata_file, index=False)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shutter Image/Video Downloader')
    parser.add_argument('--data_dir', type=str, default='data/WebVid',
                        help='Directory where webvid data is stored.')
    parser.add_argument('--csv_file', type=str, default='results_2M_train.csv',
                        help='Path to csv data to download')
    parser.add_argument('--processes', type=int, default=16)
    args = parser.parse_args()
    main(args)
    # nohup python download.py --part 0 > logs/train_download.log 2>&1 &
