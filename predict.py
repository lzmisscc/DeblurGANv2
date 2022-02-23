import os
from glob import glob
from typing import Optional

import cv2
from torchvision.transforms import transforms
import numpy as np
import torch
from torch import nn
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator


class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml') as cfg:
            config = yaml.load(cfg)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    # @staticmethod
    # def _array_to_batch(x):
    #     # x = np.transpose(x, (2, 0, 1))
    #     x = torch.stack(x, 0)
    #     return x

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        # x, _ = self.normalize_fn(x, x)
        # if mask is None:
        #     mask = np.ones_like(x, dtype=np.float32)
        # else:
        #     mask = np.round(mask.astype('float32') / 255)

        _, h, w = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        # pad_params = {'mode': 'constant',
        #               'constant_values': 0,
        #               'pad_width': ((0, 0), (0, min_height - h), (0, min_width - w))
        #               }
        
        # x = np.pad(x, **pad_params)
        # mask = np.pad(mask, **pad_params)
        x = transforms.Pad((0, 0, min_height - h, min_width - w))(x)
        x = torch.unsqueeze(x, 0)
        mask = torch.ones_like(x)
        return (x, mask), h, w

    # @staticmethod
    # def _postprocess(x: torch.Tensor) -> np.ndarray:
    #     x, = x
    #     x = x.detach().cpu().float().numpy()
    #     x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
    #     return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return pred[:, :, :h, :w]

def process_video(pairs, predictor, output_dir):
    for video_filepath, mask in tqdm(pairs):
        video_filename = os.path.basename(video_filepath)
        output_filepath = os.path.join(output_dir, os.path.splitext(video_filename)[0]+'_deblur.mp4')
        video_in = cv2.VideoCapture(video_filepath)
        fps = video_in.get(cv2.CAP_PROP_FPS)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        video_out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        tqdm.write(f'process {video_filepath} to {output_filepath}, {fps}fps, resolution: {width}x{height}')
        for frame_num in tqdm(range(total_frame_num), desc=video_filename):
            res, img = video_in.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predictor(img, mask)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            video_out.write(pred)

from dataset import NoPairDataset
from aug import get_transforms

def main(dataset_path="",
        weights_path='weights/best_fpn.h5',
         out_dir='submit/',
         chunk:bool=False,
         side_by_side: bool = False,
         video: bool = False):
    # def sorted_glob(pattern):
    #     return sorted(glob(pattern))

    # imgs = sorted_glob(img_pattern)
    # masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    # pairs = zip(imgs, masks)
    # names = sorted([os.path.basename(x) for x in glob(img_pattern)])
    pairs = NoPairDataset(dataset_path, chunk=chunk)
    names = pairs.img_list
    
    predictor = Predictor(weights_path=weights_path)

    os.makedirs(out_dir, exist_ok=True)
    mse_list = []
    mse_loss_fn = nn.MSELoss(size_average=True)
    if not video:
        for name, pair in tqdm(zip(names, pairs), total=len(names)):
            img, mask = pair
            # img, mask = map(cv2.imread, (f_img, f_mask))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred = predictor(mask, None).cpu()
            pred = torch.clip(pred, 0.0, 1.0)
            # if chunk:
            #     mse_loss = mse_loss_fn(mask, pred[0])
            #     mse_list.append((name, mse_loss.item()))
            pred = torch.cat((img, mask, pred[0]), -1)
            name = os.path.basename(name)
            transforms.ToPILImage()(pred).save(os.path.join(out_dir, name))
    else:
        process_video(pairs, predictor, out_dir)

    for name, mse_loss in mse_list:
        print(f"{str(name):20}, mse {mse_loss:10.5f}")
if __name__ == '__main__':
    Fire(main)
