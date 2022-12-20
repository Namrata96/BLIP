
import json
import numpy as np
import time
import logging
import os
import random
from torch.utils.data import Dataset
from PIL import ImageDraw
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomGrayscale, ColorJitter
from PIL import Image
from PIL import ImageFile
import torch

class sherlock_dataset(Dataset):
    def __init__(self, data, args, training=False):
        self.args = args
        self.data = data
        self.id2data = {d['instance_id']: d for d in self.data}
        self.training = training
        if self.args["widescreen_processing"] in [0, 1]:
            self.preprocess = self._transform_train(args["image_size"]) if self.training else self._transform_test(args["image_size"])
        else:
            self.preprocess = self._transform_train_pad(args["image_size"]) if self.training else self._transform_test_pad(args["image_size"])

    def url2filepath(self, url):
        if 'VG_' in url:
            return self.args["vg_dir"] + '/'.join(url.split('/')[-2:])
        else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
            if 'vcr1images' in self.args["vcr_dir"]:
                return self.args["vcr_dir"] + '/'.join(url.split('/')[-2:])
            else:
                return self.args["vcr_dir"] + '/'.join(url.split('/')[-3:])

    def hide_region(self, image, bboxes):
        image = image.convert('RGBA')
         #highlight mode
        overlay = Image.new('RGBA', image.size, '#00000000')
        draw = ImageDraw.Draw(overlay, 'RGBA')
       
        for bbox in bboxes:
            x = bbox['left']
            y = bbox['top']
           # highlight mode
            draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                            fill='#ff05cd3c', outline='#05ff37ff', width=3)
           
        image = Image.alpha_composite(image, overlay)

        return image

    def _transform_train(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomCrop(n_px),
            RandomHorizontalFlip(),
            RandomGrayscale(),
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_train_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomHorizontalFlip(),
            RandomGrayscale(),
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def image_to_torch_tensor(self, image):
        if self.args["widescreen_processing"] == 1:
            width, height = image.size
            if width >= height:
                im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
                im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
            else:
                im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
                im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
            regions = [image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]
            image = torch.stack([self.preprocess(r) for r in regions], 0)
        else:
            image = self.preprocess(image)
        return image

    def __getitem__(self, idx):
        c_data = self.data[idx]
        
        image = Image.open(self.url2filepath(c_data['inputs']['image']['url']))

        image = self.hide_region(image, c_data['inputs']['bboxes'])

        # clue = clip.tokenize(c_data['inputs']['clue'], truncate=True).squeeze()
        clue = c_data['inputs']['clue']
        
       
        # caption = clip.tokenize('{}'.format(c_data['inputs']['clue']),
        #                             truncate=True).squeeze()
        caption = clue
        cid = c_data['instance_id']
        # print("Image size before: ", image.size)
        image = image.convert('RGB')
        # print("Image size after: ", image.size)
        image = self.image_to_torch_tensor(image)
        if self.training:
            return image, clue, cid
        else:
            return image, cid, clue

    def get(self, cid):
        return self.id2data[cid]

    def __len__(self):
        return len(self.data)