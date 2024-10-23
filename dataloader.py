import numpy as np
import cv2
from scipy import ndimage
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random

#from distributed_sampler_no_evenly_divisible import *


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode, args):
    return transforms.Compose([
        ToTensor(mode=mode, args=args)
    ])


class NewDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode, args))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode, args))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode, args))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[3])
        #print(sample_path.split()[0], sample_path.split()[1], sample_path.split()[2], sample_path.split()[3], flush=True)
        if self.mode == 'train':
            image_path = os.path.join(self.args.data_path, sample_path.split()[0][1:])
            depth_path = os.path.join(self.args.gt_path, sample_path.split()[1][1:])
            normal_path = os.path.join(self.args.gt_path, sample_path.split()[2][1:])

            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            normal_gt = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            normal_b, normal_g, normal_r = cv2.split(normal_gt)
            normal_b, normal_g, normal_r = Image.fromarray(normal_b), Image.fromarray(normal_g), Image.fromarray(normal_r)

            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height / 2 - (960 / 2))
                left_margin = int(width / 2 - (960 / 2))
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 960, top_margin + 960))
                image = image.crop((left_margin, top_margin, left_margin + 960, top_margin + 960))
                normal_b = normal_b.crop((left_margin, top_margin, left_margin + 960, top_margin + 960))
                normal_g = normal_g.crop((left_margin, top_margin, left_margin + 960, top_margin + 960))
                normal_r = normal_r.crop((left_margin, top_margin, left_margin + 960, top_margin + 960))

            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                normal_b = self.rotate_image(normal_b, random_angle, flag=Image.NEAREST)
                normal_g = self.rotate_image(normal_g, random_angle, flag=Image.NEAREST)
                normal_r = self.rotate_image(normal_r, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            normal_b, normal_g, normal_r = np.array(normal_b), np.array(normal_g), np.array(normal_r)

            depth_gt = (depth_gt / 65535.0) * self.args.max_depth
            normal_b = normal_b / 65535.0  # z-axis
            normal_g = (normal_g / (65535.0 / 2)) - 1  # y-axis
            normal_r = (normal_r / (65535.0 / 2)) - 1  # x-axis

            normal_gt = cv2.merge([normal_r, normal_g, normal_b])

            image, depth_gt, normal_gt = self.train_preprocess(image, depth_gt, normal_gt)
            sample = {'image': image, 'depth': depth_gt, 'normal': normal_gt, 'focal': focal}

        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path
            image_path = os.path.join(data_path + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, sample_path.split()[1][1:])
                normal_path = os.path.join(gt_path, sample_path.split()[2][1:])
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    normal_gt = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
                    b, g, r = cv2.split(normal_gt)
                    x = (r / (65535.0 / 2)) - 1
                    y = (g / (65535.0 / 2)) - 1
                    z = b / 65535.0
                    normal_gt = cv2.merge([x, y, z])
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    depth_gt = (depth_gt / 65535.0) * self.args.max_depth

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height / 2 - (960 / 2))
                left_margin = int(width / 2 - (960 / 2))
                image = image[top_margin:top_margin + 960, left_margin:left_margin + 960, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 960, left_margin:left_margin + 960, :]
                    normal_gt = normal_gt[top_margin:top_margin + 960, left_margin:left_margin + 960, :]

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'normal': normal_gt, 'focal': focal,
                          'has_valid_depth': has_valid_depth, 'image_path': image_path, 'depth_path': depth_path}
            else:
                sample = {'image': image, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, normal, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        normal = normal[y:y + height, x:x + width, :]
        return img, depth, normal

    def train_preprocess(self, image, depth_gt, normal_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            normal_gt = (normal_gt[:, ::-1, :]).copy()
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt, normal_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode, args):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.re = args.re
        if self.re == 'True':
            print('Using random erasing')
            self.random_erase = transforms.RandomErasing(p=0.5, scale=(0.01, 0.01), ratio=(1, 1))
        self.cj = args.cj
        if self.cj == 'True':
            print('Using random color jittering')
            self.color_jitter = transforms.ColorJitter()

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)
        resize = transforms.Resize((320, 320))
        image = resize(image)
        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth, normal = sample['depth'], sample['normal']
        if self.mode == 'train':
            if self.cj == 'True': # Random color jittering
                image = self.color_jitter(image)
            if self.re == 'True': # Random erasing
                image = self.random_erase(image)
            depth = self.to_tensor(depth)
            normal = self.to_tensor(normal)
            depth = resize(depth)
            normal = resize(normal)
            return {'image': image, 'depth': depth, 'normal': normal, 'focal': focal}
        else:
            depth = self.to_tensor(depth)
            normal = self.to_tensor(normal)
            depth = resize(depth)
            normal = resize(normal)
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'normal': normal, 'focal': focal,
                    'has_valid_depth': has_valid_depth, 'image_path': sample['image_path'],
                    'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
