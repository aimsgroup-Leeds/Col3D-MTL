import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloader import *
from network import Col3D_MTL

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='Col3D-MTL PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args
parser.add_argument('--model_name', type=str, help='model name', default='Col3D_MTL')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',default='resnet50_bts')
parser.add_argument('--pretrained_encoder', type=str, help='type of pretrained encoder; resnet50_100, resnet50_300', required=False, default='')
parser.add_argument('--data_path', type=str, help='path to the data', required=False, default='/path/to/data')
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=False, default='/path/to/filenames')
parser.add_argument('--input_height', type=int, help='input height', default=320)
parser.add_argument('--input_width', type=int, help='input width', default=320)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=100.0)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='/path/to/checkpoint')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='c3vd')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--filter_size', type=int,   help='initial num_filters', default=512)

# MTL configuration
parser.add_argument('--multitask', type=str, help='if set, will compute both depth and surface normals', default=False)
parser.add_argument('--concat', type=str, help='if set, will concatenate surface normals from normal estimation to depth estimation', default=False)
parser.add_argument('--full_concat', type=str, help='if set, will concatenate surface normals from all normal estimations to depth estimations', default=False)
parser.add_argument('--CL', type=str, help='if set, will concatenate surface normals from all normal estimations to depth estimations', default=False)

# Intrinsic camera parameters
parser.add_argument('--fx', type=float, help='focal-x value of camera', default=769.807403688120)
parser.add_argument('--fy', type=float, help='focal-y value of camera', default=769.720558534159)
parser.add_argument('--cx', type=float, help='center-x value of camera', default=675.226397736271)
parser.add_argument('--cy', type=float, help='center-y value of camera', default=548.903474592445)

parser.add_argument('--re', type=str, help='if set, will apply random erasing to the training images', default=False)
parser.add_argument('--cj', type=str, help='if set, will apply random color jittering to the training images', default=False)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.multitask == 'True':
    args.multitask = True
else:
    args.multitask = False

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    """Test function."""
    args.mode = 'test'
    dataloader = NewDataLoader(args, 'test')

    model = Col3D_MTL(params=args)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []
    pred_8x8s = []
    pred_4x4s = []
    pred_2x2s = []
    pred_1x1s = []
    pred_normals = []

    start_time = time.time()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].to(device))
            focal = Variable(sample['focal'].to(device))
            # Predict
            if args.multitask:
                if args.CL == 'True':
                    _, _, _, _, depth_est, _, _, _, _, normal_est, _, _ = model(image, focal)
                else:
                    _, _, _, _, depth_est, _, _, _, _, normal_est = model(image, focal)
                pred_normal = normal_est[0].permute(1, 2, 0).cpu().numpy().squeeze()
                pred_normal[:, :, 0] = ((pred_normal[:, :, 0] + 1) * 32768) / 65536
                pred_normal[:, :, 1] = ((pred_normal[:, :, 1] + 1) * 32768) / 65536
                pred_normals.append(pred_normal)
            else:
                _, _, _, _, depth_est = model(image, focal)

            pred_depths.append(depth_est.cpu().numpy().squeeze())
            #pred_8x8s.append(lpg8x8[0].cpu().numpy().squeeze())
            #pred_4x4s.append(lpg4x4[0].cpu().numpy().squeeze())
            #pred_2x2s.append(lpg2x2[0].cpu().numpy().squeeze())
            #pred_1x1s.append(reduc1x1[0].cpu().numpy().squeeze())

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')

    if not os.path.exists(os.path.dirname('predictions/')):
        os.mkdir('predictions/')
    save_name = 'predictions/' + args.model_name

    print('Saving result pngs..')

    if not os.path.exists(save_name):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/pred_depth')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt_depth')
            if args.multitask:
                os.mkdir(save_name + '/pred_normals')
                os.mkdir(save_name + '/gt_normals')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    for s in tqdm(range(num_test_samples)):
        scene_name = lines[s].split()[0].split('/')[2]
        filename_pred_depth = save_name + '/pred_depth/' + scene_name + '_' + lines[s].split()[0].split('/')[3].replace('color.png', 'depth.npy')
        filename_gt_depth = save_name + '/gt_depth/' + scene_name + '_' + lines[s].split()[0].split('/')[3].replace('color.png', 'depth.npy')
        filename_image_png = save_name + '/rgb/' + scene_name + '_' + lines[s].split()[0].split('/')[3]
        if args.multitask:
            filename_gt_normal = save_name + '/gt_normals/' + scene_name + '_' + lines[s].split()[0].split('/')[3].replace('color.png', 'normals.npy')
            filename_pred_normal = save_name + '/pred_normals/' + scene_name + '_' + lines[s].split()[0].split('/')[3].replace('color.png', 'normals.npy')

        rgb_path = args.data_path + lines[s].split()[0]
        image = cv2.imread(rgb_path)
        image = cv2.resize(image, (320,320))
        gt_path = rgb_path.replace('color.png', 'depth.tiff')
        gt_depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        gt_depth = cv2.resize(gt_depth, (320, 320))
        gt_depth = gt_depth / 655.35
        if args.multitask:
            normal_path = rgb_path.replace('color.png','normals.tiff')
            gt_normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            gt_normal = cv2.resize(gt_normal, (320,320))
            b, g, r = cv2.split(gt_normal)
            x = (r / (65535.0 / 2)) - 1
            y = (g / (65535.0 / 2)) - 1
            z = b / 65535.0
            gt_normal = cv2.merge([x, y, z])

        pred_depth = pred_depths[s]
        #pred_8x8 = pred_8x8s[s]
        #pred_4x4 = pred_4x4s[s]
        #pred_2x2 = pred_2x2s[s]
        #pred_1x1 = pred_1x1s[s]
        if args.multitask:
            pred_normal = pred_normals[s]
            #gt_normal = gt_normals[s]

        cv2.imwrite(filename_image_png, image)
        #plt.imsave(filename_pred_png, pred_depth)
        np.save(filename_pred_depth, pred_depth)
        # cv2.imwrite(filename_gt_png, gt_depth)
        np.save(filename_gt_depth, gt_depth)
        if args.multitask:
            #plt.imsave(filename_gt_normal, gt_normal)
            #plt.imsave(filename_pred_normal, pred_normal)
            np.save(filename_gt_normal, gt_normal)
            np.save(filename_pred_normal, pred_normal)

        if args.save_lpg:
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
            plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8), cmap='Greys')
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
            plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4), cmap='Greys')
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
            plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2), cmap='Greys')
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
            plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1), cmap='Greys')

    return


if __name__ == '__main__':
    test(args)
