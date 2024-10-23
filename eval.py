import torch
import torch.backends.cudnn as cudnn

import os, sys
import argparse
import numpy as np
from tqdm import tqdm

from network import Col3D_MTL

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='Col3D-MTL PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='Col3D_MTL')
parser.add_argument('--encoder', type=str, help='type of encoder', default='resnet50-bts')
parser.add_argument('--pretrained_encoder', type=str, help='type of encoder', default='None')
parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')

# Dataset
parser.add_argument('--dataset', type=str, help='dataset to train on, kitti or nyu', default='c3vd')
parser.add_argument('--input_height', type=int, help='input height', default=320)
parser.add_argument('--input_width', type=int, help='input width',  default=320)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=100)

# Preprocessing
parser.add_argument('--do_random_rotate', help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=1)
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI', action='store_true')
parser.add_argument('--re', type=str, help='if set, will apply random erasing to the training images', default=False)
parser.add_argument('--cj', type=str, help='if set, will apply random color jittering to the training images', default=False)

# Eval
parser.add_argument('--data_path_eval', type=str, help='path to the data for evaluation', required=False)
parser.add_argument('--gt_path_eval', type=str, help='path to the groundtruth data for evaluation', required=False)
parser.add_argument('--filenames_file_eval', type=str, help='path to the filenames text file for evaluation', required=False)
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=100)
parser.add_argument('--filter_size', type=int, help='initial num_filters', default=512)

# MTL configuration
parser.add_argument('--multitask', type=str, help='if set, will compute both depth and surface normals', default='False')
parser.add_argument('--concat', type=str, help='if set, will concatenate surface normals from normal estimation to depth estimation', default='False')
parser.add_argument('--full_concat', type=str, help='if set, will concatenate surface normals from all normal estimations to all depth estimations', default='False')
parser.add_argument('--CL', type=str, help='if set, will compute cross-consistency loss', default='False')

# Intrinsic camera parameters
parser.add_argument('--fx', type=float, help='focal-x value of camera', default=769.807403688120)
parser.add_argument('--fy', type=float, help='focal-y value of camera', default=769.720558534159)
parser.add_argument('--cx', type=float, help='center-x value of camera', default=675.226397736271)
parser.add_argument('--cy', type=float, help='center-y value of camera', default=548.903474592445)


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.multitask == 'True':
    args.multitask = True
else:
    args.multitask = False

from dataloader import NewDataLoader

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def compute_sn_errors(gt, pred, mask):
    mag_gt = np.sqrt(gt[0]**2 + gt[1]**2 + gt[2]**2)
    mag_pred = np.sqrt(pred[0]**2 + pred[1]**2 + pred[2]**2)
    mag = np.expand_dims(mag_gt * mag_pred, axis=0)
    mult = gt * pred
    dot = np.sum(mult, axis=0, keepdims=True)
    arg = dot / (mag + 1e-3)
    arg [arg > 1] = 1
    rads = np.arccos(arg)
    angular_error = np.degrees(rads).squeeze()

    mean_angular_error = np.mean(angular_error[mask])
    median_angular_error = np.median(angular_error[mask])
    d1 = np.sum(angular_error[mask] <= 11.25) / np.sum(angular_error[mask] <= 360) * 100
    d2 = np.sum(angular_error[mask] <= 22.5) / np.sum(angular_error[mask] <= 360) * 100
    d3 = np.sum(angular_error[mask] <= 30) / np.sum(angular_error[mask] <= 360) * 100

    return [mean_angular_error, median_angular_error, d1, d2, d3]

def eval(model, dataloader_eval):
    eval_measures = torch.zeros(10).cuda()
    eval_measures_sn = torch.zeros(6).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    timings = np.zeros((1268,1)) #1268 204

    silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3 = [], [], [], [], [], [], [], [], []

    for i, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            gt_normal = eval_sample_batched['normal']
            focal = eval_sample_batched['focal']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                print('Invalid depth. continue.')
                continue

            if args.multitask:
                if args.CL == 'True':
                   starter.record()
                   _, _, _, _, pred_depth, _, _, _, _, pred_sn, _, _ = model(image, focal)
                   ender.record()
                else:
                   starter.record()
                   _, _, _, _, pred_depth, _, _, _, _, pred_sn = model(image, focal)
                   ender.record()
                pred_sn = pred_sn.cpu().numpy().squeeze()
                gt_normal = gt_normal.cpu().numpy().squeeze()

            else:
                starter.record()
                _, _, _, _, pred_depth = model(image, focal)
                ender.record()

        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time

        pred_depth = pred_depth.cpu().numpy().squeeze()
        gt_depth = gt_depth.cpu().numpy().squeeze()

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        if args.multitask:
            sn_measures = compute_sn_errors(gt_normal, pred_sn, valid_mask)
            eval_measures_sn[:5] += torch.tensor(sn_measures).cuda()
            eval_measures_sn[5] += 1

        depth_measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
        eval_measures[:9] += torch.tensor(depth_measures).cuda()
        eval_measures[9] += 1

        silog.append(depth_measures[0])
        abs_rel.append(depth_measures[1])
        log10.append(depth_measures[2])
        rms.append(depth_measures[3])
        sq_rel.append(depth_measures[4])
        log_rms.append(depth_measures[5])
        d1.append(depth_measures[6])
        d2.append(depth_measures[7])
        d3.append(depth_measures[8])

    mean_syn = np.sum(timings) / 1268 #1268 204
    std_syn = np.std(timings)
    print('Mean inference time: ', mean_syn, flush=True)
    print('StD inference time:	', std_syn, flush=True)

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing depth errors for {} eval samples'.format(int(cnt)), flush=True)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                    'd3'), flush=True)
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='', flush=True)
    print('{:7.4f}'.format(eval_measures_cpu[8]), flush=True)
    print(np.array(silog).mean(), np.array(silog).std(), np.array(abs_rel).mean(), np.array(abs_rel).std(),np.array(log10).mean(), np.array(log10).std(), np.array(rms).mean(), np.array(rms).std(), np.array(sq_rel).mean(), np.array(sq_rel).std(), np.array(log_rms).mean(), np.array(log_rms).std(), np.array(d1).mean(), np.array(d1).std(), np.array(d2).mean(), np.array(d2).std(), np.array(d3).mean(), np.array(d3).std())

    if args.multitask:
        eval_measures_sn_cpu = eval_measures_sn.cpu()
        cnt = eval_measures_sn_cpu[5].item()
        eval_measures_sn_cpu /= cnt
        print('Computing sn errors for {} eval samples'.format(int(cnt)), flush=True)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('mean', 'median', 'd1', 'd2', 'd3'), flush=True)
        for i in range(4):
            print('{:7.4f}, '.format(eval_measures_sn_cpu[i]), end='', flush=True)
        print('{:7.4f}'.format(eval_measures_sn_cpu[4]), flush=True)

    return eval_measures_cpu


def main_worker(args):
    model = Col3D_MTL(args)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    model = torch.nn.DataParallel(model)
    model.cuda()

    print("== Model Initialized")

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path, map_location='cuda')
            model.load_state_dict(checkpoint['model'])
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))

    cudnn.benchmark = True

    dataloader_eval = NewDataLoader(args, 'online_eval')

    # ===== Evaluation ======
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).cuda()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings=np.zeros((repetitions,1))
        # GPU Warming-up
        for _ in range(10):
          _ = model(dummy_input,1)
        # Evaluation
        eval_measures = eval(model, dataloader_eval)


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    main_worker(args)


if __name__ == '__main__':
    main()
