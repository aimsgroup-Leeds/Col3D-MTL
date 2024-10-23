import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math

from collections import namedtuple

torch.manual_seed(1)

def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True
        m.eval()
        m.affine = True
        m.requires_grad = True


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask] + 1e-3) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class rmse_loss(nn.Module):
    def __init__(self):
        super(rmse_loss, self).__init__()

    def forward(self, normal_est, normal_gt, mask):
        rms = (normal_gt[mask] - normal_est[mask]) ** 2
        return torch.sqrt(rms.mean())

class l1norm_loss(nn.Module):
    def __init__(self):
        super(l1norm_loss, self).__init__()

    def forward(self, prediction, warping, mask):
        l1_norm = torch.abs(prediction[mask] - warping[mask])
        return torch.mean(l1_norm)


class aae_loss(nn.Module):
    def __init__(self):
        super(aae_loss, self).__init__()

    def forward(self,gt, pred, mask):
        mag_gt = torch.sqrt(gt[:,0,:,:]**2 + gt[:,1,:,:]**2 + gt[:,2,:,:]**2 + 1e-3)
        mag_pred = torch.sqrt(pred[:,0,:,:]**2 + pred[:,1,:,:]**2 + pred[:,2,:,:]**2 + 1e-3)
        mag = torch.unsqueeze(mag_gt * mag_pred, dim=1)
        mult = gt * pred
        dot = torch.sum(mult, dim=1, keepdim=True)
        arg = dot / (mag + 1e-3)
        arg[arg > 1] = 1
        arg[arg < -1] = -1
        rads = torch.acos(arg)
        angular_error = torch.rad2deg(rads).squeeze()
        mean_angular_error = torch.mean(angular_error[mask[:,0,:,:]])
        return mean_angular_error



class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True,
                                                                   track_running_stats=True, eps=1.1e-5))

        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels,
                                                                              out_channels=out_channels * 2, bias=False,
                                                                              kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels * 2, momentum=0.01,
                                                                                   affine=True,
                                                                                   track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2,
                                                                              out_channels=out_channels, bias=False,
                                                                              kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation),
                                                                              dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)


class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                              padding=1)
        self.ratio = ratio

    def forward(self, x):
        up_x = torch_nn_func.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out


class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth=100, is_final=False, normal=False):
        super(reduction_1x1, self).__init__()
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()
        self.normal = normal
        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    if self.normal:
                        self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=3, bias=False, kernel_size=1, stride=1, padding=0), nn.Sigmoid()))
                    else:
                        self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=3, bias=False, kernel_size=1, stride=1, padding=0), nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(
                                          nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)

        return net


class local_planar_guidance(nn.Module):
    def __init__(self, upratio):
        super(local_planar_guidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.n1_conv = torch.nn.Sequential(nn.Conv2d(2, 1, 3, 1, 1, bias=False),nn.ELU(), nn.Conv2d(1,1,3,1,1), nn.Tanh())
        self.n2_conv = torch.nn.Sequential(nn.Conv2d(2, 1, 3, 1, 1, bias=False),nn.ELU(), nn.Conv2d(1,1,3,1,1), nn.Tanh())
        self.n3_conv = torch.nn.Sequential(nn.Conv2d(2, 1, 3, 1, 1, bias=False),nn.ELU(), nn.Conv2d(1,1,3,1,1), nn.Sigmoid())
        self.upratio = float(upratio)

    def forward(self, plane_eq, focal, surface_normal=None):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        if surface_normal != None:
            n1_concat = torch.cat([surface_normal[:, 0, :, :].unsqueeze(1), n1.unsqueeze(1)], dim=1)
            n1 = self.n1_conv(n1_concat).squeeze(1)
            n2_concat = torch.cat([surface_normal[:, 1, :, :].unsqueeze(1), n2.unsqueeze(1)], dim=1)
            n2 = self.n2_conv(n2_concat).squeeze(1)
            n3_concat = torch.cat([surface_normal[:, 2, :, :].unsqueeze(1), n3.unsqueeze(1)], dim=1)
            n3 = self.n3_conv(n3_concat).squeeze(1)
        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).to(device)
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).to(device)
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3 + 1e-9), plane_eq_expanded[:,:3,:,:]


class daspp(nn.Module):
    def __init__(self, params, feat_out_channels, num_features=512):
        super(daspp, self).__init__()
        self.params = params

        self.upconv5 = upconv(feat_out_channels[4], num_features)
        self.bn5 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.conv5 = torch.nn.Sequential(
            nn.Conv2d(num_features + feat_out_channels[3], num_features, 3, 1, 1, bias=False),
            nn.ELU())
        self.upconv4 = upconv(num_features, num_features // 2)
        self.bn4 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(num_features // 2 + feat_out_channels[2], num_features // 2, 3, 1, 1, bias=False),
            nn.ELU())
        self.bn4_2 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

        self.daspp_3 = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6 = atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[2], num_features // 4, 6)
        self.daspp_12 = atrous_conv(num_features + feat_out_channels[2], num_features // 4, 12)
        self.daspp_18 = atrous_conv(num_features + num_features // 4 + feat_out_channels[2], num_features // 4, 18)
        self.daspp_24 = atrous_conv(num_features + num_features // 2 + feat_out_channels[2], num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(
            nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())

    def forward(self, features):
        skip0, skip1, skip2, skip3 = features[0], features[1], features[2], features[3]
        dense_features = torch.nn.ReLU()(features[4])
        upconv5 = self.upconv5(dense_features)  # H/16
        upconv5 = self.bn5(upconv5)
        concat5 = torch.cat([upconv5, skip3], dim=1)
        iconv5 = self.conv5(concat5)

        upconv4 = self.upconv4(iconv5)  # H/8
        upconv4 = self.bn4(upconv4)
        concat4 = torch.cat([upconv4, skip2], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)

        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)
        return daspp_feat


class depth_decoder(nn.Module):
    def __init__(self, params, feat_out_channels, num_features=512):
        super(depth_decoder, self).__init__()
        self.params = params
        # Depth decoder
        self.ca1 = ChannelAttention(num_features // 4)
        self.sa1 = SpatialAttention()

        self.reduc8x8 = reduction_1x1(num_features // 4, num_features // 4, self.params.max_depth)
        self.lpg8x8 = local_planar_guidance(8)

        self.ca2 = ChannelAttention(feat_out_channels[1])
        self.sa2 = SpatialAttention()

        self.upconv3 = upconv(num_features // 4, num_features // 4)
        self.bn3 = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3 = torch.nn.Sequential(nn.Conv2d(num_features // 4 + feat_out_channels[1] + 1, num_features // 4, 3, 1, 1, bias=False), nn.ELU())

        self.ca3 = ChannelAttention(num_features // 4)
        self.sa3 = SpatialAttention()

        self.reduc4x4 = reduction_1x1(num_features // 4, num_features // 8, self.params.max_depth)
        self.lpg4x4 = local_planar_guidance(4)

        self.ca4 = ChannelAttention(feat_out_channels[0])
        self.sa4 = SpatialAttention()

        self.upconv2 = upconv(num_features // 4, num_features // 8)
        self.bn2 = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2 = torch.nn.Sequential(nn.Conv2d(num_features // 8 + feat_out_channels[0] + 1, num_features // 8, 3, 1, 1, bias=False), nn.ELU())

        self.ca5 = ChannelAttention(num_features // 8)
        self.sa5 = SpatialAttention()

        self.reduc2x2 = reduction_1x1(num_features // 8, num_features // 16, self.params.max_depth)
        self.lpg2x2 = local_planar_guidance(2)

        self.upconv1 = upconv(num_features // 8, num_features // 16)
        self.reduc1x1 = reduction_1x1(num_features // 16, num_features // 32, self.params.max_depth, is_final=True)
        self.conv1 = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False),nn.ELU())
        self.conv1_concat = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 7, num_features // 16, 3, 1, 1, bias=False),nn.ELU())
        self.get_depth = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False), nn.Sigmoid())

        if self.params.CL == 'True':
            self.conv1_x_d2n = torch.nn.Sequential(nn.Conv2d(1, num_features//16, 3, 1, 1, bias=False), nn.ELU())
            self.conv1_y_d2n = torch.nn.Sequential(nn.Conv2d(1, num_features//16, 3, 1, 1, bias=False), nn.ELU())
            self.conv1_z_d2n = torch.nn.Sequential(nn.Conv2d(1, num_features//16, 3, 1, 1, bias=False), nn.ELU())
            self.get_normal_x_d2n = torch.nn.Sequential(nn.Conv2d(num_features//16, 1, 3, 1, 1, bias=False), nn.Tanh())
            self.get_normal_y_d2n = torch.nn.Sequential(nn.Conv2d(num_features//16, 1, 3, 1, 1, bias=False), nn.Tanh())
            self.get_normal_z_d2n = torch.nn.Sequential(nn.Conv2d(num_features//16, 1, 3, 1, 1, bias=False), nn.Sigmoid())


    def forward(self, features, daspp_feat, focal, normal=None, lpg8_normal=None, lpg4_normal=None, lpg2_normal=None):
        skip0, skip1, skip2, skip3 = features[0], features[1], features[2], features[3]
        # Depth decoder
        ca1 = self.ca1(daspp_feat) * daspp_feat
        sa1 = self.sa1(ca1) * ca1

        reduc8x8 = self.reduc8x8(sa1) # daspp_feat
        plane_normal_8x8 = reduc8x8[:, :3, :, :]
        plane_normal_8x8 = torch_nn_func.normalize(plane_normal_8x8, 2, 1)
        plane_dist_8x8 = reduc8x8[:, 3, :, :]
        plane_eq_8x8 = torch.cat([plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)
        depth_8x8, normal_8x8_d = self.lpg8x8(plane_eq_8x8, focal, lpg8_normal)
        depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.params.max_depth
        depth_8x8_scaled_ds = torch_nn_func.interpolate(depth_8x8_scaled, scale_factor=0.25, mode='nearest')

        upconv3 = self.upconv3(sa1)  # H/4 # daspp_feat
        upconv3 = self.bn3(upconv3)

        ca2 = self.ca2(skip1) * skip1
        sa2 = self.sa2(ca2) * ca2

        concat3 = torch.cat([upconv3, sa2, depth_8x8_scaled_ds], dim=1) # skip1
        iconv3 = self.conv3(concat3)

        ca3 = self.ca3(iconv3) * iconv3
        sa3 = self.sa3(ca3) * ca3

        reduc4x4 = self.reduc4x4(sa3) # iconv3
        plane_normal_4x4 = reduc4x4[:, :3, :, :]
        plane_normal_4x4 = torch_nn_func.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]
        plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        depth_4x4, normal_4x4_d = self.lpg4x4(plane_eq_4x4, focal, lpg4_normal)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.params.max_depth
        depth_4x4_scaled_ds = torch_nn_func.interpolate(depth_4x4_scaled, scale_factor=0.5, mode='nearest')

        upconv2 = self.upconv2(sa3)  # H/2 # iconv3
        upconv2 = self.bn2(upconv2)

        ca4 = self.ca4(skip0) * skip0
        sa4 = self.sa4(ca4) * ca4

        concat2 = torch.cat([upconv2, sa4, depth_4x4_scaled_ds], dim=1) # skip0
        iconv2 = self.conv2(concat2)

        ca5 = self.ca5(iconv2) * iconv2
        sa5 = self.sa5(ca5) * ca5

        reduc2x2 = self.reduc2x2(sa5) # iconv2
        plane_normal_2x2 = reduc2x2[:, :3, :, :]
        plane_normal_2x2 = torch_nn_func.normalize(plane_normal_2x2, 2, 1)
        plane_dist_2x2 = reduc2x2[:, 3, :, :]
        plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        depth_2x2, normal_2x2_d = self.lpg2x2(plane_eq_2x2, focal, lpg2_normal)
        depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.params.max_depth

        upconv1 = self.upconv1(sa5) # iconv2
        reduc1x1 = self.reduc1x1(upconv1)
        theta = reduc1x1[:, 0, :, :] * math.pi / 3
        phi = reduc1x1[:, 1, :, :] * math.pi * 2
        n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
        n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
        n3 = torch.cos(theta).unsqueeze(1)
        n4 = reduc1x1[:, 2, :, :].unsqueeze(1)
        depth_1x1 = n4 / (n1 + n2 + n3 + 1e-9)
        if self.params.concat == 'True':
            concat1 = torch.cat([upconv1, normal, depth_1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], dim=1)
            iconv1 = self.conv1_concat(concat1)
        else:
            concat1 = torch.cat([upconv1, depth_1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], dim=1)
            iconv1 = self.conv1(concat1)

        final_depth = self.params.max_depth * self.get_depth(iconv1)

        if self.params.CL == 'True':
            iconv1_x_d2n = self.conv1_x_d2n(n1)
            iconv1_y_d2n = self.conv1_y_d2n(n2)
            iconv1_z_d2n = self.conv1_z_d2n(n3)
            final_normal_x_d2n = self.get_normal_x_d2n(iconv1_x_d2n)
            final_normal_y_d2n = self.get_normal_y_d2n(iconv1_y_d2n)
            final_normal_z_d2n = self.get_normal_z_d2n(iconv1_z_d2n)
            normal_from_depth = torch.cat([final_normal_x_d2n, final_normal_y_d2n, final_normal_z_d2n],dim=1)

            return depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth, normal_from_depth
        else:
            return depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth



class normal_decoder(nn.Module):
    def __init__(self, params, feat_out_channels, num_features=512):
        super(normal_decoder, self).__init__()
        self.params = params
        self.reduc8x8_normal = reduction_1x1(num_features // 4, num_features // 4)
        self.lpg8x8_normal = local_planar_guidance(8)

        self.upconv3_normal = upconv(num_features // 4, num_features // 4)
        self.bn3_normal = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3_normal = torch.nn.Sequential(nn.Conv2d(num_features // 4 + feat_out_channels[1] + 3, num_features // 4, 3, 1, 1, bias=False), nn.ELU())
        self.reduc4x4_normal = reduction_1x1(num_features // 4, num_features // 8)
        self.lpg4x4_normal = local_planar_guidance(4)

        self.upconv2_normal = upconv(num_features // 4, num_features // 8)
        self.bn2_normal = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2_normal = torch.nn.Sequential(nn.Conv2d(num_features // 8 + feat_out_channels[0] + 3, num_features // 8, 3, 1, 1, bias=False), nn.ELU())

        self.reduc2x2_normal = reduction_1x1(num_features // 8, num_features // 16)
        self.lpg2x2_normal = local_planar_guidance(2)

        self.upconv1_normal = upconv(num_features // 8, num_features // 16)
        self.reduc1x1_normal = reduction_1x1(num_features // 16, num_features // 32, is_final=True, normal=True)
        self.conv1_x = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False), nn.ELU())
        self.conv1_y = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False), nn.ELU())
        self.conv1_z = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False), nn.ELU())
        self.get_normal_x = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False), nn.Tanh())
        self.get_normal_y = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False), nn.Tanh())
        self.get_normal_z = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False), nn.Sigmoid())

        if self.params.CL == 'True':
            self.conv1_n2d = torch.nn.Sequential(nn.Conv2d(1, num_features // 16, 3, 1, 1, bias=False), nn.ELU())
            self.get_n2d = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, features, daspp_feat, focal):
        skip0, skip1, skip2, skip3 = features[0], features[1], features[2], features[3]
        # Normal decoder
        reduc8x8_n = self.reduc8x8_normal(daspp_feat)
        plane_normal_8x8_n = reduc8x8_n[:, :3, :, :]
        plane_normal_8x8_n = torch_nn_func.normalize(plane_normal_8x8_n, 2, 1)
        plane_dist_8x8_n = reduc8x8_n[:, 3, :, :]
        plane_eq_8x8_n = torch.cat([plane_normal_8x8_n, plane_dist_8x8_n.unsqueeze(1)], 1)
        depth_8x8_n, normal_8x8 = self.lpg8x8_normal(plane_eq_8x8_n, focal)
        normal_8x8_scaled_ds = torch_nn_func.interpolate(normal_8x8, scale_factor=0.25, mode='nearest')

        upconv3_n = self.upconv3_normal(daspp_feat)  # H/4
        upconv3_n = self.bn3_normal(upconv3_n)
        concat3_n = torch.cat([upconv3_n, skip1, normal_8x8_scaled_ds], dim=1)
        iconv3_n = self.conv3_normal(concat3_n)

        reduc4x4_n = self.reduc4x4_normal(iconv3_n)
        plane_normal_4x4_n = reduc4x4_n[:, :3, :, :]
        plane_normal_4x4_n = torch_nn_func.normalize(plane_normal_4x4_n, 2, 1)
        plane_dist_4x4_n = reduc4x4_n[:, 3, :, :]
        plane_eq_4x4_n = torch.cat([plane_normal_4x4_n, plane_dist_4x4_n.unsqueeze(1)], 1)
        depth_4x4_n, normal_4x4 = self.lpg4x4_normal(plane_eq_4x4_n, focal)
        normal_4x4_scaled_ds = torch_nn_func.interpolate(normal_4x4, scale_factor=0.5, mode='nearest')

        upconv2_n = self.upconv2_normal(iconv3_n)  # H/2
        upconv2_n = self.bn2_normal(upconv2_n)
        concat2_n = torch.cat([upconv2_n, skip0, normal_4x4_scaled_ds], dim=1)
        iconv2_n = self.conv2_normal(concat2_n)

        reduc2x2_n = self.reduc2x2_normal(iconv2_n)
        plane_normal_2x2_n = reduc2x2_n[:, :3, :, :]
        plane_normal_2x2_n = torch_nn_func.normalize(plane_normal_2x2_n, 2, 1)
        plane_dist_2x2_n = reduc2x2_n[:, 3, :, :]
        plane_eq_2x2_n = torch.cat([plane_normal_2x2_n, plane_dist_2x2_n.unsqueeze(1)], 1)
        depth_2x2_n, normal_2x2 = self.lpg2x2_normal(plane_eq_2x2_n, focal)

        upconv1_n = self.upconv1_normal(iconv2_n)
        reduc1x1_n = self.reduc1x1_normal(upconv1_n)
        theta = reduc1x1_n[:, 0, :, :] * math.pi / 3
        phi = reduc1x1_n[:, 1, :, :] * math.pi * 2
        n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
        n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
        n3 = torch.cos(theta).unsqueeze(1)
        n4 = reduc1x1_n[:, 2, :, :].unsqueeze(1)
        normal_vec_1x1 = torch.cat([n1, n2, n3], dim=1)

        concat1_x = torch.cat([upconv1_n, normal_vec_1x1[:, 0, :, :].unsqueeze(1), normal_2x2[:, 0, :, :].unsqueeze(1), normal_4x4[:, 0, :, :].unsqueeze(1), normal_8x8[:, 0, :, :].unsqueeze(1)], dim=1)
        concat1_y = torch.cat([upconv1_n, normal_vec_1x1[:, 1, :, :].unsqueeze(1), normal_2x2[:, 1, :, :].unsqueeze(1), normal_4x4[:, 1, :, :].unsqueeze(1), normal_8x8[:, 1, :, :].unsqueeze(1)], dim=1)
        concat1_z = torch.cat([upconv1_n, normal_vec_1x1[:, 2, :, :].unsqueeze(1), normal_2x2[:, 2, :, :].unsqueeze(1), normal_4x4[:, 2, :, :].unsqueeze(1), normal_8x8[:, 2, :, :].unsqueeze(1)], dim=1)

        iconv1_x = self.conv1_x(concat1_x)
        iconv1_y = self.conv1_y(concat1_y)
        iconv1_z = self.conv1_z(concat1_z)
        final_normal_x = self.get_normal_x(iconv1_x)
        final_normal_y = self.get_normal_y(iconv1_y)
        final_normal_z = self.get_normal_z(iconv1_z)
        normal_est = torch.cat([final_normal_x, final_normal_y, final_normal_z], dim=1)

        if self.params.CL == 'True':
            n2d = self.conv1_n2d(n4 / (final_normal_x + final_normal_y + final_normal_z + 1e-9))
            depth_from_normal = self.get_n2d(n2d) * self.params.max_depth

            return normal_8x8, normal_4x4, normal_2x2, reduc1x1_n, normal_est, depth_from_normal
        else:
            return normal_8x8, normal_4x4, normal_2x2, reduc1x1_n, normal_est


class encoder(nn.Module):
    def __init__(self, params):
        super(encoder, self).__init__()
        self.params = params
        import torchvision.models as models
        if params.encoder == 'densenet121_bts':
            self.base_model = models.densenet121(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif params.encoder == 'densenet161_bts':
            self.base_model = models.densenet161(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif params.encoder == 'resnet50_bts':
            self.base_model = models.resnet50(pretrained=False)
            if params.pretrained_encoder == 'resnet50_100':
                self.base_model.avgpool = torch.nn.Identity()
                self.base_model.fc = torch.nn.Identity()
                self.base_model.load_state_dict(torch.load('./pretrained_backbones/pretrained_resnet_100.pth')['state_dict'])
            elif params.pretrained_encoder == 'resnet50_300':
                self.base_model.avgpool = torch.nn.Identity()
                self.base_model.fc = torch.nn.Identity()
                self.base_model.load_state_dict(torch.load('./pretrained_backbones/pretrained_resnet_300.pth')['state_dict'])
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnet101_bts':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext50_bts':
            self.base_model = models.resnext50_32x4d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext101_bts':
            self.base_model = models.resnext101_32x8d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'mobilenetv2_bts':
            self.base_model = models.mobilenet_v2(pretrained=True).features
            self.feat_inds = [2, 4, 7, 11, 19]
            self.feat_out_channels = [16, 24, 32, 64, 1280]
            self.feat_names = []
        else:
            print('Not supported encoder: {}'.format(params.encoder))

    def forward(self, x):
        feature = x
        skip_feat = []
        i = 1
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if self.params.encoder == 'mobilenetv2_bts':
                if i == 2 or i == 4 or i == 7 or i == 11 or i == 19:
                    skip_feat.append(feature)
            else:
                if any(x in k for x in self.feat_names):
                    skip_feat.append(feature)
            i = i + 1
        return skip_feat

class sn_warper(nn.Module):
    def __init__(self, params):
        super(sn_warper, self).__init__()
        self.params = params

    def forward(self, pred_depth):
        # partial differentiations
        depth_x1 = pred_depth[:,:,:,1:]
        depth_y1 = pred_depth[:,:,1:,:]

        dZ_x1 = depth_x1 - pred_depth[:,:,:,:-1]
        dZ_y1 = depth_y1 - pred_depth[:,:,:-1,:]

        dX_x1 = pred_depth[:,:,:,:-1] / (self.params.fx * (pred_depth.shape[3]/1350)) + (((torch.tile(torch.arange(pred_depth[:,:,:,:-1].shape[-1]), (pred_depth[0,0].shape[0], 1))) - (self.params.cx * (pred_depth.shape[3]/1350))) / (self.params.fx * (pred_depth.shape[3]/1350))).to('cuda:0') * dZ_x1
        dX_y1 = (((torch.tile(torch.arange(pred_depth[0,0].shape[1]), (pred_depth[:,:,:-1,:].shape[-2], 1))).to('cuda:0') - (self.params.cx * (pred_depth.shape[3]/1350))) / (self.params.fx * (pred_depth.shape[3]/1350))) * dZ_y1

        dY_x1 = ((torch.tile(torch.arange(pred_depth[0,0].shape[0]).unsqueeze(1), (1, pred_depth[:,:,:,:-1].shape[-1])) - (self.params.cy * (pred_depth.shape[2]/1080))) / (self.params.fy * (pred_depth.shape[2]/1080))).to('cuda:0') * dZ_x1
        dY_y1 = pred_depth[:,:,:-1,:] / (self.params.fy * (pred_depth.shape[2]/1080)) + (((torch.tile(torch.arange(pred_depth[:,:,:-1,:].shape[-2]).unsqueeze(1),(1, pred_depth[0,0].shape[1]))) - (self.params.cy * (pred_depth.shape[2]/1080))) / (self.params.fx * (pred_depth.shape[2]/1080))).to('cuda:0') * dZ_y1

        # tangent vectors
        vx = torch.cat([dX_x1, dY_x1, dZ_x1], dim=1)
        vy = torch.cat([dX_y1, dY_y1, dZ_y1], dim=1)

        # cross-product
        cx = vx[:, 1, :-1] * vy[:, 2, :, :-1] - vx[:, 2, :-1] * vy[:, 1, :, :-1]
        cy = vx[:, 2, :-1] * vy[:, 0, :, :-1] - vx[:, 0, :-1] * vy[:, 2, :, :-1]
        cz = vx[:, 0, :-1] * vy[:, 1, :, :-1] - vx[:, 1, :-1] * vy[:, 0, :, :-1]

        n = torch.zeros((pred_depth.shape[0], 3, pred_depth.shape[2], pred_depth.shape[3]))
        n[:, 0, :-1, :-1] = -cx
        n[:, 1, :-1, :-1] = cy
        n[:, 2, :-1, :-1] = cz
        cz[cz < 0] = 0
        n_power = n ** 2
        n_power_sum = n_power[:, 0] + n_power[:, 1] + n_power[:, 2]
        n_sqrt = torch.sqrt(n_power_sum + 1e-9).unsqueeze(1)
        n_unit = torch.divide(n, n_sqrt)
        #n_unit[:, :2, :-1, :-1] = torch.nan_to_num(n_unit[:, :2, :-1, :-1], posinf=1, neginf=-1, nan=-1.0)
        warped_sn = torch.nan_to_num(n_unit, posinf=1, neginf=-1, nan=0.0)
        # n_unit[n_unit > 1] = 1
        # n_unit[n_unit < -1] = -1
        return warped_sn.to('cuda:0')


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Col3D_MTL(nn.Module):
    def __init__(self, params):
        super(Col3D_MTL, self).__init__()
        self.params = params
        self.encoder = encoder(params)
        self.daspp = daspp(params, self.encoder.feat_out_channels, params.filter_size)
        self.depth_decoder = depth_decoder(params, self.encoder.feat_out_channels, params.filter_size)
        self.normal_decoder = normal_decoder(params, self.encoder.feat_out_channels, params.filter_size)
        self.sn_warper = sn_warper(params)  # D2SN : SN from DIG

    def forward(self, x, focal):
        skip_feat = self.encoder(x)
        daspp_feat = self.daspp(skip_feat)
        if self.params.multitask:
          if self.params.CL == 'True' and self.params.concat == 'False':   # Case: X-Consistency w/o feature concatenation
                lpg8x8_n, lpg4x4_n, lpg2x2_n, reduc1x1_n, normal_est, depth_from_normal = self.normal_decoder(skip_feat, daspp_feat, focal)
                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, normal_from_depth = self.depth_decoder(skip_feat, daspp_feat, focal)
                warped_sn = self.sn_warper(depth_est)
                return lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, lpg8x8_n, lpg4x4_n, lpg2x2_n, reduc1x1_n, normal_est, depth_from_normal, warped_sn#normal_from_depth

          elif self.params.CL == 'False' and self.params.concat == 'True':   # Case: Feature concatenation
                lpg8x8_n, lpg4x4_n, lpg2x2_n, reduc1x1_n, normal_est = self.normal_decoder(skip_feat, daspp_feat, focal)
                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = self.depth_decoder(skip_feat, daspp_feat, focal, normal=normal_est)
                return lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, lpg8x8_n, lpg4x4_n, lpg2x2_n, reduc1x1_n, normal_est

          elif self.params.full_concat == 'True':   # Case: Full feature concatenation
                lpg8x8_n, lpg4x4_n, lpg2x2_n, reduc1x1_n, normal_est = self.normal_decoder(skip_feat, daspp_feat, focal)
                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = self.depth_decoder(skip_feat, daspp_feat, focal, normal=normal_est, lpg8_normal=lpg8x8_n, lpg4_normal=lpg4x4_n, lpg2_normal=lpg2x2_n)
                return lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, lpg8x8_n, lpg4x4_n, lpg2x2_n, reduc1x1_n, normal_est

          elif self.params.CL == 'True' and self.params.concat == 'True':   # Case: X-Consistency w/ feature concatenation
                lpg8x8_n, lpg4x4_n, lpg2x2_n, reduc1x1_n, normal_est, depth_from_normal = self.normal_decoder(skip_feat, daspp_feat,focal)
                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, normal_from_depth = self.depth_decoder(skip_feat, daspp_feat, focal, normal=normal_est)
                return lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, lpg8x8_n, lpg4x4_n, lpg2x2_n, reduc1x1_n, normal_est, depth_from_normal, normal_from_depth

          else:   # Case: Depth and surface normal
                lpg8x8_n, lpg4x4_n, lpg2x2_n, reduc1x1_n, normal_est = self.normal_decoder(skip_feat, daspp_feat, focal)
                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = self.depth_decoder(skip_feat, daspp_feat, focal)
                return lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, lpg8x8_n, lpg4x4_n, lpg2x2_n, reduc1x1_n, normal_est

        else:   # Case: Depth estimation
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = self.depth_decoder(skip_feat, daspp_feat, focal)
            return lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est
