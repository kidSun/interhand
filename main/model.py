# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.module import BackboneNet, PoseNet
from nets.loss import JointHeatmapLoss, HandTypeLoss, RelRootDepthLoss, MANOShapeLoss
from config import cfg
import math
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
class Model(nn.Module):
    def __init__(self,net):
        super(Model, self).__init__()

        # modules
        self.net = net
          
        # loss functions
        self.joint_heatmap_loss = JointHeatmapLoss()
        self.rel_root_depth_loss = RelRootDepthLoss()
        self.hand_type_loss = HandTypeLoss()
        self.mano_shape_loss = MANOShapeLoss()

    def render_gaussian_heatmap(self, joint_coord):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        z = torch.arange(cfg.output_hm_shape[0])
        zz,yy,xx = torch.meshgrid(z,y,x)
        xx = xx[None,None,:,:,:].cuda().float(); yy = yy[None,None,:,:,:].cuda().float(); zz = zz[None,None,:,:,:].cuda().float();
        
        x = joint_coord[:,:,0,None,None,None]; y = joint_coord[:,:,1,None,None,None]; z = joint_coord[:,:,2,None,None,None];
        heatmap = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 -(((yy-y)/cfg.sigma)**2)/2 - (((zz-z)/cfg.sigma)**2)/2)
        heatmap = heatmap * 255
        return heatmap
   
    def forward(self, inputs, targets, meta_info, mode):
        input_img = inputs['img']
        batch_size = input_img.shape[0]
        #pose的gt也需要修改吧
        joint_heatmap_out, rel_root_depth_out, hand_type = self.net(input_img)
        #转换成uvd
        z, idx_z = torch.max(joint_heatmap_out,2)
        zy, idx_zy = torch.max(z,2)
        zyx, joint_x = torch.max(zy,2)
        joint_x = joint_x[:,:,None]
        joint_y = torch.gather(idx_zy, 2, joint_x)
        joint_z = torch.gather(idx_z, 2, joint_y[:,:,:,None].repeat(1,1,1,cfg.output_hm_shape[1]))[:,:,0,:]
        joint_z = torch.gather(joint_z, 2, joint_x)
        joint_coord_out = torch.cat((joint_x, joint_y, joint_z),2).float()
        
        #_joint_ = utils.uvd2xyz(uvd)
        if mode == 'train':
            target_joint_heatmap = self.render_gaussian_heatmap(targets['joint_coord'])
            

            loss = {}
            loss['joint_heatmap'] = self.joint_heatmap_loss(joint_heatmap_out, target_joint_heatmap, meta_info['joint_valid'])
            loss['rel_root_depth'] = self.rel_root_depth_loss(rel_root_depth_out, targets['rel_root_depth'], meta_info['root_valid'])
            loss['hand_type'] = self.hand_type_loss(hand_type, targets['hand_type'], meta_info['hand_type_valid'])
            #loss['mano_shape'] = self.mano_shape_loss(shape_vector,targets['shape'],meta_info['shape_valid'])
            #loss['mano_pose'] = self.mano_pose_loss(shape_vector,targets['pose'],meta_info['pose_valid'])

            return loss
        elif mode == 'test':
            out = {}
            out['joint_coord'] = joint_coord_out
            out['rel_root_depth'] = rel_root_depth_out
            out['hand_type'] = hand_type
            if 'inv_trans' in meta_info:
                out['inv_trans'] = meta_info['inv_trans']
            if 'joint_coord' in targets:
                out['target_joint'] = targets['joint_coord']
            if 'joint_valid' in meta_info:
                out['joint_valid'] = meta_info['joint_valid']
            if 'hand_type_valid' in meta_info:
                out['hand_type_valid'] = meta_info['hand_type_valid']
            return out
    
    #shape采用原来的方案
    #(21,64,64,64) resize (21,64x64x64)
    #embeding -> (21,764)
    #attention x N
    #(21x764) -> (21,3) 和 label做loss


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode, joint_num):
    backbone_net = BackboneNet()
    pose_net = PoseNet(joint_num)

    if mode == 'train':
        backbone_net.init_weights()
        pose_net.apply(init_weights)

    model = Model(backbone_net, pose_net)
    return model


def get_R50_ViT_B(mode,joint_num):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_skip = 3
    config_vit.n_classes = 9#output channel of network
    img_size = 224
    vit_patches_size = 16
    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
    if mode == 'train':
        net.load_from(weights=np.load('/data/home/scv2440/run/sqy/InterHand2.6M/pretrain/imagenet21k_R50+ViT-B_16.npz'))
    #input = torch.rand(1,3,224,224)
    #net(input)
    model = Model(net)
    return model