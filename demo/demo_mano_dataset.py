# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from config import cfg
import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from utils.transforms import world2cam, cam2pixel, pixel2cam
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from utils.mano_reconstruct import reconstruct
from model import get_R50_ViT_B
from utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image, transform_input_to_output_space, trans_point2d
from utils.vis import vis_keypoints, vis_3d_keypoints
from dataset import Dataset


from tqdm import tqdm

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import trimesh
import json
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids',default='0-1')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch',default ='20')
    #parser.add_argument('--bbox',type=list,dest='bbox')
    #parser.add_argument('--focal',type=list,dest='focal')
    #parser.add_argument('--princpt',type=list,dest='princpt')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

# argument parsing



#使用测试集数
def one_sample(model,testset_loader,id,use_gt_hand_type,visual):
    _inputs,_targets,_meta_info = testset_loader[id]

    img = _inputs['img'].cuda()
    img = img[None,:,:,:]
    #bbox = _meta_info['bbox']
    inv_trans = _meta_info['inv_trans']
    cam_param = _meta_info['cam_param']
    focal = cam_param['focal']
    princpt = cam_param['princpt']
    inputs = {'img': img}
    targets = {}
    meta_info = {}

    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')

    img = img[0].cpu().numpy().transpose(1,2,0) # cfg.input_img_shape[1], cfg.input_img_shape[0], 3

    joint_coord = out['joint_coord'][0].cpu().numpy() # x,y pixel, z root-relative discretized depth
    rel_root_depth = out['rel_root_depth'][0].cpu().numpy() # discretized depth
    hand_type = out['hand_type'][0].cpu().numpy() # handedness probability

    # restore joint coord to original image space and continuous depth space
    joint_coord[:,0] = joint_coord[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    joint_coord[:,1] = joint_coord[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    joint_coord[:,:2] = np.dot(inv_trans, np.concatenate((joint_coord[:,:2], np.ones_like(joint_coord[:,:1])),1).transpose(1,0)).transpose(1,0)
    joint_coord[:,2] = (joint_coord[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

    # restore right hand-relative left hand depth to continuous depth space
    rel_root_depth = (rel_root_depth/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)

    #ground truth
    hand_type_gt = _targets['hand_type']
    pred_joint_coord_cam_gt = _targets['joint_cam']

    # right hand root depth == 0, left hand root depth == rel_root_depth
    joint_coord[joint_type['left'],2] += rel_root_depth

    # 得到相机空间下的z
    joint_coord[joint_type['right'],2] += pred_joint_coord_cam_gt[root_joint_idx['right'],2]
    joint_coord[joint_type['left'],2] += pred_joint_coord_cam_gt[root_joint_idx['left'],2]

    #得到相机空间下的xyz
    pred_joint_coord_cam = pixel2cam(joint_coord, focal, princpt)
    #interhand和mano关键关系映射
    map = [20,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,19,18,17,16]
    #map = [0,4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17]
    if False:
        import matplotlib.pyplot as plt
        plt.imshow(img)
        img_name = 'output/vis/test_%d.jpg'% id
        plt.savefig(img_name)
        plt.close()
    # handedness
    joint_valid = np.zeros((joint_num*2), dtype=np.float32)
    if use_gt_hand_type:
        used_type = hand_type_gt
    else:
        used_type = hand_type

    right_exist = False

    if used_type[0] > 0.5: 
        right_exist = True
        joint_valid[joint_type['right']] = 1
        right_hand_cam_gt = pred_joint_coord_cam_gt[joint_type['right']][map]
        right_hand_cam = pred_joint_coord_cam[joint_type['right']][map]

    left_exist = False
    if used_type[1] > 0.5:
        left_exist = True
        joint_valid[joint_type['left']] = 1
        left_hand_cam_gt = pred_joint_coord_cam_gt[joint_type['left']][map]
        left_hand_cam = pred_joint_coord_cam[joint_type['left']][map]

    if right_exist:
        reconstruct(right_hand_cam, right_hand_cam_gt,False, 'right',id)
    if left_exist:
        reconstruct(left_hand_cam, left_hand_cam_gt,False, 'left',id)

if __name__ == "__main__":
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
    joint_num = 21 # single hand
    root_joint_idx = {'right': 20, 'left': 41}
    joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}
    #skeleton = load_skeleton(osp.join('./data/RHD/data/skeleton.txt'), joint_num*2)
    # snapshot load
    model_path = './output/model_dump/snapshot_%d.pth.tar' % int(args.test_epoch)
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model = get_R50_ViT_B('test', joint_num)
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()
    testset_loader = Dataset(transforms.ToTensor(), 'test')
    for i in range(1):
        one_sample(model,testset_loader,i,True,False)