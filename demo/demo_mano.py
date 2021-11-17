# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

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
from config import cfg
from model import get_model
from utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image, transform_input_to_output_space, trans_point2d
from utils.vis import vis_keypoints, vis_3d_keypoints


from manopth import demo
from manopth import manolayer
from tqdm import tqdm
from aik_utils import AIK, align, vis
from aik_utils.eval.zimeval import EvalUtil

def reconstruct(pre_j3ds, key_i , visual, hand_types, op_shape=None):
    pose0 = torch.eye(3).repeat(1, 16, 1, 1)
    mano = manolayer.ManoLayer(flat_hand_mean=True,
                               side=hand_types,
                               mano_root='mano/models',
                               use_pca=False,
                               root_rot_mode='rotmat',
                               joint_rot_mode='rotmat')
    pose = []

    j3d_pre = pre_j3ds[i]/1000.0 #实际输入的单位是mm
    if op_shape != None:
        op_shape = torch.tensor(op_shape).float().unsqueeze(0)
    hand_type = hand_types[i]

    _, j3d_p0_ops = mano(pose0,op_shape)#得到模板关键点位置

    template = j3d_p0_ops.cpu().numpy().squeeze() / 1000.0  # template, m
    ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(j3d_pre[9] - j3d_pre[0])

    j3d_pre_process = j3d_pre * ratio  # template, m
    j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]#得到相对于模板原点的关节点位置

    pose_R = AIK.adaptive_IK(template, j3d_pre_process)#使用AIK得到pose
    pose_R = torch.from_numpy(pose_R).float()

    hand_verts, j3d_recon = mano(pose_R,op_shape)
    # visualization
    if visual:
        demo.display_hand({
            'verts': hand_verts.cpu(),
            'joints': j3d_recon.cpu()
        },
            mano_faces=mano_r.th_faces)

    j3d_recon = j3d_recon.numpy().squeeze() / 1000.
    pose_R = pose_R.numpy().squeeze()
    pose.insert(0,list(pose_R))

    # visualization
    if visual:
        vis.multi_plot3d([j3d_recon, j3d_pre_process], title=["recon", "pre"])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--bbox',type=list,dest='bbox')
    parser.add_argument('--focal',type=list,dest='focal')
    parser.add_argument('--princpt',type=list,dest='princpt')
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
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
joint_num = 21 # single hand
root_joint_idx = {'right': 20, 'left': 41}
joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num,joint_num*2)}
skeleton = load_skeleton(osp.join('../data/InterHand2.6M/annotations/skeleton.txt'), joint_num*2)
# snapshot load
model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test', joint_num)
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

transform = transforms.ToTensor()
img_path = 'input.jpg'
original_img = cv2.imread(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox
bbox = args.bbox
#bbox = [69, 137, 165, 153] # xmin, ymin, width, height
bbox = process_bbox(bbox, (original_img_height, original_img_width, original_img_height))
img, trans, inv_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, cfg.input_img_shape)
img = transform(img.astype(np.float32))/255
img = img.cuda()[None,:,:,:]
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

# right hand root depth == 0, left hand root depth == rel_root_depth
joint_coord[joint_type['left'],2] += rel_root_depth
focal = args.focal
princpt = args.princpt
#get root relativate xyz
pred_joint_coord_cam = pixel2cam(joint_coord, focal, princpt)
#interhand和mano关键关系映射
map = [20,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,19,18,17,16]

# handedness
joint_valid = np.zeros((joint_num*2), dtype=np.float32)
right_exist = False
if hand_type[0] > 0.5: 
    right_exist = True
    joint_valid[joint_type['right']] = 1
    right_hand_cam = pred_joint_coord_cam[joint_type['right']][map]

left_exist = False
if hand_type[1] > 0.5:
    left_exist = True
    joint_valid[joint_type['left']] = 1
    left_hand_cam = pred_joint_coord_cam[joint_type['left']][map]

if right_exist:
    reconstruct(pred_joint_coord_cam[:21], visual, 'right', op_shapes=None)
if left_exist:
    reconstruct(pred_joint_coord_cam[21:], visual, 'left', op_shapes=None)