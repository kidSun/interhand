# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
from config import cfg
from utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from utils.transforms import world2cam, cam2pixel, pixel2cam
from utils.vis import vis_keypoints, vis_3d_keypoints

from utils.mano_reconstruct import reconstruct
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO
import scipy.io as sio

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode):
        #1.读取文件中的数据
        #2.得到原始图片的图像坐标和相机坐标

        self.mode = mode # train, test, val
        self.img_path = './data/InterHand2.6M/images'
        self.annot_path = './data/InterHand2.6M/annotations'
        if self.mode == 'val':
            self.rootnet_output_path = './data/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_val.json'
        else:
            self.rootnet_output_path = './data/InterHand2.6M/rootnet_output/rootnet_interhand2.6m_output_test.json'
        self.transform = transform
        self.joint_num = 21 # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.skeleton = load_skeleton(osp.join(self.annot_path, 'skeleton.txt'), self.joint_num*2)
        
        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []
        
        # load annotation
        print("Load annotation from  " + osp.join(self.annot_path, self.mode))
        db = COCO(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_data.json'))
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)

        ########################################## load mano 
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot.json')) as f:
            manos = json.load(f)
        ##########################################
        if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")
        
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
 
            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            img_path = osp.join(self.img_path, self.mode, img['file_name'])
            #相机坐标单位是mm， focal, princpt相机内参,都是原始尺寸
            campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
            joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]

            joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(self.joint_num*2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
            hand_type = ann['hand_type']
            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)
            ################################################################################
            if str(capture_id) in manos.keys() and str(frame_idx) in manos[str(capture_id)].keys():
                mano_param = manos[str(capture_id)][str(frame_idx)]#不是所有帧都有
                if mano_param['right']!=None and mano_param['left']!=None:
                    shape = np.array(mano_param['right']['shape']+mano_param['left']['shape'],dtype = np.float32)
                    shape_valid = np.array([1]*20,dtype = np.float32)
                    pose = np.array(mano_param['right']['pose']+mano_param['left']['pose'],dtype = np.float32)
                    pose_valid = np.array([1]*96,dtype = np.float32)

                    trans = np.array(mano_param['right']['trans']+mano_param['left']['trans'],dtype = np.float32)
                    trans_valid = np.array([1]*6,dtype = np.float32)

                elif mano_param['right']!=None:
                    shape = np.array(mano_param['right']['shape']+[0]*10,dtype = np.float32)
                    shape_valid = np.array([1]*10+[0]*10, dtype = np.float32)
                    pose = np.array(mano_param['right']['pose']+[0]*48,dtype = np.float32)
                    pose_valid = np.array([1]*48+[0]*48, dtype = np.float32)

                    trans = np.array(mano_param['right']['trans']+[0]*3 ,dtype = np.float32)
                    trans_valid = np.array([1]*3 + [0]*3,dtype = np.float32)

                elif mano_param['left']!=None:
                    shape = np.array([0]*10+mano_param['left']['shape'],dtype = np.float32)
                    shape_valid = np.array([0]*10+[1]*10, dtype = np.float32)               
                    pose = np.array([0]*48 + mano_param['left']['pose'],dtype = np.float32)
                    pose_valid = np.array([0]*48+[1]*48, dtype = np.float32)

                    trans = np.array([0]*3 + mano_param['left']['trans'] ,dtype = np.float32)
                    trans_valid = np.array([0]*3 + [1]*3,dtype = np.float32)
            else:
                shape = np.array([0]*20,dtype = np.float32)
                shape_valid = np.array([0]*20, dtype = np.float32)  
                pose = np.array([0]*96,dtype = np.float32)
                pose_valid = np.array([0]*96, dtype = np.float32)

                trans = np.array([0]*6 ,dtype = np.float32)
                trans_valid = np.array([0]*6 ,dtype = np.float32)
            
            ################################################################################
            if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
                bbox = np.array(rootnet_result[str(aid)]['bbox'],dtype=np.float32)
                abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}
            else:
                img_width, img_height = img['width'], img['height']
                bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))
                abs_depth = {'right': joint_cam[self.root_joint_idx['right'],2], 'left': joint_cam[self.root_joint_idx['left'],2]}

            mano_parm = {'shape':shape,'shape_valid':shape_valid,'pose':pose,'pose_valid':pose_valid,'trans':trans,'trans_valid':trans}

            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            data = {'mano_parm':mano_parm, 'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param, 'bbox': bbox, 'joint': joint, 'hand_type': hand_type, 'hand_type_valid': hand_type_valid, 'abs_depth': abs_depth, 'file_name': img['file_name'], 'capture': capture_id, 'cam': cam, 'frame': frame_idx}
            if hand_type == 'right' or hand_type == 'left':
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)

        self.datalist = self.datalist_sh + self.datalist_ih
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        '''
        joint_cam:相机坐标xyz,单位：mm
        joint_img:图像坐标uv，单位：像素
        joint_coord: uvz
        图像增强，翻转，旋转，颜色改变
        生成输出空间数据：
        transform_input_to_output_space:
        目前只对joint_coord的uv转换成输出空间范围坐标
        将z转换为相对于root的坐标
        将z和rel_root_depth进一步转换成输出空间

        所以要得到相机空间坐标：z:逆变换，加上root. uv:逆变换，内参矩阵->xyz
        如果不单独训练IKNet：需要输入xyz                                                                                                                                                                                  
        '''
        data = self.datalist[idx]
        mano_parm, img_path, bbox, joint, hand_type, hand_type_valid = data['mano_parm'],data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
        joint_cam = joint['cam_coord'].copy(); joint_img = joint['img_coord'].copy(); joint_valid = joint['valid'].copy();
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None]),1)
        
        pose = mano_parm['pose'].copy()
        pose_valid = mano_parm['pose_valid'].copy()

        shape = mano_parm['shape'].copy()
        shape_valid = mano_parm['shape_valid'].copy()
        
        trans = mano_parm['shape'].copy()
        trans_valid = mano_parm['shape_valid'].copy()

        # image load
        img = load_img(img_path)
        # augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(img, bbox, joint_coord, joint_valid, hand_type, self.mode, self.joint_type)
        rel_root_depth = np.array([joint_coord[self.root_joint_idx['left'],2] - joint_coord[self.root_joint_idx['right'],2]],dtype=np.float32).reshape(1)
        root_valid = np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]],dtype=np.float32).reshape(1) if hand_type[0]*hand_type[1] == 1 else np.zeros((1),dtype=np.float32)
        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, self.root_joint_idx, self.joint_type)
        img = self.transform(img.astype(np.float32))/255.
        
        inputs = {'img': img}
        targets = {'pose':pose,'joint_cam':joint_cam,'shape':shape,'joint_coord': joint_coord, 'rel_root_depth': rel_root_depth, 'hand_type': hand_type}
        meta_info = {'pose_valid':pose_valid,'cam_param':data['cam_param'],'bbox':data['bbox'],'trans':trans,'trans_valid':trans_valid, 'joint_valid': joint_valid, 'root_valid': root_valid, 'hand_type_valid': hand_type_valid, 'inv_trans': inv_trans, 'capture': int(data['capture']), 'cam': int(data['cam']), 'frame': int(data['frame'])}
        return inputs, targets, meta_info

    
    '''
    添加验证mano手部重建后的关节点精度，只使用了MPJPE，未来还要加上ACC of PCK
    '''
    def evaluate(self, preds):

        print() 
        print('Evaluation start...')

        gts = self.datalist
        preds_joint_coord, preds_rel_root_depth, preds_hand_type, inv_trans = preds['joint_coord'], preds['rel_root_depth'], preds['hand_type'], preds['inv_trans']
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)
        
        mpjpe_sh = [[] for _ in range(self.joint_num*2)] 
        mpjpe_ih = [[] for _ in range(self.joint_num*2)]
        mpjpe_sh_recon = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih_recon = [[] for _ in range(self.joint_num*2)]
        mpjpe_sh_process = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih_process = [[] for _ in range(self.joint_num*2)]
        mrrpe = []
        acc_hand_cls = 0; hand_cls_cnt = 0;
        for n in range(sample_num):
            data = gts[n]
            bbox, cam_param, joint, gt_hand_type, hand_type_valid = data['bbox'], data['cam_param'], data['joint'], data['hand_type'], data['hand_type_valid']
            focal = cam_param['focal']
            princpt = cam_param['princpt']
            gt_joint_coord = joint['cam_coord']
            joint_valid = joint['valid']
            
            # restore xy coordinates to original image space
            pred_joint_coord_img = preds_joint_coord[n].copy()
            pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/cfg.output_hm_shape[2]*cfg.input_img_shape[1]
            pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/cfg.output_hm_shape[1]*cfg.input_img_shape[0]
            for j in range(self.joint_num*2):
                pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2],inv_trans[n])
            # restore depth to original camera space
            pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)
 

            # mrrpe 得到了相机空间下root关键点的xyz
            if gt_hand_type == 'interacting' and joint_valid[self.root_joint_idx['left']] and joint_valid[self.root_joint_idx['right']]:
                pred_rel_root_depth = (preds_rel_root_depth[n]/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)

                pred_left_root_img = pred_joint_coord_img[self.root_joint_idx['left']].copy()
                pred_left_root_img[2] += data['abs_depth']['right'] + pred_rel_root_depth
                pred_left_root_cam = pixel2cam(pred_left_root_img[None,:], focal, princpt)[0]

                pred_right_root_img = pred_joint_coord_img[self.root_joint_idx['right']].copy()
                pred_right_root_img[2] += data['abs_depth']['right']
                pred_right_root_cam = pixel2cam(pred_right_root_img[None,:], focal, princpt)[0]
                
                
                pred_rel_root = pred_left_root_cam - pred_right_root_cam
                gt_rel_root = gt_joint_coord[self.root_joint_idx['left']] - gt_joint_coord[self.root_joint_idx['right']]
                pred_joint_coord_img[self.joint_type['left'],2] += pred_rel_root_depth#我觉得还是要加上
                mrrpe.append(float(np.sqrt(np.sum((pred_rel_root - gt_rel_root)**2))))

            #如果是单只手，就没必要使用pred_rel_root_depth,双手情况下左手还是相对于右手的
            # add root joint depth
            pred_joint_coord_img[self.joint_type['right'],2] += data['abs_depth']['right']
            pred_joint_coord_img[self.joint_type['left'],2] += data['abs_depth']['left']

            # back project to camera coordinate system
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)#这里的x,y受到了z的影响
            ##################################################################
            map = [20,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,19,18,17,16]
            inv_map = [4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0]
            j3d_recon_r = np.zeros((21,3),dtype = np.float32)
            j3d_recon_l = np.zeros((21,3),dtype = np.float32)
            j3d_gt_process_r = np.zeros((21,3),dtype = np.float32)
            j3d_gt_process_l = np.zeros((21,3),dtype = np.float32)
            j3d_pre_process_r = np.zeros((21,3),dtype = np.float32)
            j3d_pre_process_l = np.zeros((21,3),dtype = np.float32)


            recon_coord_cam = pred_joint_coord_cam.copy()
            recon_coord_gt = gt_joint_coord.copy()
            if gt_hand_type == 'interacting' and joint_valid[self.root_joint_idx['left']] and joint_valid[self.root_joint_idx['right']]:
                right_hand_cam_gt = recon_coord_gt[self.joint_type['right']][map]
                right_hand_cam = recon_coord_cam[self.joint_type['right']][map]
                left_hand_cam_gt = recon_coord_gt[self.joint_type['left']][map]
                left_hand_cam = recon_coord_cam[self.joint_type['left']][map]
                j3d_recon_r, j3d_gt_process_r, j3d_pre_process_r = reconstruct(right_hand_cam, right_hand_cam_gt,False, 'right',0)
                j3d_recon_l, j3d_gt_process_l, j3d_pre_process_l = reconstruct(left_hand_cam, left_hand_cam_gt,False, 'left',0)
            elif joint_valid[self.root_joint_idx['left']]:
                left_hand_cam_gt = recon_coord_gt[self.joint_type['left']][map]
                left_hand_cam = recon_coord_cam[self.joint_type['left']][map]
                j3d_recon_l, j3d_gt_process_l, j3d_pre_process_l = reconstruct(left_hand_cam, left_hand_cam_gt,False, 'left',0)
            elif joint_valid[self.root_joint_idx['right']]:
                right_hand_cam_gt = recon_coord_gt[self.joint_type['right']][map]
                right_hand_cam = recon_coord_cam[self.joint_type['right']][map]
                j3d_recon_r, j3d_gt_process_r, j3d_pre_process_r = reconstruct(right_hand_cam, right_hand_cam_gt,False, 'right',0)
            
            j3d_recon_r = j3d_recon_r - j3d_recon_r[0]
            j3d_recon_l = j3d_recon_l - j3d_recon_l[0]
            j3d_gt_process_r = j3d_gt_process_r - j3d_gt_process_r[0]
            j3d_gt_process_l = j3d_gt_process_l - j3d_gt_process_l[0]
            j3d_pre_process_r = j3d_pre_process_r - j3d_pre_process_r[0]
            j3d_pre_process_l = j3d_pre_process_l - j3d_pre_process_l[0]

            j3d_recon_r = j3d_recon_r[inv_map]
            j3d_recon_l = j3d_recon_l[inv_map]
            j3d_gt_process_r = j3d_gt_process_r[inv_map]
            j3d_gt_process_l = j3d_gt_process_l[inv_map]
            j3d_pre_process_r = j3d_pre_process_r[inv_map]
            j3d_pre_process_l = j3d_pre_process_l[inv_map]
            j3d_recons = np.concatenate((j3d_recon_r,j3d_recon_l),axis=0)
            j3d_gt_process = np.concatenate((j3d_gt_process_r,j3d_gt_process_l),axis=0)
            j3d_pre_process = np.concatenate((j3d_pre_process_r,j3d_pre_process_l),axis=0)
            
            ##################################################################
            

            # root joint alignment 
            for h in ('right', 'left'):
                pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h],None,:]
                gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h],None,:]
            
            # mpjpe
            for j in range(self.joint_num*2):
                if joint_valid[j]:
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                        mpjpe_sh_recon[j].append(np.sqrt(np.sum((j3d_recons[j] - j3d_gt_process[j])**2)))
                        mpjpe_sh_process[j].append(np.sqrt(np.sum((j3d_pre_process[j] - j3d_gt_process[j])**2)))
                    else:
                        mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                        mpjpe_ih_recon[j].append(np.sqrt(np.sum((j3d_recons[j] - j3d_gt_process[j])**2)))
                        mpjpe_ih_process[j].append(np.sqrt(np.sum((j3d_pre_process[j] - j3d_gt_process[j])**2)))


            # handedness accuray
            if hand_type_valid:
                if gt_hand_type == 'right' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] < 0.5:
                    acc_hand_cls += 1
                elif gt_hand_type == 'left' and preds_hand_type[n][0] < 0.5 and preds_hand_type[n][1] > 0.5:
                    acc_hand_cls += 1
                elif gt_hand_type == 'interacting' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] > 0.5:
                    acc_hand_cls += 1
                hand_cls_cnt += 1

            vis = False
            if vis:
                img_path = data['img_path']
                cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                _img = cvimg[:,:,::-1].transpose(2,0,1)
                vis_kps = pred_joint_coord_img.copy()
                vis_valid = joint_valid.copy()
                capture = str(data['capture'])
                cam = str(data['cam'])
                frame = str(data['frame'])
                filename = 'out_' + str(n) + '_' + gt_hand_type + '.jpg'
                vis_keypoints(_img, vis_kps, vis_valid, self.skeleton, filename)

            vis = False
            if vis:
                filename = 'out_' + str(n) + '_3d.jpg'
                vis_3d_keypoints(pred_joint_coord_cam, joint_valid, self.skeleton, filename)
        

        if hand_cls_cnt > 0: print('Handedness accuracy: ' + str(acc_hand_cls / hand_cls_cnt))
        if len(mrrpe) > 0: print('MRRPE: ' + str(sum(mrrpe)/len(mrrpe)))
        print()
 
        tot_err = []
        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num*2):
            tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        print(eval_summary)
        print('MPJPE for all hand sequences: %.2f' % (np.mean(tot_err)))
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num*2):
            mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
        print(eval_summary)
        print('MPJPE for single hand sequences: %.2f' % (np.mean(mpjpe_sh)))
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num*2):
            mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
        print(eval_summary)
        print('MPJPE for interacting hand sequences: %.2f' % (np.mean(mpjpe_ih)))

        eval_summary = 'recon MPJPE for each joint: \n'
        for j in range(self.joint_num*2):
            mpjpe_sh_recon[j] = np.mean(np.stack(mpjpe_sh_recon[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh_recon[j])
        print(eval_summary)
        print('MPJPE for recon single hand sequences: %.2f' % (np.mean(mpjpe_sh_recon)))
        print()

        eval_summary = 'recon MPJPE for each joint: \n'
        for j in range(self.joint_num*2):
            mpjpe_ih_recon[j] = np.mean(np.stack(mpjpe_ih_recon[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih_recon[j])
        print(eval_summary)
        print('MPJPE for recon interacting hand sequences: %.2f' % (np.mean(mpjpe_ih_recon)))
        print()

        eval_summary = 'process MPJPE for each joint: \n'
        for j in range(self.joint_num*2):
            mpjpe_sh_process[j] = np.mean(np.stack(mpjpe_sh_process[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh_process[j])
        print(eval_summary)
        print('MPJPE for process single hand sequences: %.2f' % (np.mean(mpjpe_sh_process)))
        print()

        eval_summary = 'process MPJPE for each joint: \n'
        for j in range(self.joint_num*2):
            mpjpe_ih_process[j] = np.mean(np.stack(mpjpe_ih_process[j]))
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih_process[j])
        print(eval_summary)
        print('MPJPE for process interacting hand sequences: %.2f' % (np.mean(mpjpe_ih_process)))
