from manopth import demo
from manopth import manolayer
from utils.aik_utils import AIK, align, vis
import torch
import numpy as np
import os
import smplx
import os.path as osp
import cv2
def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()
'''
未调试好
'''
def rendermano(pose,hand_type):
    smplx_path_left = '/media/sdb/sunqy/mano_proj/mano_v1_2/models/MANO_LEFT.pkl'
    smplx_path_right = '/media/sdb/sunqy/mano_proj/mano_v1_2/models/MANO_RIGHT.pkl'
    mano_layer = {'right': smplx.create(smplx_path_right, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path_left, 'mano', use_pca=False, is_rhand=False)}
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:,0,:] *= -1
    save_path = osp.join('/media/sdb/sunqy/Code/hand/interhand/mesh')
    os.makedirs(save_path, exist_ok=True)
    #img_height, img_width, _ = img.shape
    mano_pose = torch.FloatTensor(pose).view(-1,3)
    root_pose = mano_pose[0].view(1,3)
    hand_pose = mano_pose[1:,:].view(1,-1)
    output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose)
    mesh = output.vertices[0].detach().numpy() * 1000 
    save_obj(mesh, mano_layer[hand_type].faces, osp.join(save_path,hand_type + '.obj'))

def matrix2vector(matrix):
    result = []
    matrixs = matrix[0]
    for i in range(len(matrixs)):
        m = matrixs[i]
        a,_ = cv2.Rodrigues(m)
        a = a.squeeze()
        result.append(a)
    result = np.array(result)
    return result

def reconstruct(pre_j3ds,j3ds_gt, visual, hand_type, id, op_shape=None):
    pose0 = torch.eye(3).repeat(1, 16, 1, 1).cuda()
    mano = manolayer.ManoLayer(flat_hand_mean=True,
                               side=hand_type,
                               mano_root='mano/models',
                               use_pca=False,
                               root_rot_mode='rotmat',
                               joint_rot_mode='rotmat').cuda()
    pose = []

    j3d_pre = pre_j3ds.copy()/1000.0 #实际输入的单位是mm
    j3ds_gt = j3ds_gt.copy()/1000.0
    if op_shape != None:
        op_shape = torch.tensor(op_shape).float().unsqueeze(0).cuda()


    _, j3d_p0_ops = mano(pose0,op_shape)#得到模板关键点位置

    template = j3d_p0_ops.cpu().numpy().squeeze() / 1000.0  # template, m
    ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(j3d_pre[9] - j3d_pre[0])
    j3d_pre_process = j3d_pre * ratio  # template, m
    
    ratio_gt = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(j3ds_gt[9] - j3ds_gt[0])
    j3d_gt_process = j3ds_gt * ratio_gt
    
    pred_root = j3d_pre_process[0]
    gt_root = j3d_gt_process[0]

    j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]#得到相对于模板原点的关节点位置
    j3d_gt_process = j3d_gt_process - j3d_gt_process[0] + template[0]

    pose_R = AIK.adaptive_IK(template, j3d_pre_process)#使用AIK得到pose
    if False:
        pose_vector = matrix2vector(pose_R)
        rendermano(pose_vector,hand_type)
    
    pose_R = torch.from_numpy(pose_R).float()
    hand_verts, j3d_recon = mano(pose_R,op_shape)
    # visualization
    if visual:
        demo.display_hand({
            'verts': hand_verts.cpu(),
            'joints': j3d_recon.cpu()
        },
            mano_faces=mano.th_faces)

    j3d_recon = j3d_recon.cpu().numpy().squeeze() / 1000.
    #pose_R = pose_R.cpu().numpy().squeeze()

    # visualization
    if visual:
        vis.multi_plot3d([j3d_gt_process, j3d_pre_process], hand_type,id,title=["gt", "pre"])
    j3d_recon = (j3d_recon - template[0]+pred_root) / ratio
    j3d_recon  = j3d_recon * 1000.0

    j3d_gt_process = (j3d_gt_process - template[0]+gt_root) / ratio_gt
    j3d_gt_process = j3d_gt_process * 1000.0

    j3d_pre_process =(j3d_pre_process - template[0]+pred_root) / ratio
    j3d_pre_process = j3d_pre_process * 1000.0

    return j3d_recon, j3d_gt_process, j3d_pre_process