'''
preds:21X3,相机坐标系下坐标 numpy
targets:21x3 numpy
key:数据集名称 Interhand2.6M RHD STB
'''
import numpy as np 
def global_align(gtj0, prj0, key):
    gtj = gtj0.copy()
    prj = prj0.copy()
    if key in ["STB","RHD"]:
        root_idx = 0
        ref_bone_link = [0, 9] #wrist -> middle1
    elif key in ["Inerhand2.6M"]:
        root_idx = 0
        ref_bone_link = [0, 9]

    pred_align = prj.copy()
    pred_ref_bone_len = np.linalg.norm(prj[ref_bone_link[0]] - prj[ref_bone_link[1]])
    gt_ref_bone_len = np.linalg.norm(gtj[ref_bone_link[0]] - gtj[ref_bone_link[1]])
    scale = gt_ref_bone_len / pred_ref_bone_len
            
    for j in range(21):
        pred_align[j] = gtj[root_idx] + scale * (prj[j] - prj[root_idx])
    return gtj, pred_align

def auc_validate(pred, target,key,evaluator):
    pred_joint = pred.copy()
    gt_joint = target.copy()
    gt_joint, pred_joint_align = global_align(gt_joint, pred_joint, key=key)
    evaluator.feed(gt_joint, pred_joint_align)