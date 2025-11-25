#=======       bvh2smpl        ========#
import logging
import pickle
import sys
import numpy as np
import smplx
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
import pdb
import math
 
# import bvh_tool, 
# import quat

import numpy as np
# 把旧 pickle 会找的 numpy 别名一次性建出来
_compat = {
    'int': np.int32,      # 旧脚本里 int 默认 32 位
    'float': np.float32,  # SMPL 原始模型保存的是 float32
    'complex': np.complex64,
    'bool': np.bool_,
    'object': np.object_,
    'str': np.str_,
    'unicode': np.str_,   # np.unicode_ 已合并到 str_
    'infty': np.inf, 
}
for alias, target in _compat.items():
    if not hasattr(np, alias):
        setattr(np, alias, target)
 
names = [
    "Hips",
    "LeftUpLeg",
    "RightUpLeg",
    "Spine1",
    "LeftLeg",
    "RightLeg",
    "Spine2",
    "LeftFoot",#7
    "RightFoot",
    "Spine3",
    "LeftFoot",#10
    "RightFoot",
    "Neck",
    "LeftShoulder",
    "RightShoulder",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
]#22

def bvh_to_smplx(mocap, n_frames=None,scale_v=1):
    # with open(bvh_file, 'r') as f:
    #     mocap = Bvh(f.read())
 
    if n_frames is None:
        num_frames = len(mocap.frames)
    else:
        num_frames = min(n_frames, len(mocap.frames))
 
    smplx_poses = np.zeros((num_frames, 24*3))
    smplx_trans = np.zeros((num_frames, 3))
 
    bvh_joint_names = set(mocap.get_joints_names())
 
    for joint_index, joint_name in enumerate(mocap.get_joints_names()):
        print(joint_name, joint_index)
 
    # 定义一个绕x转90°，再绕y转90°的旋转，用于校正BVH与SMPLX坐标系的差异（z朝前，x朝左，y朝上）
    rotation_correction = R.from_euler('XYZ', [90, 90, 0], degrees=True)
 
    for i in range(num_frames):
        print('Processing frame {}/{}'.format(i, num_frames), end='\r')
        for joint_index,joint_name in enumerate(names):
            if joint_name not in bvh_joint_names:
                logging.error(f'not joint_name:{joint_name}')
                exit(123)
 
            rotation = R.from_euler('XYZ', mocap.frame_joint_channels(i, joint_name, ['Xrotation','Yrotation','Zrotation' ]), degrees=True)
 
            # 仅对根关节（Hips）应用朝向校正
            if joint_name in ['Pelvis','Hips']:
                # rotation = rotation * rotation_correction
                rotation = rotation_correction * rotation
 
            smplx_poses[i,  3*joint_index:3 * (joint_index + 1)] = rotation.as_rotvec()
 
            # 提取根关节平移
            if joint_name in ['Pelvis','Hips']:
                x, y, z = mocap.frame_joint_channels(i, joint_name, [ 'Xposition','Yposition', 'Zposition'])
                smplx_trans[i] = np.array([x, y, z])
 
    # 应用朝向校正
    # smplx_trans = rotation_correction_trans.apply(smplx_trans)
 
    # 反转Y轴平移方向
    # smplx_trans[:, 1] *= -1
 
    # 应用整体缩放
    smplx_trans /=scale_v
 
    return smplx_trans, smplx_poses
 
 
def save_npz(output_file, smplx_trans, smplx_poses, gender='FEMALE', model_type='smplx', frame_rate=30):

    np.savez(output_file, trans=smplx_trans, poses=smplx_poses, gender=gender, surface_model_type=model_type,
             mocap_frame_rate=frame_rate, betas=np.zeros(16))

import os
from pathlib import Path
def process_folder(bvh_dir, smpl_dir, scale_v=1.0):
    bvh_dir = Path(bvh_dir)
    smpl_dir = Path(smpl_dir)
    smpl_dir.mkdir(parents=True, exist_ok=True)

    bvh_files = list(bvh_dir.glob("*.bvh"))
    print(f"Found {len(bvh_files)} BVH files in {bvh_dir}")
    for bvh_file in bvh_files:
        print(f"Processing {bvh_file.name}...")
        with open(bvh_file, 'r') as f:
            mocap = Bvh(f.read())
        trans, rots_o = bvh_to_smplx(mocap,scale_v=scale_v)
        output_file = smpl_dir / (bvh_file.stem + ".npz")
        save_npz(output_file, trans, rots_o, frame_rate=1.0/mocap.frame_time)

 
def main():
    
    model_path=r"f:\fsy\project\HumanML3D\body_models"
    model = smplx.create(model_path=model_path, model_type="smpl", gender="FEMALE", batch_size=1)
    parents = model.parents.detach().cpu().numpy()

    rest = model(  # betas = torch.randn([1, num_betas], dtype=torch.float32)
    )
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24, :]

    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset #修正root
    # offsets *= scale_v


    #单文件
    bvh_file = r"f:\fsy\project\BVH\F01\F01A1V1.bvh"
    output_file = r"F:\fsy\project\kae_process\kae_smpl\F01A1V1.npz"
    scale_v=100.0

    with open(bvh_file, 'r') as f:
            mocap = Bvh(f.read())

    trans, rots_o = bvh_to_smplx(mocap,scale_v=scale_v)
    save_npz(output_file, trans, rots_o, frame_rate=1.0/mocap.frame_time)

    #save pkl
    pkl_out = {"poses": rots_o, "trans": trans, "offsets": offsets, "parents": parents}
    pickle.dump(pkl_out, open(f"template.pkl", "wb"))    


    #批量文件
    # bvh_dir=r"f:\fsy\project\BVH\F01"
    # smpl_dir=r"F:\fsy\project\kae_process\kae_smpl\F01"
    # scale_v=100.0
    # process_folder(bvh_dir, smpl_dir, scale_v=scale_v)

if __name__ == "__main__":
    main()
