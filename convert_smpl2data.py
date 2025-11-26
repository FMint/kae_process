#=======       smpl2data        ========#
# ...existing code...
# 参考 amass2humanml3d 的处理流程，把 SMPL npz（trans + poses[axis-angle]）转换为 HumanML3D 的 data 向量

import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import joblib

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

# from common.skeleton import Skeleton
from common.quaternion import (
    qrot_np, qmul_np, qinv_np, qbetween_np, qfix, quaternion_to_cont6d_np
)
# from paramUtil import t2m_raw_offsets, t2m_kinematic_chain

# # 与 [motion_representation.ipynb](e:\szu\2025-11\motion_representation.ipynb) 保持一致的索引设置
# # Lower legs
# l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# # Face direction index: r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# # l_hip, r_hip
# r_hip, l_hip = 2, 1

joints_num = 22

# T2M 骨架
skeleton = joblib.load("template.pkl")
offsets = (skeleton["offsets"]).astype(np.float32) #(J,3)
parents = skeleton["parents"] #(J)

def forward_kinematics_np(quat, trans):
    """
    输入:
        quat: (T, J, 4) 四元数表示的关节旋转
        trans: (T, 3) 根关节平移
    输出:
        positions: (T, J, 3) 全局关节位置
    """
    T, J, _ = quat.shape
    positions = np.zeros((T, J, 3), dtype=np.float32)
    global_rot = np.zeros((T, J, 4), dtype=np.float32)

    # 计算每个关节的全局位置
    for t in range(T):
        positions[t, 0] = trans[t]
        global_rot[t, 0] = quat[t, 0]
        for j in range(1,J):
            parent = parents[j]
            global_rot[t, j] = qmul_np(global_rot[t, parent][None, :], quat[t, j][None, :])[0]
            offset = offsets[j]
            rotated_offset = qrot_np(global_rot[t, parent][None, :], offset[None, :])[0]
            positions[t, j] = positions[t, parents[j]] + rotated_offset
    return positions

def aa_to_quat_wxyz(aa):
    """
    axis-angle -> quaternion (w,x,y,z)
    aa: (T, J, 3)
    """
    T, J, _ = aa.shape
    q_xyzw = R.from_rotvec(aa.reshape(-1, 3)).as_quat()  # (N,4) -> [x,y,z,w]
    q_xyzw = q_xyzw.reshape(T, J, 4)
    # 转为 (w,x,y,z)
    q_wxyz = np.concatenate([q_xyzw[..., 3:4], q_xyzw[..., 0:3]], axis=-1).astype(np.float32)
    return q_wxyz

def fk_smpl22_positions(trans, poses_axis_angle):
    """
    使用 T2M Skeleton 做前向运动学，得到全局关节位置。
    trans: (T,3) in meters
    poses_axis_angle: (T, 22*3) axis-angle
    return:
      positions: (T, 22, 3)
    """
    T = trans.shape[0]
    aa = poses_axis_angle.reshape(T, joints_num, 3).astype(np.float32)
    quat_params = aa_to_quat_wxyz(aa)  # (T,22,4)

    positions = forward_kinematics_np(quat_params, trans.astype(np.float32))  # (T,22,3)
    return positions

def foot_detect(positions, thres):
    velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2

    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2

    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)

    return feet_l, feet_r

def _safe_norm(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def _root_orientation_from_positions(frame_pos, face_joint_indx):
    # frame_pos： (J,3)
    # 用髋+肩的 across 定根坐标系：x=across, z=cross(up,x), y=cross(z,x)
    # x朝前，z朝右，y朝上
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = frame_pos[r_hip] - frame_pos[l_hip] + frame_pos[sdr_r] - frame_pos[sdr_l]
    x = _safe_norm(across)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    z = _safe_norm(np.cross(up, x))
    y = _safe_norm(np.cross(z, x))
    Rm = np.stack([x, y, z], axis=1)  # 列为基
    q_xyzw = R.from_matrix(Rm).as_quat()  # x,y,z,w
    q_wxyz = np.concatenate([q_xyzw[3:4], q_xyzw[:3]], axis=0).astype(np.float32)
    return q_wxyz  # (4,)

def inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False):
    """
    从全局关节位置 positions(T,J,3) 和静态 offsets(J,3)、parents(J) 恢复局部四元数 quat_local(T,J,4)
    思路：逐关节让 parent 坐标系下的 offset 向量旋到当前骨向量；转过去所需的最小旋转就是该关节的局部四元数
    """
    T, J, _ = positions.shape
    quat_local = np.zeros((T, J, 4), dtype=np.float32)
    quat_global = np.zeros((T, J, 4), dtype=np.float32)

    for t in range(T):
        root_quat = _root_orientation_from_positions(positions[t], face_joint_indx) #wxyz
        quat_local[t, 0] = root_quat
        quat_global[t, 0] = root_quat

        for j in range(1, J):
            parent = parents[j]
            parent_quat = quat_global[t, parent]
            parent_quat_inv = qinv_np(parent_quat[None, :])[0]

            bone_dir = positions[t, j] - positions[t, parent]  # (3,)
            bone_dir_local = qrot_np(parent_quat_inv[None, :], bone_dir[None, :])[0]
            offset = offsets[j]
            offset_norm = np.linalg.norm(offset)
            if offset_norm < 1e-8:
                quat_rel = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                target_dir = bone_dir_local / np.linalg.norm(bone_dir_local) * offset_norm
                quat_rel = qbetween_np(offset[None, :], target_dir[None, :])[0]  # (4,)
            quat_local[t, j] = quat_rel
            quat_global[t, j] = qmul_np(parent_quat[None, :], quat_rel[None, :])[0]

    return quat_local


def get_cont6d_params(positions):
    """
    复用 HumanML3D 的 IK + cont6d 生成方式：
      - IK 得到每帧关节四元数（smooth_forward=True）
      - 转连续6D
      - 根线速度/角速度
    """
    # skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # (T, J, 4)
    # quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
    quat_params = inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
    # 修正四元数
    quat_params = qfix(quat_params)

    cont_6d_params = quaternion_to_cont6d_np(quat_params)  # (T, J, 6)

    r_rot = quat_params[:, 0].copy()  # (T,4) root quat
    '''Root Linear Velocity'''
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()  # (T-1,3)
    velocity = qrot_np(r_rot[1:], velocity)
    '''Root Angular Velocity'''
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # (T-1,4)

    return cont_6d_params, r_velocity, velocity, r_rot

def smplnpz_to_data(npz_path, out_vec_dir, out_joints_dir=None, feet_thre=0.002):
    """
    从 SMPL npz(trans, poses) 生成 HumanML3D data 并保存 .npy
    """
    os.makedirs(out_vec_dir, exist_ok=True)
    if out_joints_dir is not None:
        os.makedirs(out_joints_dir, exist_ok=True)

    npz = np.load(npz_path)
    trans = npz["trans"].astype(np.float32)           # (T,3)
    poses = npz["poses"].astype(np.float32)           # (T,24*3)
    T = trans.shape[0]
    poses = poses.reshape(T, 24, 3)[:, :22, :3]       #23\24为0
    poses = poses.reshape(T, -1)                      # (T, 22*3) 轴角

    # 1) FK 得到原始全局关节位置
    positions = fk_smpl22_positions(trans, poses)     # (T,22,3)

    # 2) 放到地面
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height              # y 轴放到地面，所有关节

    # 3) XZ 原点——把初始帧移到原点
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1], dtype=np.float32)
    positions = positions - root_pose_init_xz

    # # 4) 初始统一朝向 Z+
    # r_hip_i, l_hip_i, sdr_r_i, sdr_l_i = face_joint_indx
    # across1 = root_pos_init[r_hip_i] - root_pos_init[l_hip_i]
    # across2 = root_pos_init[sdr_r_i] - root_pos_init[sdr_l_i]
    # across = across1 + across2
    # across = across / np.linalg.norm(across, axis=-1, keepdims=True)
    # forward_init = np.cross(np.array([[0, 1, 0]], dtype=np.float32), across, axis=-1)
    # forward_init = forward_init / np.linalg.norm(forward_init, axis=-1, keepdims=True)
    # target = np.array([[0, 0, 1]], dtype=np.float32)
    # root_quat_init = qbetween_np(forward_init, target)
    # root_quat_init = np.ones(positions.shape[:-1] + (4,), dtype=np.float32) * root_quat_init
    # positions = qrot_np(root_quat_init, positions)

    # # 5) 作为 new ground truth positions
    global_positions = positions.copy()

    # 6) 脚接触
    feet_l, feet_r = foot_detect(positions, feet_thre)

    # 7) cont6d / root 速度 等
    cont_6d_params, r_velocity_q, velocity, r_rot = get_cont6d_params(positions)

    # 8) RIFKE（根局部化 + 用 r_rot 对齐所有帧到根朝向坐标系）
    def get_rifke(pos):
        pos = pos.copy()
        pos[..., 0] -= pos[:, 0:1, 0]
        pos[..., 2] -= pos[:, 0:1, 2]
        return qrot_np(np.repeat(r_rot[:, None], pos.shape[1], axis=1), pos)
    positions_rifke = get_rifke(positions)

    # 9) 构建 data 向量（与motion_representation.ipynb一致）
    root_y = positions[:, 0, 1:2]  # (T,1)
    r_velocity = np.arcsin(r_velocity_q[:, 2:3])      # (T-1,1) 取 y 轴分量
    l_velocity = velocity[:, [0, 2]]                  # (T-1,2)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)  # (T-1,4)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)           # (T, (J-1)*6)
    ric_data = positions_rifke[:, 1:].reshape(len(positions_rifke), -1)         # (T, (J-1)*3)

    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)                            # (T-1, J*3)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)                      # (T-1, ...)

    # 保存
    base = os.path.splitext(os.path.basename(npz_path))[0]
    out_vec_path = os.path.join(out_vec_dir, base + ".npy")
    np.save(out_vec_path, data)

    if out_joints_dir is not None:
        # 可选：保存用 recover_from_ric 可重建的 joints（用于自检）
        # 这里直接保存 positions（对齐后）
        out_joints_path = os.path.join(out_joints_dir, base + ".npy")
        positions = positions[:-1,...]
        np.save(out_joints_path, positions.astype(np.float32))

    print(f"[OK] {base}: data {data.shape}, positions {positions.shape}")
    return data, positions, global_positions, l_velocity

# 示例用法
if __name__ == "__main__":

    #单文件
    # # 输入：由 BVH 转来的 SMPL npz（键：trans, poses）
    # smpl_npz = r"f:\fsy\project\kae_process\kae_smpl\F01A1V1.npz"
    # # 输出目录：data（HumanML3D/new_joint_vecs 风格）和可选 joints
    # out_vec_dir = r"f:\fsy\project\kae_process\kae_data"
    # out_joints_dir = r"f:\fsy\project\kae_process\kae_joints"
    # smplnpz_to_data(smpl_npz, out_vec_dir, out_joints_dir, feet_thre=0.002)

    #批量处理文件夹
    smpl_dir = r"f:\fsy\project\kae_process\kae_smpl"
    data_dir = r"f:\fsy\project\kae_process\kae_data"
    joint_dir= r"f:\fsy\project\kae_process\kae_joints"
    smpl_files = [os.path.join(smpl_dir, f) for f in os.listdir(smpl_dir) if f.endswith(".npz")]
    for smpl_npz in smpl_files:
        smplnpz_to_data(smpl_npz, data_dir, joint_dir, feet_thre=0.002)