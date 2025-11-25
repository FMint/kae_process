import numpy as np
import torch
import smplx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

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

def vis_frame_with_skeleton(joints, parents):
    """joints: (24,3), parents: (24,)"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]
    ax.scatter(xs, ys, zs, c='r')

    # 画骨骼连线
    for i in range(len(joints)):
        p = parents[i]
        if p < 0 or p >= len(joints):
            continue
        x = [joints[p, 0], joints[i, 0]]
        y = [joints[p, 1], joints[i, 1]]
        z = [joints[p, 2], joints[i, 2]]
        ax.plot(x, y, z, c='k', linewidth=1)

    # === 关键：设置各轴等比例 ===
    max_range = np.array([
        xs.max() - xs.min(),
        ys.max() - ys.min(),
        zs.max() - zs.min()
    ]).max() / 2.0

    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # ============================

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('First 24 SMPL joints with skeleton')
    plt.show()

def vis_frame(joints):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = joints[:,0], joints[:,1], joints[:,2]
    ax.scatter(xs, ys, zs, c='r')

    # 简单连骨架，parents 你可以从 pkl 里读
    # 这里只画散点示意
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.show()

def main():
    npz_path = r"F:\fsy\project\kae_process\kae_smpl\F01A1V1.npz"
    model_path = r"f:\fsy\project\HumanML3D\body_models"

    data = np.load(npz_path)
    trans = torch.tensor(data["trans"], dtype=torch.float32)    # (N,3)
    poses = torch.tensor(data["poses"], dtype=torch.float32)    # (N,72) 轴角

    model = smplx.create(model_path=model_path, model_type="smpl",
                         gender="FEMALE", batch_size=1)

    frame_idx = 446
    body = model(global_orient=poses[frame_idx:frame_idx+1, :3],
                 body_pose=poses[frame_idx:frame_idx+1, 3:],
                 transl=trans[frame_idx:frame_idx+1])
    joints = body.joints[0].detach().cpu().numpy()  # (J,3)

    # vis_frame(joints[:24])
    vis_frame_with_skeleton(joints[:24], model.parents.detach().cpu().numpy()[:24])

if __name__ == "__main__":
    main()