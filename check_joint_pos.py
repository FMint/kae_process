import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 触发 3D 支持
import numpy as np

# kae_pos=r"F:\fsy\project\HumanML3D\joints\000001.npy"
kae_pos=r"F:\fsy\project\kae_process\kae_pos3d\F01A1V1.npy"
# kae_pos=r"F:\fsy\project\HumanML3D\kae\new_joints\F01A1V1.npy"
# kae_pos=r"F:\fsy\project\HumanML3D\HumanML3D\new_joints\000000.npy"
data_kae_pos=np.load(kae_pos)
print("kae pos shape:", data_kae_pos.shape)
print(data_kae_pos[0])

joints = data_kae_pos      # (T,24,3)
frame_id = 0               # 看第 0 帧
pts = joints[frame_id]     # (24,3)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

# 画关节点
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='r')
for j in range(pts.shape[0]):
    ax.text(pts[j, 0], pts[j, 1], pts[j, 2], str(j), fontsize=8)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(f'frame {frame_id} 3D view')

# === 让坐标系等距 ===
x = pts[:, 0]
y = pts[:, 1]
z = pts[:, 2]

max_range = np.array([x.max()-x.min(),
                      y.max()-y.min(),
                      z.max()-z.min()]).max() / 2.0

mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.view_init(elev=15, azim=-60,vertical_axis="y")  # 调整观察角度
plt.show()