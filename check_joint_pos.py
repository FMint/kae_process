import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 触发 3D 支持
import numpy as np

# kae_pos=r"F:\fsy\project\HumanML3D\joints\000001.npy"
kae_pos=r"F:\fsy\project\kae_process\kae_pos3d\F01A1V1.npy"
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
ax.view_init(elev=15, azim=-60)  # 调整观察角度
plt.show()