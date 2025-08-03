import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 设置字体路径（根据你系统中文和英文字体的路径来）
times_path = "/home/ll/.fonts/TimesNewRoman/Times New Roman.ttf"  # 或 Windows: "C:/Windows/Fonts/times.ttf"
simu_path = "/home/ll/.fonts/simsun/simsun.ttc"
times_font = fm.FontProperties(fname=times_path)    # 英文 Times New Roman
simsun_font = fm.FontProperties(fname=simu_path)    # 中文 宋体
# 数据
labels = ['Background', 'LFM', 'NLFM', 'SFM', 'FSK', 'QAM']
pixel_counts = np.array([9.08459655e+10, 4.29517708e+08, 1.11131328e+09, 2.57445442e+09, 7.37024991e+08, 4.14172412e+09])
weights = np.array([0.3963154, 0.5030643, 0.4801043, 0.4614908, 0.4897608, 0.4515819])

fig, ax1 = plt.subplots(figsize=(8, 5))

# 柱状图
ax1.bar(labels, pixel_counts , color='b', alpha=0.6, label='类别像素计数')
# ax1.set_xlabel("Lable Category", fontproperties=times_font, fontsize=20)
ax1.set_xlabel("标签种类", fontproperties=simsun_font, fontsize=22)
# ax1.set_ylabel("Number of pixels", fontproperties=times_font, fontsize=20)
ax1.set_ylabel("像素数量", fontproperties=simsun_font, fontsize=22)
ax1.tick_params(axis='x',labelsize=22)
ax1.tick_params(axis='y', labelcolor='b', labelsize=22)
ax1.set_yscale("log")  # 对像素计数使用对数坐标轴，防止数值相差过大导致显示不清

# 设置 x 和 y 轴刻度标签字体
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontproperties(times_font)
    label.set_fontsize(18)

# 折线图（右侧坐标轴）
ax2 = ax1.twinx()
ax2.plot(labels, weights, color='y', marker='o', linestyle='-')
# ax2.set_ylabel("Weight coefficient", fontproperties=times_font, fontsize=20)
ax2.set_ylabel("权重系数", fontproperties=simsun_font, fontsize=22)
ax2.tick_params(axis='y', labelcolor='y',labelsize=22)

for label in ax2.get_yticklabels():
    label.set_fontproperties(times_font)
    label.set_fontsize(22)

# 标出折线图上的点
for i, (x, y) in enumerate(zip(labels, weights)):
    ax2.text(x, y, f'{y:.4f}', color='y', ha='center', va='bottom', fontproperties=times_font, fontsize=22)

# 标题
fig.tight_layout()

# 保存图片
plt.savefig("weight.png", dpi=900, bbox_inches='tight')
plt.show()
