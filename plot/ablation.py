import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 设置字体路径（根据你系统中文和英文字体的路径来）
times_path = "/home/ll/.fonts/TimesNewRoman/Times New Roman.ttf"  # 或 Windows: "C:/Windows/Fonts/times.ttf"

times_font = fm.FontProperties(fname=times_path)    # 英文 Times New Roman

# 数据
snr = [-5, 0, 5, 10, 15, 20]

accuracy1 = {
    'N = 1': [46.54, 71.03, 77.38, 80.28, 82.17, 83.25],
    'N = 2': [48.94, 72.06, 78.09, 80.84, 82.86, 83.80],
    'N = 3': [49.60, 72.68, 78.35, 81.20, 83.04, 83.98],
    'N = 4': [48.11, 71.20, 77.45, 80.17, 82.35, 83.33],
}

accuracy2 = {
    'Backbone': [49.60, 72.68, 78.35, 81.20, 83.04, 83.98],
    'Backbone+MCFM': [49.98, 73.21, 78.95, 81.48, 83.33, 84.38],
    'Backbone+CAM': [49.72, 72.68, 78.58, 81.32, 83.20, 84.12],
    'Backbone+MCFM+CAM': [50.39, 73.43, 79.11, 81.72, 83.50, 84.49],
}

# 创建画布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 设置字体大小
font_title = 22
font_label = 22
iner_label = 22
font_legend = 22
font_ticks = 22

# 绘制第一个图
ax1 = axes[0]
for label, acc in accuracy1.items():
    ax1.plot(snr, acc, marker='o', label=label)
ax1.set_xlabel("SNR (dB)", fontproperties=times_font, fontsize=font_label)
ax1.set_ylabel("mIoU (%)", fontproperties=times_font, fontsize=font_label)
ax1.tick_params(axis='both', labelsize=font_ticks)

ax1.set_xticks(snr)
ax1.set_xticklabels([str(x) for x in snr], fontproperties=times_font, fontsize=font_ticks)
ax1.set_yticklabels([f"{y:.0f}" for y in ax1.get_yticks()], fontproperties=times_font, fontsize=font_ticks)
ax1.legend(loc='upper left', prop=times_font, fontsize=font_legend)
ax1.grid(False)
ax1.text(0.5, -0.2, "(a)", fontproperties=times_font, fontsize=font_title, ha='center', va='top', transform=ax1.transAxes)

# 第二个子图
ax2 = axes[1]
for label, acc in accuracy2.items():
    ax2.plot(snr, acc, marker='s', label=label)
ax2.set_xlabel("SNR (dB)", fontproperties=times_font, fontsize=font_label)
ax2.set_ylabel("mIoU (%)", fontproperties=times_font, fontsize=font_label)
ax2.tick_params(axis='both', labelsize=font_ticks)
ax2.set_xticks(snr)
ax2.set_xticklabels([str(x) for x in snr], fontproperties=times_font, fontsize=font_ticks)
ax2.set_yticklabels([f"{y:.0f}" for y in ax2.get_yticks()], fontproperties=times_font, fontsize=font_ticks)
ax2.legend(loc='upper left', prop=times_font, fontsize=font_legend)
ax2.grid(False)
ax2.text(0.5, -0.2, "(b)", fontproperties=times_font, fontsize=font_title, ha='center', va='top', transform=ax2.transAxes)

# 放大图部分
zoom_in_snr = snr[-3:]
zoom_in_acc1 = [accuracy1['N = 1'][-3:], accuracy1['N = 2'][-3:], accuracy1['N = 3'][-3:], accuracy1['N = 4'][-3:]]
zoom_in_acc2 = [accuracy2['Backbone'][-3:], accuracy2['Backbone+MCFM'][-3:], accuracy2['Backbone+CAM'][-3:], accuracy2['Backbone+MCFM+CAM'][-3:]]

axins1 = inset_axes(ax1, width="50%", height="40%", loc="center right")
for i, label in enumerate(accuracy1.keys()):
    axins1.plot(zoom_in_snr, zoom_in_acc1[i], marker='o', label=label)
axins1.set_xticks(zoom_in_snr)
axins1.set_xticklabels([str(x) for x in zoom_in_snr], fontproperties=times_font, fontsize=font_ticks)
axins1.set_yticklabels([f"{y:.0f}" for y in axins1.get_yticks()], fontproperties=times_font, fontsize=font_ticks)
axins1.grid(True)

axins2 = inset_axes(ax2, width="50%", height="55%", loc="center right")
for i, label in enumerate(accuracy2.keys()):
    axins2.plot(zoom_in_snr, zoom_in_acc2[i], marker='s', label=label)
axins2.set_xticks(zoom_in_snr)
axins2.set_xticklabels([str(x) for x in zoom_in_snr], fontproperties=times_font, fontsize=font_ticks)
axins2.set_yticklabels([f"{y:.0f}" for y in axins2.get_yticks()], fontproperties=times_font, fontsize=font_ticks)
axins2.grid(True)

# 整体调整
plt.tight_layout()
plt.savefig("ablation_study.png", dpi=1200, bbox_inches='tight')
plt.show()