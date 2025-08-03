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
snr = [-5, 0, 5, 10, 15, 20]

accuracy = {
    'MFPFNet': [50.39, 73.43, 79.11, 81.72, 83.50, 84.49],
    'Bisenet': [43.20, 67.73, 75.17, 78.34, 80.65, 81.78],
    'ResNet': [34.08, 52.95, 58.93, 60.05, 61.63, 61.67],
    'DDRNet': [24.25, 50.77, 65.13, 70.03, 73.07, 74.09]
}


# 绘制图像
plt.figure(figsize=(6, 4))

# 设置不同颜色，THMNet 为红色
colors = {
    'MFPFNet': 'red',
    'Bisenet': 'blue',
    'DDRNet': 'green',
    'ResNet': 'purple',
}

# 绘制每个模型的折线并标注数据点
for model in accuracy:
    plt.plot(snr, accuracy[model], label=model, color=colors[model], marker='o')  # marker='o' 表示每个数据点

# 设置图例
plt.legend(loc='lower right', prop=times_font, fontsize=18)  # 图例位置和字体大小

# # 设置坐标轴标签（中文）
plt.xlabel('SNR (dB)', fontproperties=times_font, fontsize=14)
plt.ylabel('mIoU (%)', fontproperties=times_font, fontsize=14)

# 调整坐标轴刻度数字大小
plt.tick_params(axis='both', labelsize=12)  # 设置横纵坐标数字的字体大小为10
plt.xticks(fontproperties=times_font, fontsize=12)  # x轴刻度
plt.yticks(fontproperties=times_font, fontsize=12)  # y轴刻度

# 显示网格
plt.grid(True)

# 保存图像到文件
plt.tight_layout()
plt.savefig('compare.png', dpi=900)

# 显示图像
plt.show()