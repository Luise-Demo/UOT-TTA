import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 提取原图的精确4×4数据
data = np.array([
    [82.4, 4.8, 6.0, 6.8],
    [5.9, 84.3, 5.6, 4.2],
    [4.9, 5.7, 86.6, 2.8],
    [6.8, 7.3, 5.7, 80.2]
])

# 全局绘图参数设置，保证高清显示
plt.rcParams['figure.dpi'] = 300  # 高分辨率
plt.rcParams['font.sans-serif'] = ['Arial']  # 清晰无锯齿的字体
plt.rcParams['font.size'] = 12

# 创建正方形画布，匹配原图比例
fig, ax = plt.subplots(figsize=(4, 4))


# 绘制热力图，完全对齐原图风格
sns.heatmap(
    data,
    annot=True,        # 单元格内显示数值
    fmt='.1f',         # 保留1位小数，和原图完全一致
    cmap='Blues',      # 蓝色渐变配色，匹配原图色阶
    cbar=True,         # 显示右侧颜色条
    vmin=0,            # 色阶最小值，贴合原图范围
    vmax=80,           # 色阶最大值，覆盖数据最高值77.8
    square=True,       # 强制单元格为正方形
    linewidths=0.5,    # 单元格细分隔线
    ax=ax
)

# 隐藏坐标轴刻度标签，和原图无标签的样式一致
ax.set_xticklabels([])
ax.set_yticklabels([])

# 自动调整布局，避免内容截断
plt.tight_layout()
# 保存高清图片到当前目录
plt.savefig('clear_heatmap.png', bbox_inches='tight')
# 弹出窗口显示图片
plt.show()