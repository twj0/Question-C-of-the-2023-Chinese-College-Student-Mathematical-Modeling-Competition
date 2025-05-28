import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import numpy as np  # Added for numpy functions used in subplot 5

def generate_vegetable_visualization(vegetable_type):
    """
    生成包含利润时间趋势的完整可视化图表
    
    参数：
    vegetable_type : str - 蔬菜类别名称（如："花菜类"）
    """
    # 构建文件路径
    input_path = f"D:\\Mathematical Modeling\\Main\\out\\{vegetable_type}_优化策略.xlsx"
    output_path = f"D:\\Mathematical Modeling\\Main\\Texwork\\tmpl-main\\fig\\{vegetable_type}_可视化.jpg"
    
    try:
        # 读取并预处理数据
        df = pd.read_excel(input_path, sheet_name='Sheet1').iloc[:-1]  # 删除最后一行
        df['日期'] = pd.to_datetime(df['日期'])
        df['当日利润(元)'] = df['当日利润(元)'].astype(float)  # 确保数值类型
    except Exception as e:
        print(f"数据读取失败[{vegetable_type}]: {str(e)}")
        return

    # 初始化画布
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(16, 14), dpi=100)
    fig.suptitle(f"{vegetable_type} 经营分析", y=0.98, fontsize=16)

    # ----------------- 子图1：价格趋势 -----------------
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(df['日期'], df['成本价(元/千克)'], marker='o', label='成本价')
    ax1.plot(df['日期'], df['建议售价(元/千克)'], marker='s', label='建议售价')
    ax1.set_title('价格趋势分析')
    ax1.set_ylabel('元/千克')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ----------------- 子图2：利润率与销量 -----------------
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(df['日期'], df['建议利润率']*100, color='g', marker='o', label='利润率')
    ax2.set_ylabel('利润率(%)', color='g')
    ax2.tick_params(axis='y', colors='g')
    ax2b = ax2.twinx()
    ax2b.bar(df['日期'], df['预测销量(kg)'], alpha=0.3, color='b', label='销量')
    ax2b.set_ylabel('销量(kg)', color='b')
    ax2.set_title('利润率与销量关系')
    ax2.tick_params(axis='x', rotation=45)

    # ----------------- 子图3：补货量与残值 -----------------
    ax3 = plt.subplot(3, 2, 3)
    width = 0.4
    x = range(len(df))
    ax3.bar(x, df['建议补货量(kg)'], width, label='补货量')
    ax3.bar([i+width for i in x], df['残值回收(元)'], width, label='残值')
    ax3.set_xticks([i + width/2 for i in x])
    ax3.set_xticklabels(df['日期'].dt.strftime('%m-%d'), rotation=45)
    ax3.set_title('补货量与残值回收')
    ax3.legend()

    # ----------------- 子图4：当日利润柱状图 -----------------
    ax4 = plt.subplot(3, 2, 4)
    bars = ax4.bar(df['日期'], df['当日利润(元)'], color='#2ca02c')
    ax4.set_title('当日利润分析')
    ax4.set_ylabel('利润(元)')
    ax4.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height,
                 f'{height:.0f}', ha='center', va='bottom')

   # ... existing code ...

    # ----------------- 优化后的子图5：三维关系图 -----------------
    ax5 = plt.subplot(3, 2, 5)
    
    # 优化点1：使用双色渐变区分高低利润
    cmap = plt.get_cmap('RdYlGn')  # 红-黄-绿色彩映射

    # Check and remove NaN and inf values
    mask = df[['成本价(元/千克)', '建议售价(元/千克)', '当日利润(元)']].notna().all(axis=1)
    df_clean = df[mask]
    df_clean = df_clean[~np.isinf(df_clean[['成本价(元/千克)', '建议售价(元/千克)', '当日利润(元)']]).any(axis=1)]

    # Remove duplicate points
    df_clean = df_clean.drop_duplicates(subset=['成本价(元/千克)', '建议售价(元/千克)'])

    # Add small random noise to break collinearity
    noise_scale = 1e-6
    df_clean['成本价(元/千克)'] += np.random.normal(0, noise_scale, len(df_clean))
    df_clean['建议售价(元/千克)'] += np.random.normal(0, noise_scale, len(df_clean))

    scatter = ax5.scatter(
        df_clean['成本价(元/千克)'],
        df_clean['建议售价(元/千克)'],
        c=df_clean['当日利润(元)'],
        s=df_clean['预测销量(kg)']/df_clean['预测销量(kg)'].max()*300 + 50,  # 动态尺寸计算
        cmap=cmap,
        edgecolors='w',  # 增加白色边框
        linewidths=0.5,
        alpha=0.8
    )
    
    # 优化点2：添加利润等高线
    if len(df_clean) >= 3:  # Ensure there are enough points for triangulation
        try:
            density = ax5.tricontourf(
                df_clean['成本价(元/千克)'],
                df_clean['建议售价(元/千克)'],
                df_clean['当日利润(元)'],
                levels=10,
                cmap=cmap,
                alpha=0.2
            )
        except RuntimeError as e:
            print(f"Triangulation error in {vegetable_type}: {e}")
    else:
        print(f"Not enough points for triangulation in {vegetable_type}")

    # ... existing code ...

    # ----------------- 新增子图6：利润时间趋势 -----------------
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(df['日期'], df['当日利润(元)'], 
            marker='o', linestyle='--', 
            color='#d62728', linewidth=2)
    ax6.fill_between(df['日期'], df['当日利润(元)'], 
                    alpha=0.2, color='#d62728')
    ax6.set_title('利润时间趋势分析')
    ax6.set_ylabel('利润(元)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # 设置统一日期格式
    for ax in [ax1, ax2, ax4, ax6]:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # 保存并关闭
    try:
        plt.tight_layout(pad=3, h_pad=2, w_pad=2)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"成功生成：{vegetable_type}")
    except Exception as e:
        print(f"保存失败[{vegetable_type}]: {str(e)}")
        plt.close()

# 执行所有类别处理
vegetable_types = ["花菜类", "花叶类", "辣椒类", 
                  "茄类", "食用菌", "水生根茎类"]

for veg_type in vegetable_types:
    generate_vegetable_visualization(veg_type)