import pandas as pd
import matplotlib.pyplot as plt
import os

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_replenishment(input_path, output_folder):
    # 读取数据
    df = pd.read_excel(input_path)
    
    # 创建画布
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('补货策略优化可视化分析', fontsize=16)
    
    # 价格对比分析
    df.plot(kind='bar', x='单品名称', y=['成本价', '建议售价'], 
           ax=axs[0,0], width=0.8)
    axs[0,0].set_title('价格对比分析')
    axs[0,0].set_ylabel('金额')
    axs[0,0].tick_params(axis='x', rotation=90)
    
    # 补货量分布
    df.plot(kind='bar', x='单品名称', y='建议补货量', 
           ax=axs[0,1], color='green')
    axs[0,1].set_title('补货量分布')
    axs[0,1].set_ylabel('数量')
    axs[0,1].tick_params(axis='x', rotation=90)
    
    # 利润分布
    df.plot(kind='bar', x='单品名称', y='预测利润', 
           ax=axs[1,0], color='orange')
    axs[1,0].set_title('利润预测')
    axs[1,0].set_ylabel('利润')
    axs[1,0].tick_params(axis='x', rotation=90)
    
    # 补货量与利润关系
    df.plot(kind='scatter', x='建议补货量', y='预测利润', 
           ax=axs[1,1], color='red', s=100)
    axs[1,1].set_title('补货量-利润关系')
    axs[1,1].set_xlabel('建议补货量')
    axs[1,1].set_ylabel('预测利润')
    
    # 调整布局
    plt.tight_layout(pad=4, h_pad=3, w_pad=3)
    
    # 保存结果
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, '策略可视化.jpg')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'结果保存至：{output_path}')

# 执行示例
input_excel = r"D:\Mathematical Modeling\Main\out\补货策略优化结果.xlsx"
output_folder = r"D:\Mathematical Modeling\Main\Texwork\tmpl-main\fig"
visualize_replenishment(input_excel, output_folder)