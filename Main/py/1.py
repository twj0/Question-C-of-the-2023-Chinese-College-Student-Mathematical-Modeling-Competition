import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# 设置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_clean_data():
    """
    加载并清洗数据
    :return: 清洗后的 DataFrame
    """
    excel_file = os.path.join("C题", "collet_preprocessed.xlsx")
    df = pd.read_excel(excel_file, sheet_name="Sheet1")
    df = df.rename(columns={"销量(千克)": "销量", "销售单价(元/千克)": "单价"})
    return df[df["销量"] > 0]



def plot_bar(data, title, xlabel, ylabel, save_path):
    """
    绘制柱状图并保存
    :param data: 绘图数据
    :param title: 图表标题
    :param xlabel: x 轴标签
    :param ylabel: y 轴标签
    :param save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    ax = data.plot(kind='bar')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # 隐藏 x 轴标签
    ax.set_xticklabels([])
    
    # 为每个柱子添加标签连线
    for i, v in enumerate(data):
        ax.text(i, v + 0.1, data.index[i], ha='center', rotation=45, fontsize=6)
        # 绘制连线
        ax.plot([i, i], [0, v], color='gray', linestyle='--', alpha=0.5)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()



def plot_time_series(df, category_name, save_dir):
    """
    绘制特定品类销量随时间变化的曲线
    :param df: 数据 DataFrame
    :param category_name: 品类名称
    :param save_dir: 保存目录
    """
    category_data = df[df["分类名称"] == category_name]
    if category_data.empty:
        print(f"未找到 {category_name} 相关数据，跳过绘制。")
        return
    category_data["扫码销售时间"] = pd.to_datetime(category_data["扫码销售时间"])
    category_sales = category_data.groupby("扫码销售时间")["销量"].sum().sort_index()

    plt.figure(figsize=(12, 6))
    category_sales.plot(kind='line')
    plt.title(f"{category_name}销量随时间变化")
    plt.xlabel("扫码销售时间")
    plt.xticks(rotation=45)
    plt.ylabel("销量(千克)")

    file_name = f"{category_name.replace('类', '')}_sales.pdf"
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
    print(f"{category_name} 图表已保存至 {os.path.join(save_dir, file_name)}")
    plt.show()

def plot_monthly_sales(df, category_name, save_dir):
    """
    绘制特定品类在不同月份的销售柱状图
    :param df: 数据 DataFrame
    :param category_name: 品类名称
    :param save_dir: 保存目录
    """
    category_data = df[df["分类名称"] == category_name]
    if category_data.empty:
        print(f"未找到 {category_name} 相关数据，跳过绘制。")
        return
    category_data["销售日期"] = pd.to_datetime(category_data["销售日期"])
    category_data["月份"] = category_data["销售日期"].dt.month
    monthly_sales = category_data.groupby("月份")["销量"].sum()

    all_months = pd.Series(range(1, 13), name='月份')
    monthly_sales = pd.merge(all_months, monthly_sales, on='月份', how='left').fillna(0)
    monthly_sales.set_index('月份', inplace=True)

    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='bar')
    plt.title(f"{category_name} 不同月份的销售柱状图")
    plt.xlabel("月份")
    plt.ylabel("销量(千克)")

    file_name = f"{category_name.replace('类', '')}_monthly_sales.pdf"
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
    print(f"{category_name} 图表已保存至 {os.path.join(save_dir, file_name)}")
    plt.show()

def plot_correlation_heatmap(df, categories, save_dir):
    """
    计算并绘制品类销售向量相关性热力图
    :param df: 数据 DataFrame
    :param categories: 品类列表
    :param save_dir: 保存目录
    """
    sales_vectors = pd.DataFrame()
    for category in categories:
        category_data = df[df["分类名称"] == category]
        if not category_data.empty:
            category_data["扫码销售时间"] = pd.to_datetime(category_data["扫码销售时间"])
            category_sales = category_data.groupby("扫码销售时间")["销量"].sum().sort_index()
            sales_vectors[category] = category_sales

    methods = [('pearson', 'category_pearson_correlation_heatmap.pdf'),
               ('spearman', 'category_spearman_correlation_heatmap.pdf')]
    for method, file_name in methods:
        corr = sales_vectors.corr(method=method)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'不同品类销售向量 {method.capitalize()} 相关性热力图')
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
        print(f"{method.capitalize()} 相关性热力图已保存至 {os.path.join(save_dir, file_name)}")
        plt.show()

def main():
    df = load_and_clean_data()
    save_dir = r"D:\Mathematical Modeling\Main\Texwork\tmpl-main\fig"
    os.makedirs(save_dir, exist_ok=True)

    # 品类和单品销售分布分析
    category_sales = df.groupby("分类名称")["销量"].sum().sort_values(ascending=False)
    item_sales = df.groupby("单品名称")["销量"].sum().sort_values(ascending=False)
    plot_bar(category_sales, "各品类销量占比分布", "品类名称", "销量(千克)",
             os.path.join(save_dir, "category_sales.pdf"))
    plot_bar(item_sales, "各单品销量占比分布", "单品名称", "销量(千克)",
             os.path.join(save_dir, "item_sales.pdf"))

    categories = ["花叶类", "辣椒类", "食用菌", "花菜类", "水生根茎类", "茄类"]
    # 不同种类随时间的销售变化曲线
    for category in categories:
        plot_time_series(df, category, save_dir)
    # 不同种类在不同月份的销售柱状图
    for category in categories:
        plot_monthly_sales(df, category, save_dir)
    # 计算不同品类销售向量的相关性并绘制热力图
    plot_correlation_heatmap(df, categories, save_dir)

if __name__ == "__main__":
    main()