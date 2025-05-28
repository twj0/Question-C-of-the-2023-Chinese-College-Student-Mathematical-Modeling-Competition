import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
excel_file = os.path.join("C题", "collet_preprocessed.xlsx")
df = pd.read_excel(excel_file, sheet_name="Sheet1")


def plot_profit_ratio_by_category(df, category_name, save_dir, fig_format='pdf'):
    try:
        df = df.copy()
        df['利润率'] = (df['销售单价(元/千克)'] - df['批发价格(元/千克)']) / df['销售单价(元/千克)'] * 100
        category_data = df[df["分类名称"] == category_name]
        if category_data.empty:
            print(f"警告：未找到 {category_name} 相关数据，跳过绘制。")
            return

        category_data["扫码销售时间"] = pd.to_datetime(category_data["扫码销售时间"], format='mixed')
        if '销售日期' in category_data.columns:
            category_data["销售日期"] = pd.to_datetime(category_data["销售日期"])
            category_data["完整时间"] = category_data["销售日期"].dt.normalize() + pd.to_timedelta(
                category_data["扫码销售时间"].dt.time.astype(str))
        else:
            print("警告：数据中缺少销售日期列，无法绘图。")
            return

        category_data.sort_values(by='完整时间', inplace=True)
        time_series = category_data.groupby('完整时间')['利润率'].mean()

        plt.figure(figsize=(12, 6))
        time_series.plot(kind='line', marker='o', markersize=4, linestyle='-', linewidth=1, alpha=0.7)
        plt.title(f"{category_name} 随时间的利润率变化")
        plt.xlabel("时间")
        plt.xticks(rotation=45)
        plt.ylabel("利润率 (%)")
        plt.grid(True, linestyle='--', alpha=0.5)

        filename = f"{category_name.replace('类', '')}_profit_ratio_over_time.{fig_format}"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"图表已保存至：{save_path}")
        plt.close()

    except KeyError as e:
        print(f"数据列缺失错误：{str(e)}")
    except Exception as e:
        print(f"绘制 {category_name} 图表时出现未预期错误：{str(e)}")


def analyze_sales_profit_relationship(df, save_dir):
    # 计算每个品类的销售总量和总利润
    df['利润'] = (df['销售单价(元/千克)'] - df['批发价格(元/千克)']) * df['销量(千克)']
    category_summary = df.groupby('分类名称').agg({
        '销量(千克)': 'sum',
        '利润': 'sum'
    }).reset_index()

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=category_summary, x='销量(千克)', y='利润')
    plt.title('各蔬菜品类销售总量与利润的关系')
    plt.xlabel('销售总量(千克)')
    plt.ylabel('总利润')

    # 保存散点图
    scatter_filename = 'sales_profit_scatter.png'
    scatter_save_path = os.path.join(save_dir, scatter_filename)
    plt.savefig(scatter_save_path, bbox_inches='tight', dpi=300)
    print(f"销售总量与利润关系散点图已保存至：{scatter_save_path}")
    plt.close()

    # 计算相关系数
    correlation = category_summary['销量(千克)'].corr(category_summary['利润'])
    print(f"销售总量与利润的相关系数为：{correlation}")


if __name__ == "__main__":
    save_dir = r"D:\Mathematical Modeling\Main\Texwork\tmpl-main\fig"
    os.makedirs(save_dir, exist_ok=True)
    categories = ["花叶类", "辣椒类", "食用菌", "花菜类", "水生根茎类", "茄类"]
    for category in categories:
        plot_profit_ratio_by_category(df, category, save_dir, 'png')

    # 分析销售总量与利润的关系
    analyze_sales_profit_relationship(df, save_dir)