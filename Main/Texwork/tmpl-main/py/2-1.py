import pandas as pd
import matplotlib.pyplot as plt
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

        # 转换扫码销售时间为时间格式
        category_data["扫码销售时间"] = pd.to_datetime(category_data["扫码销售时间"], format='mixed')
        # 提取日期信息（假设存在销售日期列，若没有需要从其他地方获取日期）
        if '销售日期' in category_data.columns:
            category_data["销售日期"] = pd.to_datetime(category_data["销售日期"])
            category_data["完整时间"] = category_data["销售日期"].dt.normalize() + pd.to_timedelta(
                category_data["扫码销售时间"].dt.time.astype(str))
        else:
            print("警告：数据中缺少销售日期列，无法绘图。")
            return

        # 按完整时间排序
        category_data.sort_values(by='完整时间', inplace=True)

        # 按完整时间分组计算平均利润率
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


if __name__ == "__main__":
    save_dir = r"D:\Mathematical Modeling\Main\Texwork\tmpl-main\fig"
    os.makedirs(save_dir, exist_ok=True)
    categories = ["花叶类", "辣椒类", "食用菌", "花菜类", "水生根茎类", "茄类"]
    for category in categories:
        plot_profit_ratio_by_category(df, category, save_dir, 'png')