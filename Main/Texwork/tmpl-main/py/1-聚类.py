import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# 设置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """
    加载数据
    :return: 数据 DataFrame
    """
    excel_file = os.path.join("C题", "collet_preprocessed.xlsx")
    df = pd.read_excel(excel_file, sheet_name="Sheet1")
    df = df.rename(columns={"销量(千克)": "销量", "销售单价(元/千克)": "单价"})
    return df

def perform_kmeans_clustering(df, features, n_clusters=3):
    """
    执行 KMeans 聚类
    :param df: 数据 DataFrame
    :param features: 用于聚类的特征列表
    :param n_clusters: 聚类的数量，默认为 3
    :return: 带有聚类标签的 DataFrame
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[features])
    return df

def plot_kmeans_clusters(df, features, save_path):
    """
    绘制 KMeans 聚类结果
    :param df: 带有聚类标签的 DataFrame
    :param features: 用于聚类的特征列表
    :param save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(df[features[0]], df[features[1]], c=df['cluster'], cmap='viridis')
    plt.title('KMeans 聚类结果')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend(*scatter.legend_elements(), title="聚类标签")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def main():
    df = load_data()
    save_dir = r"D:\Mathematical Modeling\Main\Texwork\tmpl-main\fig"
    os.makedirs(save_dir, exist_ok=True)

    # 选择用于聚类的特征
    features = ["销量", "单价"]

    # 执行 KMeans 聚类
    df_clustered = perform_kmeans_clustering(df, features, n_clusters=3)

    # 绘制聚类结果
    plot_kmeans_clusters(df_clustered, features, os.path.join(save_dir, "kmeans_clustering.pdf"))

if __name__ == "__main__":
    main()
