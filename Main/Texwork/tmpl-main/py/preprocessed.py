import pandas as pd

# ====================== 数据加载与基础信息检查 ======================
# 定义文件路径（使用原始字符串避免转义）
input_file = r"D:\Mathematical Modeling\Main\C题\collet.xlsx"
output_dir = r"D:\Mathematical Modeling\Main\C题"

# 读取Excel数据
df = pd.read_excel(input_file)

# 数据基础信息记录（可用于日志或报告）
print(f"原始数据形状：{df.shape}")
print("数据缺失值统计：")
print(df.isnull().sum())  # 检查各列缺失值


# ====================== 缺失值处理 ======================
# 剔除含有缺失值的行（与论文描述一致：发现缺失值则剔除）
df_clean = df.dropna(how='any', axis=0)  # axis=0表示按行删除，how='any'只要有缺失就删除
print(f"剔除缺失值后数据量：{df_clean.shape[0]}条")


# ====================== 异常值处理 - 销量负数 ======================
# 识别销量为负数的记录（视为退货事件，论文明确说明）
negative_sales_count = df_clean[df_clean['销量(千克)'] < 0].shape[0]
if negative_sales_count > 0:
    print(f"检测到{negative_sales_count}条退货记录（销量为负数）")
    # 这里根据业务需求，退货记录是否保留？论文未说明剔除，仅识别说明，故保留但标记
    df_clean['是否退货'] = df_clean['销量(千克)'].apply(lambda x: 1 if x < 0 else 0)
else:
    print("未检测到销量为负数的退货记录")


# ====================== 异常值处理 - 批发价格（3σ标准差原则） ======================
def handle_outliers_3sigma(column_data):
    """
    3σ原则处理异常值并返回处理后数据
    参数：column_data - 目标数据列（Series）
    返回：处理后的数据列（异常值替换为中位数）
    """
    mean = column_data.mean()
    std = column_data.std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    
    # 计算中位数
    median = column_data.median()
    
    # 替换异常值
    processed_data = column_data.apply(
        lambda x: median if (x < lower_bound or x > upper_bound) else x
    )
    return processed_data, lower_bound, upper_bound, median

# 对批发价格列进行异常值处理
df_clean['批发价格(元/千克)'], price_lower, price_upper, price_median = \
    handle_outliers_3sigma(df_clean['批发价格(元/千克)'])

print(f"批发价格异常值处理参数：")
print(f"均值：{price_lower:.4f}，标准差：{price_upper:.4f}")
print(f"异常值阈值范围：[{price_lower:.4f}, {price_upper:.4f}]")
print(f"中位数替换值：{price_median:.4f}")


# ====================== 数据质量校验 ======================
# 校验处理后的数据是否存在新的缺失值（严谨性检查）
post_missing = df_clean.isnull().sum().sum()
if post_missing > 0:
    raise ValueError(f"处理后数据仍存在{post_missing}个缺失值，需检查处理逻辑")


# ====================== 结果输出 ======================
output_file = f"{output_dir}/collet_preprocessed.xlsx"
df_clean.to_excel(output_file, index=False)
print(f"数据预处理完成，结果已保存至：{output_file}")
print(f"最终数据形状：{df_clean.shape}")