import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pygad
import os
from datetime import datetime

# 配置参数
CONFIG = {
    "input_file": r"D:\Mathematical Modeling\Main\C题\collet_preprocessed.xlsx",  # 输入文件路径
    "output_dir": r"D:\Mathematical Modeling\Main\out",             # 输出目录
    "start_date": "2023-07-01",                                     # 预测起始日期
    "prediction_days": 7,                                            # 预测天数
    "salvage_rate": 0.3,                                            # 库存残值率(按成本价比例)
    "max_profit_margin": 0.5,                                       # 最大允许利润率
    
    "ga_settings": {                                                # 遗传算法参数
        # 可以适当调整这些参数以获得更好的结果
        "generations": 1000,                                         # 进化代数
        "population_size": 100,                                      # 种群规模
        "parent_mating_num": 50,                                     # 交配父代数量
        "mutation_rate": 0.05                                       # 变异概率
    }
}

def validate_config():
    """
    验证配置有效性
    此函数会检查输入文件是否存在，起始日期格式是否正确，同时确保输出目录存在。
    """
    if not os.path.exists(CONFIG["input_file"]):
        raise FileNotFoundError(f"输入文件不存在: {CONFIG['input_file']}")
    try:
        datetime.strptime(CONFIG["start_date"], "%Y-%m-%d")
    except ValueError:
        raise ValueError("起始日期格式应为YYYY-MM-DD")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

def load_and_preprocess():
    """
    数据加载与预处理
    该函数会从指定文件加载数据，进行数据清洗和特征工程，最后按分类聚合日数据。
    """
    print("\n➤ 正在加载数据...")
    df = pd.read_excel(
        CONFIG["input_file"],
        sheet_name="Sheet1",
        parse_dates=["销售日期"],
        usecols=["分类名称", "销售日期", "销量(千克)", 
                "销售单价(元/千克)", "批发价格(元/千克)", "损耗率(%)"]
    )
    
    # 数据清洗
    print("➤ 正在清洗数据...")
    initial_count = len(df)
    df = df.dropna(subset=["分类名称", "销量(千克)", "销售单价(元/千克)", 
                          "批发价格(元/千克)", "损耗率(%)"])
    print(f"原始记录数: {initial_count}, 有效记录数: {len(df)}")
    
    # 特征工程
    print("➤ 正在构造特征...")
    df["利润率"] = df["销售单价(元/千克)"] / df["批发价格(元/千克)"] - 1
    df["星期几"] = df["销售日期"].dt.dayofweek
    df["月份"] = df["销售日期"].dt.month
    df["是否促销"] = df["销售单价(元/千克)"] < df["批发价格(元/千克)"] * 1.1  # 假设低于10%利润为促销
    
    # 按分类聚合日数据
    agg_df = df.groupby(["分类名称", "销售日期"]).agg({
        "销量(千克)": "sum",
        "利润率": "mean",
        "批发价格(元/千克)": "first",
        "损耗率(%)": "first",
        "星期几": "first",
        "月份": "first",
        "是否促销": "max"
    }).reset_index()
    
    return agg_df

def train_sales_model(data):
    """
    训练销量预测模型
    此函数会针对数据中的每个品类训练 XGBRegressor 回归模型，并评估模型性能。
    """
    print("\n➤ 开始训练预测模型...")
    models = {}
    
    for category in data["分类名称"].unique():
        print(f"正在处理品类: {category}")
        cat_data = data[data["分类名称"] == category]
        
        # 检查数据量
        if len(cat_data) < 10:
            print(f"⚠ 数据不足({len(cat_data)}条)，跳过该品类")
            continue
        
        # 准备特征矩阵
        features = cat_data[["利润率", "星期几", "月份", "是否促销"]]
        target = cat_data["销量(千克)"]
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # 训练模型
        try:
            model = XGBRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # 评估模型
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred)) if len(y_test) > 0 else 0
            print(f"训练RMSE: {train_rmse:.2f}, 测试RMSE: {test_rmse:.2f}")
            
            models[category] = model
        except Exception as e:
            print(f"模型训练失败: {str(e)}")
    
    return models

class ProfitOptimizer:
    """
    收益优化器
    该类用于通过遗传算法优化各品类的利润率，以实现利润最大化。
    """
    def __init__(self, model, cost_price, loss_rate):
        """
        初始化收益优化器
        :param model: 销量预测模型
        :param cost_price: 成本价
        :param loss_rate: 损耗率
        """
        self.model = model
        self.cost_price = cost_price
        self.loss_rate = loss_rate / 100  # 转换为小数
        
        # 生成未来日期特征
        self.dates = pd.date_range(
            start=CONFIG["start_date"],
            periods=CONFIG["prediction_days"]
        )
        self.features = pd.DataFrame({
            "星期几": self.dates.dayofweek,
            "月份": self.dates.month,
            "是否促销": [False] * CONFIG["prediction_days"]  # 假设未来无促销
        })
    
    def _create_features(self, margins):
        """
        构造特征矩阵
        :param margins: 利润率数组
        :return: 特征矩阵
        """
        return np.column_stack((
            margins,
            self.features["星期几"],
            self.features["月份"],
            self.features["是否促销"]
        ))
    
    def _calculate_profit(self, margins):
        """
        核心利润计算
        :param margins: 利润率数组
        :return: 总利润
        """
        try:
            # 预测销量
            X = self._create_features(margins)
            sales_pred = self.model.predict(X)
            
            # 计算补货量
            replenishment = sales_pred / (1 - self.loss_rate)
            
            # 计算残值和成本
            unsold = replenishment - sales_pred
            salvage_value = CONFIG["salvage_rate"] * self.cost_price * unsold
            total_cost = self.cost_price * replenishment
            
            # 计算利润
            revenue = self.cost_price * (1 + margins) * sales_pred
            profit = np.sum(revenue + salvage_value - total_cost)
            return profit
        except:
            return -np.inf
    
    def optimize(self):
        """
        执行遗传算法优化
        :return: 最优解和最优适应度值
        """
        ga = pygad.GA(
            num_generations=CONFIG["ga_settings"]["generations"],
            num_parents_mating=CONFIG["ga_settings"]["parent_mating_num"],
            fitness_func=lambda ga, s, i: self._calculate_profit(s),
            sol_per_pop=CONFIG["ga_settings"]["population_size"],
            num_genes=CONFIG["prediction_days"],
            gene_space=[{"low":0, "high":CONFIG["max_profit_margin"]}] * CONFIG["prediction_days"],  # 正确闭合
            mutation_probability=CONFIG["ga_settings"]["mutation_rate"],
            stop_criteria=["saturate_25", "reach_10000"],
            random_seed=42,
            suppress_warnings=True
        )
        ga.run()
        solution, fitness, _ = ga.best_solution()
        return solution, fitness  # 明确返回两个值

def generate_report(solution, optimizer, category):
    """
    生成优化报告
    :param solution: 最优解（利润率数组）
    :param optimizer: 收益优化器实例
    :param category: 品类名称
    :return: 包含优化结果的 DataFrame
    """
    margins = solution
    
    # 预测各天数据
    X = optimizer._create_features(margins)
    sales_pred = optimizer.model.predict(X)
    replenishment = sales_pred / (1 - optimizer.loss_rate)
    prices = optimizer.cost_price * (1 + margins)
    
    # 计算详细财务指标
    revenue = prices * sales_pred
    cost = optimizer.cost_price * replenishment
    salvage = CONFIG["salvage_rate"] * optimizer.cost_price * (replenishment - sales_pred)
    profit = revenue + salvage - cost
    
    # 构建报告
    report = pd.DataFrame({
        "日期": optimizer.dates.strftime("%Y-%m-%d"),
        "成本价(元/千克)": optimizer.cost_price,
        "建议利润率": margins.round(4),
        "建议售价(元/千克)": prices.round(2),
        "预测销量(kg)": sales_pred.round(2),
        "建议补货量(kg)": replenishment.round(2),
        "残值回收(元)": salvage.round(2),
        "当日利润(元)": profit.round(2)
    })
    
    # 添加汇总行
    total_row = pd.Series({
        "日期": "总计",
        "当日利润(元)": report["当日利润(元)"].sum()
    }, name="total")
    return pd.concat([report, total_row.to_frame().T], ignore_index=True)

def main():
    """
    主函数，控制整个程序的执行流程
    """
    # 配置验证
    validate_config()
    
    try:
        # 数据预处理
        agg_data = load_and_preprocess()
        
        # 模型训练
        models = train_sales_model(agg_data)
        if not models:
            raise ValueError("没有成功训练任何模型，请检查数据")
        
        # 优化流程
        print("\n➤ 开始优化计算...")
        for category, model in models.items():
            print(f"\n正在优化品类: {category}")
            
            # 获取品类参数
            category_data = agg_data[agg_data["分类名称"] == category].iloc[0]
            cost_price = category_data["批发价格(元/千克)"]
            loss_rate = category_data["损耗率(%)"]
            
            # 执行优化
            optimizer = ProfitOptimizer(model, cost_price, loss_rate)
            solution, solution_fitness = optimizer.optimize()
            
            # 生成报告
            report = generate_report(solution, optimizer, category)  # 正确传递参数
            print(report)
            
            # 保存结果
            filename = f"{category.replace('/', '_')}_优化策略.xlsx"
            filepath = os.path.join(CONFIG["output_dir"], filename)
            report.to_excel(filepath, index=False)
            print(f"✅ 已保存: {filepath}")
            
    except Exception as e:
        print(f"\n❌ 程序运行出错: {str(e)}")
    finally:
        print("\n优化流程结束")

if __name__ == "__main__":
    main()