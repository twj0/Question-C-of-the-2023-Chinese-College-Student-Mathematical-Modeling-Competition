import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import os
import sys

class Config:
    input_path = r"D:\Mathematical Modeling\Main\C题\collet_preprocessed.xlsx"
    output_dir = r"D:\Mathematical Modeling\Main\out"
    date_range = ('2023-06-24', '2023-06-30')
    filter_params = {
        'min_available_days': 3,
        'min_avg_sales': 1.5,
        'max_adj_times': 5
    }
    pso_config = {
        'n_particles': 50,
        'max_iter': 200,
        'w': 0.8,
        'c1': 1.5,
        'c2': 2.0,
        'salvage_rate': 0.3,
        'max_quantity': 100
    }

class DataProcessor:
    def __init__(self):
        self.df = self._load_data()
        self.item_stats = None

    def _load_data(self):
        """安全加载数据并验证基础结构"""
        try:
            df = pd.read_excel(
                Config.input_path,
                parse_dates=['销售日期'],
                usecols=['销售日期', '单品编码', '单品名称', '销量(千克)', '销售单价(元/千克)', '批发价格(元/千克)']
            ).rename(columns={
                '销量(千克)': '销量',
                '销售单价(元/千克)': '售价',
                '批发价格(元/千克)': '成本价'
            })
            
            # 基础数据验证
            if df.empty:
                raise ValueError("数据文件为空")
            if not {'销售日期', '单品编码', '单品名称', '销量', '售价', '成本价'}.issubset(df.columns):
                missing = {'销售日期', '单品编码', '单品名称', '销量', '售价', '成本价'} - set(df.columns)
                raise ValueError(f"缺失必要列: {missing}")
            return df
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            sys.exit(1)

    def preprocess(self):
        """增强鲁棒性的预处理流程"""
        try:
            # 1. 日期过滤
            df_filtered = self._filter_dates()
            
            # 2. 计算可售状态
            df_filtered = df_filtered.assign(是否可售=df_filtered['销量'].fillna(0) > 0)
            
            # 3. 单品统计
            self._calculate_item_stats(df_filtered)
            
            # 4. 动态阈值调整
            return self._smart_threshold_adjustment()
        except Exception as e:
            print(f"预处理失败: {str(e)}")
            self._diagnose_data_issues()
            sys.exit(1)

    def _filter_dates(self):
        """安全过滤日期范围"""
        try:
            start_date = pd.to_datetime(Config.date_range[0])
            end_date = pd.to_datetime(Config.date_range[1])
            mask = (self.df['销售日期'] >= start_date) & (self.df['销售日期'] <= end_date)
            df_filtered = self.df.loc[mask].copy()
            
            if df_filtered.empty:
                raise ValueError(f"在{Config.date_range}范围内无数据")
            return df_filtered
        except pd.errors.OutOfBoundsDatetime as e:
            print("日期格式错误: 请确保销售日期列为合法日期格式")
            sys.exit(1)

    def _calculate_item_stats(self, df):
        """计算单品统计数据"""
        self.item_stats = df.groupby('单品编码').agg(
            可售天数=('是否可售', 'sum'),
            平均销量=('销量', 'mean'),
            成本价=('成本价', 'first')
        ).reset_index()

    def _smart_threshold_adjustment(self):
        """智能阈值调整算法"""
        thresholds = [
            (d, s) 
            for d in range(Config.filter_params['min_available_days'], 0, -1)
            for s in np.arange(Config.filter_params['min_avg_sales'], 0.1, -0.5)
        ]
        
        for days, sales in thresholds:
            mask = (self.item_stats['可售天数'] >= days) & \
                  (self.item_stats['平均销量'] >= sales)
            valid_count = sum(mask)
            if valid_count >= 27:
                print(f"有效单品数: {valid_count} (条件: ≥{days}天, ≥{sales}kg)")
                return self.item_stats[mask]['单品编码'].values
        
        # 最终尝试
        final_mask = (self.item_stats['可售天数'] >= 1) & (self.item_stats['平均销量'] >= 0.1)
        final_count = sum(final_mask)
        if final_count >= 27:
            print(f"使用最低条件找到{final_count}个单品")
            return self.item_stats[final_mask]['单品编码'].values
        
        self._diagnose_data_issues()
        raise ValueError("无法找到足够有效单品")

    def _diagnose_data_issues(self):
        """生成详细数据诊断报告"""
        print("\n=== 数据诊断报告 ===")
        print("可售天数分布:")
        print(self.item_stats['可售天数'].value_counts().sort_index())
        print("\n平均销量分布:")
        print(pd.cut(self.item_stats['平均销量'], 
                   bins=[0, 0.5, 1.0, 1.5, 2.0, np.inf]).value_counts())

    def get_item_data(self, item_ids):
        """获取优化所需单品数据"""
        return {
            item_id: {
                'cost': self.item_stats[self.item_stats['单品编码'] == item_id]['成本价'].values[0],
                'name': self.df[self.df['单品编码'] == item_id]['单品名称'].values[0],  # 获取单品名称
                'historical_prices': self.df[self.df['单品编码'] == item_id]['售价'].values,
                'historical_sales': self.df[self.df['单品编码'] == item_id]['销量'].values
            }
            for item_id in item_ids
        }

class PSOOptimizer:
    def __init__(self, items, config):
        self.items = items
        self.config = config
        self.demand_models = self._train_models()
        self.item_list = list(items.keys())  # 维护有序列表

    def _train_models(self):
        """训练需求预测模型"""
        models = {}
        for item_id, data in self.items.items():
            try:
                if len(data['historical_prices']) < 3:
                    continue
                X = np.array(data['historical_prices']).reshape(-1, 1)
                y = np.array(data['historical_sales'])
                models[item_id] = LinearRegression().fit(X, y)
            except Exception as e:
                print(f"单品{item_id}建模失败: {str(e)}")
        return models

    def _init_particle(self):
        """生成符合约束的粒子"""
        selected = random.sample(self.item_list, random.randint(27, 33))
        return {
            'selection': set(selected),
            'prices': {item: self._gen_price(item) for item in selected},
            'quantities': {item: self._gen_quantity() for item in selected}
        }

    def _gen_price(self, item_id):
        """生成合理售价"""
        cost = self.items[item_id]['cost']
        return round(random.uniform(cost*1.1, cost*1.5), 2)

    def _gen_quantity(self):
        """生成合理补货量"""
        return round(random.uniform(2.5, self.config['max_quantity']), 1)

    def _calculate_profit(self, particle):
        """计算粒子适应度"""
        total = 0
        for item in particle['selection']:
            data = self.items[item]
            price = particle['prices'][item]
            quantity = particle['quantities'][item]
            
            try:
                demand = max(self.demand_models[item].predict([[price]])[0], 0)
            except KeyError:
                continue
                
            sold = min(demand, quantity)
            total += (price * sold + 
                     self.config['salvage_rate'] * data['cost'] * (quantity - sold) -
                     data['cost'] * quantity)
        return total

    def optimize(self):
        """执行优化流程"""
        particles = [self._init_particle() for _ in range(self.config['n_particles'])]
        global_best = {'fitness': -float('inf')}
        
        progress = tqdm(range(self.config['max_iter']), desc="优化进度")
        for _ in progress:
            # 评估粒子
            for p in particles:
                current = self._calculate_profit(p)
                if current > p.get('best', -float('inf')):
                    p['best'] = current
                    p['best_state'] = {
                        'selection': set(p['selection']),
                        'prices': dict(p['prices']),
                        'quantities': dict(p['quantities'])
                    }
                
                if current > global_best['fitness']:
                    global_best.update({
                        'fitness': current,
                        'selection': set(p['selection']),
                        'prices': dict(p['prices']),
                        'quantities': dict(p['quantities'])
                    })
            
            # 更新粒子
            for p in particles:
                self._update_particle(p, global_best)
            
            progress.set_postfix({"最佳利润": f"{global_best['fitness']:.2f}元"})
        
        return global_best

    def _update_particle(self, particle, global_best):
        """粒子更新策略"""
        # 向全局最优学习
        for item in global_best['selection']:
            if item not in particle['selection'] and len(particle['selection']) < 33:
                particle['selection'].add(item)
                particle['prices'][item] = global_best['prices'][item]
                particle['quantities'][item] = global_best['quantities'][item]
        
        # 移除表现差的单品
        if len(particle['selection']) > 27:
            candidates = list(particle['selection'] - global_best['selection'])
            if candidates:
                remove_item = random.choice(candidates)
                particle['selection'].remove(remove_item)
                del particle['prices'][remove_item]
                del particle['quantities'][remove_item]

def main():
    try:
        # 数据准备
        processor = DataProcessor()
        valid_items = processor.preprocess()
        items = processor.get_item_data(valid_items)
        
        # 执行优化
        optimizer = PSOOptimizer(items, Config.pso_config)
        result = optimizer.optimize()
        
        # 生成结果报告
        report = [{
            '单品编码': item,
            '单品名称': items[item]['name'],  # 添加单品名称
            '成本价': items[item]['cost'],
            '建议售价': result['prices'][item],
            '建议补货量': result['quantities'][item],
            '预测利润': round((result['prices'][item] - items[item]['cost']) * 
                         min(optimizer.demand_models[item].predict([[result['prices'][item]]])[0],
                             result['quantities'][item]), 2)
        } for item in result['selection']]
        
        # 保存结果
        os.makedirs(Config.output_dir, exist_ok=True)
        pd.DataFrame(report).to_excel(
            os.path.join(Config.output_dir, "补货策略优化结果.xlsx"), 
            index=False
        )
        print(f"优化结果已保存至: {os.path.join(Config.output_dir, '补货策略优化结果.xlsx')}")
        
    except Exception as e:
        print(f"运行失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()