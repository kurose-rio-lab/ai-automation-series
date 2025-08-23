"""
安全在庫レベル自動計算システム
統計的手法による最適な安全在庫レベルとリオーダーポイントの計算
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SafetyStockCalculator:
    def __init__(self, config_path="config/config.yaml"):
        """安全在庫計算クラスの初期化"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 設定値の取得
        self.service_level = self.config['model_settings']['inventory_optimization']['service_level']
        self.review_period = self.config['model_settings']['inventory_optimization']['review_period']
        
    def load_demand_data(self, file_path):
        """需要データの読み込み"""
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"需要データ読み込み完了: {len(df)}件")
            return df
        except Exception as e:
            print(f"需要データ読み込みエラー: {e}")
            return None
    
    def analyze_demand_patterns(self, df):
        """需要パターンの分析"""
        print("需要パターンを分析中...")
        
        # 商品別需要統計の計算
        demand_stats = df.groupby('product_id').agg({
            'quantity': ['mean', 'std', 'count', 'min', 'max']
        }).round(4)
        
        demand_stats.columns = ['avg_demand', 'demand_std', 'data_points', 'min_demand', 'max_demand']
        demand_stats = demand_stats.reset_index()
        
        # 変動係数の計算（需要の変動性指標）
        demand_stats['cv'] = demand_stats['demand_std'] / demand_stats['avg_demand']
        demand_stats['cv'] = demand_stats['cv'].fillna(0)
        
        # 需要分類（変動性による）
        demand_stats['demand_category'] = 'stable'
        demand_stats.loc[demand_stats['cv'] > 0.5, 'demand_category'] = 'variable'
        demand_stats.loc[demand_stats['cv'] > 1.0, 'demand_category'] = 'highly_variable'
        
        print(f"需要分析完了: {len(demand_stats)}商品")
        return demand_stats
    
    def calculate_lead_time_demand(self, df, lead_times_df):
        """リードタイム需要の計算"""
        print("リードタイム需要を計算中...")
        
        results = []
        
        for _, product_row in lead_times_df.iterrows():
            product_id = product_row['product_id']
            lead_time = product_row['lead_time_days']
            
            # 該当商品の需要データ
            product_demand = df[df['product_id'] == product_id].copy()
            
            if len(product_demand) < 30:  # 最低30日のデータが必要
                continue
            
            # 日次需要の準備
            product_demand = product_demand.set_index('date').resample('D')['quantity'].sum().fillna(0)
            
            # リードタイム期間の需要計算
            lead_time_demands = []
            for i in range(len(product_demand) - lead_time):
                lt_demand = product_demand.iloc[i:i+lead_time].sum()
                lead_time_demands.append(lt_demand)
            
            if len(lead_time_demands) < 10:
                continue
            
            # 統計値計算
            avg_lt_demand = np.mean(lead_time_demands)
            std_lt_demand = np.std(lead_time_demands)
            
            results.append({
                'product_id': product_id,
                'lead_time_days': lead_time,
                'avg_lt_demand': avg_lt_demand,
                'std_lt_demand': std_lt_demand,
                'data_points': len(lead_time_demands)
            })
        
        return pd.DataFrame(results)
    
    def calculate_safety_stock(self, demand_stats, lead_time_demand):
        """安全在庫の計算"""
        print("安全在庫を計算中...")
        
        # データをマージ
        safety_stock_data = pd.merge(demand_stats, lead_time_demand, on='product_id', how='inner')
        
        # Z値の計算（正規分布の場合）
        z_score = stats.norm.ppf(self.service_level)
        
        # 安全在庫計算の複数手法
        
        # 1. 基本的な安全在庫計算
        safety_stock_data['safety_stock_basic'] = (
            z_score * safety_stock_data['std_lt_demand']
        )
        
        # 2. 需要とリードタイムの変動を考慮した安全在庫
        # SS = Z × √(LT × σ_D² + D² × σ_LT²)
        # リードタイム変動は10%と仮定
        lt_variance = (safety_stock_data['lead_time_days'] * 0.1) ** 2
        
        safety_stock_data['safety_stock_advanced'] = z_score * np.sqrt(
            safety_stock_data['lead_time_days'] * safety_stock_data['demand_std'] ** 2 +
            safety_stock_data['avg_demand'] ** 2 * lt_variance
        )
        
        # 3. 変動係数を考慮した安全在庫
        # 高変動商品には追加の安全マージンを適用
        cv_multiplier = 1 + safety_stock_data['cv'].clip(0, 2)
        safety_stock_data['safety_stock_cv_adjusted'] = (
            safety_stock_data['safety_stock_basic'] * cv_multiplier
        )
        
        # 最終的な安全在庫レベル（3つの手法の最大値を採用）
        safety_stock_data['recommended_safety_stock'] = safety_stock_data[[
            'safety_stock_basic', 'safety_stock_advanced', 'safety_stock_cv_adjusted'
        ]].max(axis=1)
        
        # リオーダーポイントの計算
        safety_stock_data['reorder_point'] = (
            safety_stock_data['avg_lt_demand'] + safety_stock_data['recommended_safety_stock']
        )
        
        # 最大在庫レベル（経済的発注量 + 安全在庫）
        # 簡易EOQ計算（実際の運用では詳細なコスト分析が必要）
        annual_demand = safety_stock_data['avg_demand'] * 365
        ordering_cost = 5000  # 1回の発注コスト（円）
        holding_cost_rate = 0.2  # 年間在庫保管コスト率（20%）
        unit_cost = 1000  # 商品単価（円）- 実際は商品マスタから取得
        
        eoq = np.sqrt(2 * annual_demand * ordering_cost / (holding_cost_rate * unit_cost))
        safety_stock_data['economic_order_quantity'] = eoq
        safety_stock_data['max_stock_level'] = (
            safety_stock_data['recommended_safety_stock'] + eoq
        )
        
        # 整数に丸める
        integer_columns = [
            'recommended_safety_stock', 'reorder_point', 
            'economic_order_quantity', 'max_stock_level'
        ]
        for col in integer_columns:
            safety_stock_data[col] = safety_stock_data[col].round().astype(int)
        
        return safety_stock_data
    
    def generate_inventory_report(self, safety_stock_data):
        """在庫管理レポートの生成"""
        print("\n" + "="*60)
        print("安全在庫計算レポート")
        print("="*60)
        
        print(f"サービスレベル: {self.service_level:.1%}")
        print(f"対象商品数: {len(safety_stock_data)}商品")
        
        # カテゴリ別集計
        category_stats = safety_stock_data.groupby('demand_category').agg({
            'product_id': 'count',
            'recommended_safety_stock': 'mean',
            'reorder_point': 'mean',
            'cv': 'mean'
        }).round(2)
        
        print("\n需要カテゴリ別統計:")
        print(category_stats)
        
        # 全体統計
        print(f"\n全体統計:")
        print(f"平均安全在庫: {safety_stock_data['recommended_safety_stock'].mean():.1f}個")
        print(f"平均リオーダーポイント: {safety_stock_data['reorder_point'].mean():.1f}個")
        print(f"平均経済的発注量: {safety_stock_data['economic_order_quantity'].mean():.1f}個")
        
        # 高変動商品の特定
        high_var_products = safety_stock_data[safety_stock_data['cv'] > 1.0]
        print(f"\n高変動商品数: {len(high_var_products)}商品 ({len(high_var_products)/len(safety_stock_data)*100:.1f}%)")
        
        return category_stats
    
    def create_inventory_visualization(self, safety_stock_data):
        """在庫管理の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 安全在庫分布
        ax1 = axes[0, 0]
        ax1.hist(safety_stock_data['recommended_safety_stock'], bins=30, alpha=0.7, color='skyblue')
        ax1.set_title('安全在庫レベル分布')
        ax1.set_xlabel('安全在庫数')
        ax1.set_ylabel('商品数')
        
        # 2. 変動係数vs安全在庫
        ax2 = axes[0, 1]
        scatter = ax2.scatter(safety_stock_data['cv'], safety_stock_data['recommended_safety_stock'], 
                            alpha=0.6, c=safety_stock_data['avg_demand'], cmap='viridis')
        ax2.set_title('需要変動性vs安全在庫')
        ax2.set_xlabel('変動係数 (CV)')
        ax2.set_ylabel('安全在庫数')
        plt.colorbar(scatter, ax=ax2, label='平均需要')
        
        # 3. カテゴリ別安全在庫
        ax3 = axes[0, 2]
        category_means = safety_stock_data.groupby('demand_category')['recommended_safety_stock'].mean()
        colors = ['green', 'orange', 'red']
        bars = ax3.bar(category_means.index, category_means.values, color=colors, alpha=0.7)
        ax3.set_title('需要カテゴリ別平均安全在庫')
        ax3.set_ylabel('平均安全在庫数')
        
        # 数値ラベル追加
        for bar, value in zip(bars, category_means.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 4. リードタイムvs安全在庫
        ax4 = axes[1, 0]
        ax4.scatter(safety_stock_data['lead_time_days'], safety_stock_data['recommended_safety_stock'], 
                   alpha=0.6, color='coral')
        ax4.set_title('リードタイムvs安全在庫')
        ax4.set_xlabel('リードタイム（日）')
        ax4.set_ylabel('安全在庫数')
        
        # 5. EOQvs安全在庫の関係
        ax5 = axes[1, 1]
        ax5.scatter(safety_stock_data['economic_order_quantity'], 
                   safety_stock_data['recommended_safety_stock'], alpha=0.6, color='lightgreen')
        ax5.set_title('経済的発注量vs安全在庫')
        ax5.set_xlabel('経済的発注量')
        ax5.set_ylabel('安全在庫数')
        
        # 6. 在庫投資額分析
        ax6 = axes[1, 2]
        unit_cost = 1000  # 仮の単価
        inventory_value = safety_stock_data['recommended_safety_stock'] * unit_cost / 1000  # 千円単位
        ax6.hist(inventory_value, bins=30, alpha=0.7, color='gold')
        ax6.set_title('安全在庫投資額分布')
        ax6.set_xlabel('投資額（千円）')
        ax6.set_ylabel('商品数')
        
        plt.tight_layout()
        
        # 保存
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "safety_stock_analysis.png", dpi=300, bbox_inches='tight')
        print(f"\n可視化結果を保存: {output_dir / 'safety_stock_analysis.png'}")
        
        return fig
    
    def save_safety_stock_results(self, safety_stock_data):
        """安全在庫計算結果の保存"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        # 詳細結果の保存
        detailed_file = output_dir / "safety_stock_detailed.csv"
        safety_stock_data.to_csv(detailed_file, index=False, encoding='utf-8-sig')
        
        # 管理用サマリーの作成
        summary_columns = [
            'product_id', 'demand_category', 'avg_demand', 'cv',
            'lead_time_days', 'recommended_safety_stock', 'reorder_point',
            'economic_order_quantity', 'max_stock_level'
        ]
        
        summary_data = safety_stock_data[summary_columns].copy()
        summary_file = output_dir / "safety_stock_recommendations.csv"
        summary_data.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        # 発注管理用ファイル
        order_management = safety_stock_data[['product_id', 'reorder_point', 'economic_order_quantity']].copy()
        order_management['current_stock'] = 0  # 実際の運用では現在庫を設定
        order_management['order_needed'] = False
        
        order_file = output_dir / "order_management.csv"
        order_management.to_csv(order_file, index=False, encoding='utf-8-sig')
        
        print(f"安全在庫計算結果を保存:")
        print(f"  詳細結果: {detailed_file}")
        print(f"  推奨値サマリー: {summary_file}")
        print(f"  発注管理用: {order_file}")
        
        return detailed_file, summary_file, order_file

def create_sample_demand_data():
    """サンプル需要データの生成"""
    np.random.seed(42)
    
    # 商品リスト（ABC分析結果から一部を使用）
    products = [f"PROD_{i:04d}" for i in range(1, 101)]  # 100商品
    
    # 日付範囲（過去1年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    demand_data = []
    
    for product_id in products:
        # 商品ごとの需要特性を設定
        base_demand = np.random.uniform(10, 100)
        seasonality = np.random.uniform(0.8, 1.2)
        trend = np.random.uniform(-0.1, 0.1)
        noise_level = np.random.uniform(0.1, 0.5)
        
        for i, date in enumerate(date_range):
            # 季節性とトレンド
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)
            trend_factor = 1 + trend * i / 365
            
            # 曜日効果
            weekday_effect = 1.2 if date.weekday() < 5 else 0.8
            
            # 基本需要の計算
            expected_demand = base_demand * seasonal_factor * trend_factor * weekday_effect
            
            # ノイズ追加
            actual_demand = max(0, np.random.normal(expected_demand, expected_demand * noise_level))
            
            demand_data.append({
                'date': date,
                'product_id': product_id,
                'quantity': int(actual_demand)
            })
    
    # DataFrame作成
    df = pd.DataFrame(demand_data)
    
    # 保存
    output_dir = Path("data/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "demand_data.csv", index=False, encoding='utf-8-sig')
    print(f"サンプル需要データを生成: {output_dir / 'demand_data.csv'}")
    
    return df

def create_sample_lead_times():
    """サンプルリードタイムデータの生成"""
    np.random.seed(42)
    
    products = [f"PROD_{i:04d}" for i in range(1, 101)]
    
    lead_time_data = []
    for product_id in products:
        # リードタイムは1-14日の範囲で設定
        lead_time = np.random.choice([1, 3, 5, 7, 10, 14], p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
        
        lead_time_data.append({
            'product_id': product_id,
            'lead_time_days': lead_time,
            'supplier_id': f"SUP_{np.random.randint(1, 21):02d}"
        })
    
    df = pd.DataFrame(lead_time_data)
    
    # 保存
    output_dir = Path("data/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "lead_times.csv", index=False, encoding='utf-8-sig')
    print(f"サンプルリードタイムデータを生成: {output_dir / 'lead_times.csv'}")
    
    return df

def main():
    """メイン実行関数"""
    print("安全在庫計算システムを開始します...")
    
    # サンプルデータ生成
    if not Path("data/sample_data/demand_data.csv").exists():
        print("サンプルデータを生成中...")
        create_sample_demand_data()
        create_sample_lead_times()
    
    # 計算器の初期化
    calculator = SafetyStockCalculator()
    
    # データ読み込み
    demand_df = calculator.load_demand_data("data/sample_data/demand_data.csv")
    lead_times_df = pd.read_csv("data/sample_data/lead_times.csv")
    
    if demand_df is None:
        print("需要データの読み込みに失敗しました。")
        return
    
    # 需要パターン分析
    demand_stats = calculator.analyze_demand_patterns(demand_df)
    
    # リードタイム需要計算
    lead_time_demand = calculator.calculate_lead_time_demand(demand_df, lead_times_df)
    
    # 安全在庫計算
    safety_stock_results = calculator.calculate_safety_stock(demand_stats, lead_time_demand)
    
    # レポート生成
    calculator.generate_inventory_report(safety_stock_results)
    
    # 可視化
    calculator.create_inventory_visualization(safety_stock_results)
    
    # 結果保存
    calculator.save_safety_stock_results(safety_stock_results)
    
    print("\n✅ 安全在庫計算が完了しました！")
    print("📊 results/ フォルダに計算結果が保存されています。")
    print("📋 order_management.csv を使用して発注管理を行えます。")

if __name__ == "__main__":
    main()
