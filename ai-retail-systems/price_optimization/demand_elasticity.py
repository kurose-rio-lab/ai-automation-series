"""
需要弾力性分析と価格最適化
価格変動に対する需要の反応を分析し、最適価格を算出
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DemandElasticityAnalyzer:
    def __init__(self):
        """需要弾力性分析クラスの初期化"""
        self.elasticity_models = {}
        self.price_optimization_results = {}
        
    def load_price_sales_data(self, file_path: str):
        """価格・売上データの読み込み"""
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            
            required_columns = ['date', 'product_id', 'price', 'quantity', 'sales_amount']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"必要な列が不足: {missing_columns}")
                return None
            
            print(f"価格・売上データ読み込み完了: {len(df)}レコード")
            return df
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return None
    
    def calculate_price_elasticity(self, df, product_id):
        """特定商品の価格弾力性を計算"""
        # 商品データを抽出
        product_data = df[df['product_id'] == product_id].copy()
        
        if len(product_data) < 10:
            print(f"商品 {product_id} のデータが不足しています")
            return None
        
        # データ前処理
        product_data = product_data.sort_values('date')
        
        # 外れ値除去（IQR法）
        Q1 = product_data['quantity'].quantile(0.25)
        Q3 = product_data['quantity'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        product_data = product_data[
            (product_data['quantity'] >= lower_bound) & 
            (product_data['quantity'] <= upper_bound)
        ]
        
        # 対数変換（弾力性計算のため）
        product_data['log_price'] = np.log(product_data['price'])
        product_data['log_quantity'] = np.log(product_data['quantity'] + 1)  # +1でゼロ避け
        
        # 線形回帰による弾力性推定
        X = product_data[['log_price']]
        y = product_data['log_quantity']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 価格弾力性（回帰係数）
        elasticity = model.coef_[0]
        r2 = r2_score(y, model.predict(X))
        
        # 非線形関係も考慮（多項式回帰）
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(product_data[['price']])
        
        poly_model = Ridge(alpha=1.0)
        poly_model.fit(X_poly, product_data['quantity'])
        
        poly_r2 = r2_score(product_data['quantity'], poly_model.predict(X_poly))
        
        # 結果保存
        elasticity_result = {
            'product_id': product_id,
            'elasticity': elasticity,
            'r2_linear': r2,
            'r2_polynomial': poly_r2,
            'data_points': len(product_data),
            'price_range': {
                'min': product_data['price'].min(),
                'max': product_data['price'].max(),
                'mean': product_data['price'].mean()
            },
            'quantity_range': {
                'min': product_data['quantity'].min(),
                'max': product_data['quantity'].max(),
                'mean': product_data['quantity'].mean()
            },
            'linear_model': model,
            'polynomial_model': poly_model,
            'poly_features': poly_features
        }
        
        self.elasticity_models[product_id] = elasticity_result
        
        print(f"商品 {product_id} の価格弾力性: {elasticity:.3f} (R²: {r2:.3f})")
        
        return elasticity_result
    
    def analyze_all_products(self, df):
        """全商品の価格弾力性を分析"""
        products = df['product_id'].unique()
        elasticity_summary = []
        
        print(f"{len(products)}商品の価格弾力性を分析中...")
        
        for product_id in products:
            result = self.calculate_price_elasticity(df, product_id)
            if result:
                elasticity_summary.append({
                    'product_id': product_id,
                    'elasticity': result['elasticity'],
                    'r2_linear': result['r2_linear'],
                    'r2_polynomial': result['r2_polynomial'],
                    'data_points': result['data_points'],
                    'avg_price': result['price_range']['mean'],
                    'avg_quantity': result['quantity_range']['mean']
                })
        
        elasticity_df = pd.DataFrame(elasticity_summary)
        
        # 弾力性による分類
        elasticity_df['elasticity_category'] = 'normal'
        elasticity_df.loc[elasticity_df['elasticity'] < -1, 'elasticity_category'] = 'elastic'
        elasticity_df.loc[
            (elasticity_df['elasticity'] >= -1) & (elasticity_df['elasticity'] < 0), 
            'elasticity_category'
        ] = 'inelastic'
        elasticity_df.loc[elasticity_df['elasticity'] >= 0, 'elasticity_category'] = 'abnormal'
        
        return elasticity_df
    
    def optimize_price_single_product(self, product_id, cost_per_unit, 
                                    target_margin=0.3, price_change_limit=0.2):
        """単一商品の価格最適化"""
        if product_id not in self.elasticity_models:
            print(f"商品 {product_id} の弾力性データが見つかりません")
            return None
        
        elasticity_data = self.elasticity_models[product_id]
        current_price = elasticity_data['price_range']['mean']
        elasticity = elasticity_data['elasticity']
        
        # 価格範囲の設定
        min_price = current_price * (1 - price_change_limit)
        max_price = current_price * (1 + price_change_limit)
        min_price = max(min_price, cost_per_unit * (1 + target_margin))
        
        def profit_function(price):
            """利益関数（最大化したい）"""
            # 需要予測（弾力性モデルベース）
            price_change_ratio = price / current_price
            
            if elasticity != 0:
                quantity_change_ratio = price_change_ratio ** elasticity
            else:
                quantity_change_ratio = 1
            
            predicted_quantity = elasticity_data['quantity_range']['mean'] * quantity_change_ratio
            predicted_quantity = max(0, predicted_quantity)  # 負の需要は0
            
            # 利益計算
            profit = (price - cost_per_unit) * predicted_quantity
            
            return -profit  # 最小化問題として解くため負値
        
        # 制約条件
        constraints = [
            {'type': 'ineq', 'fun': lambda p: p - min_price},  # 最低価格制約
            {'type': 'ineq', 'fun': lambda p: max_price - p}   # 最高価格制約
        ]
        
        # 最適化実行
        result = minimize(
            profit_function,
            x0=current_price,
            method='SLSQP',
            constraints=constraints,
            options={'disp': False}
        )
        
        if result.success:
            optimal_price = result.x[0]
            
            # 最適価格での予測値計算
            price_change_ratio = optimal_price / current_price
            quantity_change_ratio = price_change_ratio ** elasticity if elasticity != 0 else 1
            predicted_quantity = elasticity_data['quantity_range']['mean'] * quantity_change_ratio
            predicted_profit = (optimal_price - cost_per_unit) * predicted_quantity
            
            # 現在価格での利益
            current_quantity = elasticity_data['quantity_range']['mean']
            current_profit = (current_price - cost_per_unit) * current_quantity
            
            optimization_result = {
                'product_id': product_id,
                'current_price': current_price,
                'optimal_price': optimal_price,
                'price_change_percent': ((optimal_price - current_price) / current_price) * 100,
                'current_quantity': current_quantity,
                'predicted_quantity': predicted_quantity,
                'quantity_change_percent': ((predicted_quantity - current_quantity) / current_quantity) * 100,
                'current_profit': current_profit,
                'predicted_profit': predicted_profit,
                'profit_improvement_percent': ((predicted_profit - current_profit) / current_profit) * 100,
                'elasticity': elasticity,
                'cost_per_unit': cost_per_unit
            }
            
            self.price_optimization_results[product_id] = optimization_result
            
            print(f"商品 {product_id} 価格最適化完了:")
            print(f"  最適価格: {optimal_price:.2f}円 ({optimization_result['price_change_percent']:+.1f}%)")
            print(f"  予測利益改善: {optimization_result['profit_improvement_percent']:+.1f}%")
            
            return optimization_result
        
        else:
            print(f"商品 {product_id} の価格最適化に失敗しました")
            return None
    
    def dynamic_pricing_strategy(self, df, cost_data, competitor_prices=None):
        """動的価格設定戦略"""
        strategies = {}
        
        for product_id in self.elasticity_models.keys():
            if product_id not in cost_data:
                continue
            
            elasticity_data = self.elasticity_models[product_id]
            cost = cost_data[product_id]
            
            # 基本戦略の決定
            elasticity = elasticity_data['elasticity']
            
            if elasticity < -1:  # 弾力的
                strategy = 'penetration'  # 浸透価格戦略
                recommended_margin = 0.15
            elif elasticity > -0.5:  # 非弾力的
                strategy = 'premium'  # プレミアム価格戦略
                recommended_margin = 0.4
            else:  # 中程度
                strategy = 'competitive'  # 競争価格戦略
                recommended_margin = 0.25
            
            # 競合価格考慮
            if competitor_prices and product_id in competitor_prices:
                competitor_price = competitor_prices[product_id]
                current_price = elasticity_data['price_range']['mean']
                
                if competitor_price < current_price * 0.95:
                    strategy = 'competitive'
                    recommended_margin = max(0.1, (competitor_price - cost) / cost)
            
            strategies[product_id] = {
                'strategy': strategy,
                'recommended_margin': recommended_margin,
                'elasticity': elasticity,
                'elasticity_category': elasticity_data.get('category', 'normal')
            }
        
        return strategies
    
    def visualize_elasticity_analysis(self, elasticity_df):
        """弾力性分析結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 弾力性分布
        ax1 = axes[0, 0]
        ax1.hist(elasticity_df['elasticity'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=-1, color='red', linestyle='--', label='弾力性=-1')
        ax1.set_title('価格弾力性の分布')
        ax1.set_xlabel('価格弾力性')
        ax1.set_ylabel('商品数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 弾力性カテゴリ別商品数
        ax2 = axes[0, 1]
        category_counts = elasticity_df['elasticity_category'].value_counts()
        colors = {'elastic': 'red', 'inelastic': 'blue', 'normal': 'green', 'abnormal': 'orange'}
        pie_colors = [colors.get(cat, 'gray') for cat in category_counts.index]
        
        wedges, texts, autotexts = ax2.pie(
            category_counts.values, 
            labels=category_counts.index,
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90
        )
        ax2.set_title('弾力性カテゴリ別分布')
        
        # 3. 価格 vs 弾力性
        ax3 = axes[1, 0]
        scatter = ax3.scatter(
            elasticity_df['avg_price'], 
            elasticity_df['elasticity'],
            c=elasticity_df['r2_linear'],
            cmap='viridis',
            s=60,
            alpha=0.7
        )
        ax3.set_title('平均価格 vs 価格弾力性')
        ax3.set_xlabel('平均価格（円）')
        ax3.set_ylabel('価格弾力性')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='R² (線形モデル)')
        
        # 4. モデル精度比較
        ax4 = axes[1, 1]
        ax4.scatter(elasticity_df['r2_linear'], elasticity_df['r2_polynomial'], alpha=0.6)
        
        # 対角線
        max_r2 = max(elasticity_df['r2_linear'].max(), elasticity_df['r2_polynomial'].max())
        ax4.plot([0, max_r2], [0, max_r2], 'r--', alpha=0.5)
        
        ax4.set_title('線形 vs 多項式モデル精度')
        ax4.set_xlabel('R² (線形モデル)')
        ax4.set_ylabel('R² (多項式モデル)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_dir = Path("results/price_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "elasticity_analysis.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_price_optimization_results(self):
        """価格最適化結果の可視化"""
        if not self.price_optimization_results:
            print("可視化する最適化結果がありません")
            return
        
        # 結果をDataFrameに変換
        results_list = list(self.price_optimization_results.values())
        results_df = pd.DataFrame(results_list)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 価格変更率の分布
        ax1 = axes[0, 0]
        ax1.hist(results_df['price_change_percent'], bins=15, alpha=0.7, color='lightcoral')
        ax1.set_title('推奨価格変更率の分布')
        ax1.set_xlabel('価格変更率 (%)')
        ax1.set_ylabel('商品数')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # 2. 利益改善率の分布
        ax2 = axes[0, 1]
        ax2.hist(results_df['profit_improvement_percent'], bins=15, alpha=0.7, color='lightgreen')
        ax2.set_title('予測利益改善率の分布')
        ax2.set_xlabel('利益改善率 (%)')
        ax2.set_ylabel('商品数')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. 現在価格 vs 最適価格
        ax3 = axes[1, 0]
        ax3.scatter(results_df['current_price'], results_df['optimal_price'], alpha=0.6)
        
        # 対角線（変更なし）
        min_price = min(results_df['current_price'].min(), results_df['optimal_price'].min())
        max_price = max(results_df['current_price'].max(), results_df['optimal_price'].max())
        ax3.plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.5, label='変更なし')
        
        ax3.set_title('現在価格 vs 最適価格')
        ax3.set_xlabel('現在価格（円）')
        ax3.set_ylabel('最適価格（円）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 弾力性 vs 利益改善率
        ax4 = axes[1, 1]
        ax4.scatter(results_df['elasticity'], results_df['profit_improvement_percent'], alpha=0.6)
        ax4.set_title('価格弾力性 vs 利益改善率')
        ax4.set_xlabel('価格弾力性')
        ax4.set_ylabel('利益改善率 (%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.axvline(x=-1, color='red', linestyle='--', alpha=0.5, label='弾力性=-1')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_dir = Path("results/price_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "price_optimization_results.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_optimization_results(self):
        """最適化結果の保存"""
        if not self.price_optimization_results:
            print("保存する結果がありません")
            return
        
        output_dir = Path("results/price_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 最適化結果の保存
        results_df = pd.DataFrame(list(self.price_optimization_results.values()))
        results_file = output_dir / "price_optimization_results.csv"
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        
        # 実行用価格変更リスト
        price_changes = results_df[['product_id', 'current_price', 'optimal_price', 'price_change_percent']].copy()
        price_changes['action'] = price_changes['price_change_percent'].apply(
            lambda x: 'increase' if x > 0 else 'decrease' if x < 0 else 'maintain'
        )
        
        action_file = output_dir / "price_change_actions.csv"
        price_changes.to_csv(action_file, index=False, encoding='utf-8-sig')
        
        print(f"価格最適化結果を保存:")
        print(f"  詳細結果: {results_file}")
        print(f"  実行アクション: {action_file}")
        
        return results_file, action_file

def create_sample_price_sales_data():
    """サンプル価格・売上データの生成"""
    np.random.seed(42)
    
    products = [f"PROD_{i:04d}" for i in range(1, 51)]  # 50商品
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    price_sales_data = []
    
    for product_id in products:
        # 商品の基本特性
        base_price = np.random.uniform(1000, 10000)
        base_demand = np.random.uniform(10, 100)
        true_elasticity = np.random.uniform(-2.5, -0.2)  # 負の弾力性
        
        for date in dates:
            # 価格変動（±20%範囲）
            price_variation = np.random.uniform(-0.2, 0.2)
            price = base_price * (1 + price_variation)
            
            # 需要計算（弾力性考慮）
            price_ratio = price / base_price
            demand_multiplier = price_ratio ** true_elasticity
            
            # 季節性・ランダム要因
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            random_factor = np.random.uniform(0.8, 1.2)
            
            quantity = max(0, base_demand * demand_multiplier * seasonal_factor * random_factor)
            sales_amount = price * quantity
            
            price_sales_data.append({
                'date': date,
                'product_id': product_id,
                'price': round(price, 2),
                'quantity': int(quantity),
                'sales_amount': round(sales_amount, 2)
            })
    
    df = pd.DataFrame(price_sales_data)
    
    # 保存
    output_dir = Path("data/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "price_sales_data.csv", index=False, encoding='utf-8-sig')
    print(f"サンプル価格・売上データを生成: {output_dir / 'price_sales_data.csv'}")
    
    return df

def main():
    """メイン実行関数"""
    print("需要弾力性分析・価格最適化を開始します...")
    
    # サンプルデータ生成
    if not Path("data/sample_data/price_sales_data.csv").exists():
        print("サンプルデータを生成中...")
        create_sample_price_sales_data()
    
    # 分析器の初期化
    analyzer = DemandElasticityAnalyzer()
    
    # データ読み込み
    df = analyzer.load_price_sales_data("data/sample_data/price_sales_data.csv")
    if df is None:
        return
    
    # 全商品の弾力性分析
    print("全商品の価格弾力性を分析中...")
    elasticity_df = analyzer.analyze_all_products(df)
    
    # 可視化
    analyzer.visualize_elasticity_analysis(elasticity_df)
    
    # 価格最適化（上位10商品）
    print("\n価格最適化を実行中...")
    top_products = elasticity_df.nlargest(10, 'avg_quantity')['product_id']
    
    # 仮のコストデータ
    cost_data = {product_id: np.random.uniform(500, 3000) for product_id in top_products}
    
    optimization_count = 0
    for product_id in top_products:
        result = analyzer.optimize_price_single_product(
            product_id, 
            cost_data[product_id],
            target_margin=0.25,
            price_change_limit=0.15
        )
        if result:
            optimization_count += 1
    
    # 最適化結果の可視化
    if optimization_count > 0:
        analyzer.visualize_price_optimization_results()
        analyzer.save_optimization_results()
    
    # レポート生成
    print(f"\n=== 需要弾力性分析レポート ===")
    print(f"分析商品数: {len(elasticity_df)}商品")
    print(f"平均弾力性: {elasticity_df['elasticity'].mean():.3f}")
    
    # 弾力性カテゴリ別集計
    category_summary = elasticity_df['elasticity_category'].value_counts()
    print(f"\n弾力性カテゴリ別商品数:")
    for category, count in category_summary.items():
        print(f"  {category}: {count}商品")
    
    print(f"\n価格最適化実行数: {optimization_count}商品")
    
    if optimization_count > 0:
        results_df = pd.DataFrame(list(analyzer.price_optimization_results.values()))
        avg_profit_improvement = results_df['profit_improvement_percent'].mean()
        print(f"平均利益改善率: {avg_profit_improvement:+.1f}%")
    
    print(f"\n✅ 需要弾力性分析・価格最適化が完了しました！")
    print("📊 results/price_optimization/ フォルダに結果が保存されています。")

if __name__ == "__main__":
    main()
