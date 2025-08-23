"""
利益最大化価格設定システム
需要予測、競合価格、コスト構造を考慮した最適価格の決定
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import yaml

class ProfitMaximizer:
    def __init__(self, config_path="config/config.yaml"):
        """利益最大化システムの初期化"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.demand_models = {}
        self.optimization_results = {}
        
        # 価格設定制約
        self.max_price_change = self.config['model_settings']['price_optimization']['max_price_change_percent'] / 100
        self.min_margin = self.config['model_settings']['price_optimization']['min_margin_percent'] / 100
    
    def load_comprehensive_data(self, sales_file, cost_file, competitor_file=None):
        """包括的データの読み込み"""
        try:
            # 売上データ
            sales_df = pd.read_csv(sales_file)
            sales_df['date'] = pd.to_datetime(sales_df['date'])
            
            # コストデータ
            cost_df = pd.read_csv(cost_file)
            
            # 競合価格データ（オプション）
            competitor_df = None
            if competitor_file and Path(competitor_file).exists():
                competitor_df = pd.read_csv(competitor_file)
                competitor_df['date'] = pd.to_datetime(competitor_df['date'])
            
            print(f"データ読み込み完了:")
            print(f"  売上データ: {len(sales_df)}レコード")
            print(f"  コストデータ: {len(cost_df)}商品")
            if competitor_df is not None:
                print(f"  競合価格データ: {len(competitor_df)}レコード")
            
            return sales_df, cost_df, competitor_df
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return None, None, None
    
    def build_demand_prediction_model(self, sales_df, product_id):
        """需要予測モデルの構築"""
        # 商品データの抽出
        product_data = sales_df[sales_df['product_id'] == product_id].copy()
        
        if len(product_data) < 30:
            print(f"商品 {product_id} のデータが不足")
            return None
        
        # 特徴量エンジニアリング
        product_data = product_data.sort_values('date')
        
        # 時間特徴量
        product_data['dayofweek'] = product_data['date'].dt.dayofweek
        product_data['month'] = product_data['date'].dt.month
        product_data['quarter'] = product_data['date'].dt.quarter
        product_data['is_weekend'] = (product_data['dayofweek'] >= 5).astype(int)
        
        # ラグ特徴量
        for lag in [1, 7, 14]:
            product_data[f'price_lag_{lag}'] = product_data['price'].shift(lag)
            product_data[f'quantity_lag_{lag}'] = product_data['quantity'].shift(lag)
        
        # 移動平均
        for window in [7, 14]:
            product_data[f'price_ma_{window}'] = product_data['price'].rolling(window).mean()
            product_data[f'quantity_ma_{window}'] = product_data['quantity'].rolling(window).mean()
        
        # 価格変化率
        product_data['price_change'] = product_data['price'].pct_change()
        
        # 欠損値除去
        product_data = product_data.dropna()
        
        if len(product_data) < 20:
            print(f"商品 {product_id} の有効データが不足")
            return None
        
        # 特徴量とターゲット
        feature_columns = [
            'price', 'dayofweek', 'month', 'quarter', 'is_weekend',
            'price_lag_1', 'price_lag_7', 'quantity_lag_1', 'quantity_lag_7',
            'price_ma_7', 'quantity_ma_7', 'price_change'
        ]
        
        X = product_data[feature_columns]
        y = product_data['quantity']
        
        # モデル訓練
        if len(X) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Random Forest モデル
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # 性能評価
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test) if len(X_test) > 0 else train_score
        
        model_info = {
            'model': model,
            'feature_columns': feature_columns,
            'train_score': train_score,
            'test_score': test_score,
            'data_points': len(product_data),
            'latest_features': product_data[feature_columns].iloc[-1].to_dict()
        }
        
        self.demand_models[product_id] = model_info
        
        print(f"商品 {product_id} 需要予測モデル構築完了 (R²: {test_score:.3f})")
        
        return model_info
    
    def predict_demand(self, product_id, price, additional_features=None):
        """指定価格での需要予測"""
        if product_id not in self.demand_models:
            print(f"商品 {product_id} の需要予測モデルが見つかりません")
            return 0
        
        model_info = self.demand_models[product_id]
        model = model_info['model']
        feature_columns = model_info['feature_columns']
        
        # 基準特徴量（最新データから取得）
        base_features = model_info['latest_features'].copy()
        base_features['price'] = price
        
        # 価格変化率の更新
        if 'price_change' in base_features:
            original_price = model_info['latest_features']['price']
            base_features['price_change'] = (price - original_price) / original_price
        
        # 追加特徴量の反映
        if additional_features:
            base_features.update(additional_features)
        
        # 予測実行
        feature_vector = [base_features[col] for col in feature_columns]
        predicted_demand = model.predict([feature_vector])[0]
        
        return max(0, predicted_demand)  # 負の需要は0
    
    def calculate_profit(self, product_id, price, cost_per_unit, fixed_cost=0):
        """利益計算"""
        predicted_demand = self.predict_demand(product_id, price)
        
        revenue = price * predicted_demand
        variable_cost = cost_per_unit * predicted_demand
        total_profit = revenue - variable_cost - fixed_cost
        
        return {
            'price': price,
            'predicted_demand': predicted_demand,
            'revenue': revenue,
            'variable_cost': variable_cost,
            'total_profit': total_profit,
            'margin_percent': ((price - cost_per_unit) / price) * 100 if price > 0 else 0
        }
    
    def optimize_single_product_price(self, product_id, cost_per_unit, 
                                    current_price=None, competitor_price=None):
        """単一商品の価格最適化"""
        if product_id not in self.demand_models:
            print(f"商品 {product_id} の需要予測モデルが必要です")
            return None
        
        # 現在価格の設定
        if current_price is None:
            current_price = self.demand_models[product_id]['latest_features']['price']
        
        # 価格範囲の設定
        min_price = cost_per_unit * (1 + self.min_margin)
        max_price = current_price * (1 + self.max_price_change)
        
        # 競合価格を考慮した上限調整
        if competitor_price:
            max_price = min(max_price, competitor_price * 1.1)  # 競合価格の110%以下
        
        # 利益関数（最大化したい値の負数）
        def negative_profit_function(price):
            profit_info = self.calculate_profit(product_id, price[0], cost_per_unit)
            return -profit_info['total_profit']
        
        # 制約条件
        bounds = [(min_price, max_price)]
        
        # 最適化実行（差分進化アルゴリズム使用）
        result = differential_evolution(
            negative_profit_function,
            bounds,
            seed=42,
            maxiter=1000,
            atol=1e-6
        )
        
        if result.success:
            optimal_price = result.x[0]
            
            # 最適化結果の詳細計算
            current_profit_info = self.calculate_profit(product_id, current_price, cost_per_unit)
            optimal_profit_info = self.calculate_profit(product_id, optimal_price, cost_per_unit)
            
            optimization_result = {
                'product_id': product_id,
                'current_price': current_price,
                'optimal_price': optimal_price,
                'cost_per_unit': cost_per_unit,
                'competitor_price': competitor_price,
                
                # 現在の状況
                'current_demand': current_profit_info['predicted_demand'],
                'current_profit': current_profit_info['total_profit'],
                'current_margin': current_profit_info['margin_percent'],
                
                # 最適化後の予測
                'optimal_demand': optimal_profit_info['predicted_demand'],
                'optimal_profit': optimal_profit_info['total_profit'],
                'optimal_margin': optimal_profit_info['margin_percent'],
                
                # 改善効果
                'price_change_percent': ((optimal_price - current_price) / current_price) * 100,
                'demand_change_percent': ((optimal_profit_info['predicted_demand'] - current_profit_info['predicted_demand']) / current_profit_info['predicted_demand']) * 100,
                'profit_improvement_percent': ((optimal_profit_info['total_profit'] - current_profit_info['total_profit']) / abs(current_profit_info['total_profit'])) * 100 if current_profit_info['total_profit'] != 0 else 0,
                
                # メタデータ
                'optimization_success': True,
                'model_r2': self.demand_models[product_id]['test_score']
            }
            
            self.optimization_results[product_id] = optimization_result
            
            print(f"商品 {product_id} 価格最適化完了:")
            print(f"  最適価格: {optimal_price:.2f}円 ({optimization_result['price_change_percent']:+.1f}%)")
            print(f"  予測利益改善: {optimization_result['profit_improvement_percent']:+.1f}%")
            
            return optimization_result
        
        else:
            print(f"商品 {product_id} の価格最適化に失敗")
            return None
    
    def portfolio_price_optimization(self, sales_df, cost_df, competitor_df=None):
        """商品ポートフォリオ全体の価格最適化"""
        print("商品ポートフォリオの価格最適化を実行中...")
        
        products = cost_df['product_id'].tolist()
        optimization_summary = []
        
        for product_id in products:
            # 需要予測モデル構築
            model_info = self.build_demand_prediction_model(sales_df, product_id)
            if model_info is None:
                continue
            
            # コスト情報取得
            cost_info = cost_df[cost_df['product_id'] == product_id].iloc[0]
            cost_per_unit = cost_info['cost_per_unit']
            
            # 競合価格取得（ある場合）
            competitor_price = None
            if competitor_df is not None:
                competitor_data = competitor_df[competitor_df['product_id'] == product_id]
                if len(competitor_data) > 0:
                    competitor_price = competitor_data['competitor_price'].iloc[-1]
            
            # 価格最適化実行
            result = self.optimize_single_product_price(
                product_id, 
                cost_per_unit,
                competitor_price=competitor_price
            )
            
            if result:
                optimization_summary.append(result)
        
        print(f"ポートフォリオ最適化完了: {len(optimization_summary)}商品")
        
        return optimization_summary
    
    def simulate_price_scenarios(self, product_id, cost_per_unit, 
                               price_range=None, num_points=50):
        """価格シナリオシミュレーション"""
        if product_id not in self.demand_models:
            print(f"商品 {product_id} の需要予測モデルが必要です")
            return None
        
        # 価格範囲の設定
        if price_range is None:
            current_price = self.demand_models[product_id]['latest_features']['price']
            min_price = current_price * 0.7
            max_price = current_price * 1.3
        else:
            min_price, max_price = price_range
        
        # 価格ポイント生成
        prices = np.linspace(min_price, max_price, num_points)
        
        # シミュレーション実行
        simulation_results = []
        
        for price in prices:
            profit_info = self.calculate_profit(product_id, price, cost_per_unit)
            simulation_results.append(profit_info)
        
        simulation_df = pd.DataFrame(simulation_results)
        
        return simulation_df
    
    def visualize_price_optimization(self):
        """価格最適化結果の可視化"""
        if not self.optimization_results:
            print("可視化する最適化結果がありません")
            return
        
        results_df = pd.DataFrame(list(self.optimization_results.values()))
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. 価格変更分布
        ax1 = axes[0, 0]
        ax1.hist(results_df['price_change_percent'], bins=15, alpha=0.7, color='skyblue')
        ax1.set_title('価格変更率の分布')
        ax1.set_xlabel('価格変更率 (%)')
        ax1.set_ylabel('商品数')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax1.grid(True, alpha=0.3)
        
        # 2. 利益改善分布
        ax2 = axes[0, 1]
        ax2.hist(results_df['profit_improvement_percent'], bins=15, alpha=0.7, color='lightgreen')
        ax2.set_title('利益改善率の分布')
        ax2.set_xlabel('利益改善率 (%)')
        ax2.set_ylabel('商品数')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax2.grid(True, alpha=0.3)
        
        # 3. 現在価格 vs 最適価格
        ax3 = axes[0, 2]
        ax3.scatter(results_df['current_price'], results_df['optimal_price'], alpha=0.6)
        min_p = min(results_df['current_price'].min(), results_df['optimal_price'].min())
        max_p = max(results_df['current_price'].max(), results_df['optimal_price'].max())
        ax3.plot([min_p, max_p], [min_p, max_p], 'r--', alpha=0.7, label='変更なし')
        ax3.set_title('現在価格 vs 最適価格')
        ax3.set_xlabel('現在価格（円）')
        ax3.set_ylabel('最適価格（円）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. マージン変化
        ax4 = axes[1, 0]
        ax4.scatter(results_df['current_margin'], results_df['optimal_margin'], alpha=0.6)
        min_m = min(results_df['current_margin'].min(), results_df['optimal_margin'].min())
        max_m = max(results_df['current_margin'].max(), results_df['optimal_margin'].max())
        ax4.plot([min_m, max_m], [min_m, max_m], 'r--', alpha=0.7, label='変更なし')
        ax4.set_title('現在マージン vs 最適マージン')
        ax4.set_xlabel('現在マージン (%)')
        ax4.set_ylabel('最適マージン (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 需要変化 vs 利益改善
        ax5 = axes[1, 1]
        ax5.scatter(results_df['demand_change_percent'], results_df['profit_improvement_percent'], alpha=0.6)
        ax5.set_title('需要変化 vs 利益改善')
        ax5.set_xlabel('需要変化率 (%)')
        ax5.set_ylabel('利益改善率 (%)')
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax5.grid(True, alpha=0.3)
        
        # 6. モデル精度 vs 利益改善
        ax6 = axes[1, 2]
        ax6.scatter(results_df['model_r2'], results_df['profit_improvement_percent'], alpha=0.6)
        ax6.set_title('モデル精度 vs 利益改善')
        ax6.set_xlabel('モデルR²')
        ax6.set_ylabel('利益改善率 (%)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_dir = Path("results/price_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "profit_maximization_analysis.png", dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_price_scenario_chart(self, product_id, cost_per_unit):
        """価格シナリオチャートの作成"""
        simulation_df = self.simulate_price_scenarios(product_id, cost_per_unit)
        
        if simulation_df is None:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 価格 vs 利益
        ax1 = axes[0]
        ax1.plot(simulation_df['price'], simulation_df['total_profit'], 'b-', linewidth=2)
        
        # 最適価格をマーク
        if product_id in self.optimization_results:
            optimal_price = self.optimization_results[product_id]['optimal_price']
            optimal_profit_row = simulation_df.iloc[(simulation_df['price'] - optimal_price).abs().argsort()[:1]]
            ax1.plot(optimal_profit_row['price'], optimal_profit_row['total_profit'], 'ro', markersize=8, label='最適価格')
        
        ax1.set_title(f'商品 {product_id}: 価格 vs 利益')
        ax1.set_xlabel('価格（円）')
        ax1.set_ylabel('利益（円）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 価格 vs 需要
        ax2 = axes[1]
        ax2.plot(simulation_df['price'], simulation_df['predicted_demand'], 'g-', linewidth=2)
        
        if product_id in self.optimization_results:
            optimal_price = self.optimization_results[product_id]['optimal_price']
            optimal_demand_row = simulation_df.iloc[(simulation_df['price'] - optimal_price).abs().argsort()[:1]]
            ax2.plot(optimal_demand_row['price'], optimal_demand_row['predicted_demand'], 'ro', markersize=8, label='最適価格')
        
        ax2.set_title(f'商品 {product_id}: 価格 vs 需要')
        ax2.set_xlabel('価格（円）')
        ax2.set_ylabel('予測需要（個）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def generate_pricing_recommendations(self):
        """価格設定推奨レポート"""
        if not self.optimization_results:
            print("推奨事項を生成する結果がありません")
            return {}
        
        results_df = pd.DataFrame(list(self.optimization_results.values()))
        
        recommendations = {
            'summary': {
                'total_products': len(results_df),
                'avg_profit_improvement': results_df['profit_improvement_percent'].mean(),
                'products_with_price_increase': len(results_df[results_df['price_change_percent'] > 0]),
                'products_with_price_decrease': len(results_df[results_df['price_change_percent'] < 0]),
            },
            'top_opportunities': results_df.nlargest(5, 'profit_improvement_percent')[
                ['product_id', 'price_change_percent', 'profit_improvement_percent']
            ].to_dict('records'),
            'high_risk_changes': results_df[
                abs(results_df['price_change_percent']) > 10
            ][['product_id', 'price_change_percent', 'model_r2']].to_dict('records'),
            'implementation_priority': results_df[
                (results_df['profit_improvement_percent'] > 5) & 
                (abs(results_df['price_change_percent']) < 15) &
                (results_df['model_r2'] > 0.7)
            ].nlargest(10, 'profit_improvement_percent')[
                ['product_id', 'current_price', 'optimal_price', 'profit_improvement_percent']
            ].to_dict('records')
        }
        
        return recommendations

def create_sample_cost_data():
    """サンプルコストデータの生成"""
    products = [f"PROD_{i:04d}" for i in range(1, 51)]
    
    cost_data = []
    for product_id in products:
        # コスト構造の生成
        cost_per_unit = np.random.uniform(300, 2500)
        
        cost_data.append({
            'product_id': product_id,
            'cost_per_unit': cost_per_unit,
            'fixed_cost_monthly': np.random.uniform(10000, 100000),
            'supplier': f"SUPPLIER_{np.random.randint(1, 11):02d}"
        })
    
    df = pd.DataFrame(cost_data)
    
    # 保存
    output_dir = Path("data/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "cost_data.csv", index=False, encoding='utf-8-sig')
    print(f"サンプルコストデータを生成: {output_dir / 'cost_data.csv'}")
    
    return df

def create_sample_competitor_data():
    """サンプル競合価格データの生成"""
    products = [f"PROD_{i:04d}" for i in range(1, 51)]
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='W')  # 週次
    
    competitor_data = []
    for product_id in products:
        base_competitor_price = np.random.uniform(1200, 12000)
        
        for date in dates:
            # 価格変動
            price_variation = np.random.uniform(-0.1, 0.1)
            competitor_price = base_competitor_price * (1 + price_variation)
            
            competitor_data.append({
                'date': date,
                'product_id': product_id,
                'competitor_price': round(competitor_price, 2),
                'competitor_name': f"COMP_{np.random.randint(1, 6)}"
            })
    
    df = pd.DataFrame(competitor_data)
    
    # 保存
    output_dir = Path("data/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "competitor_prices.csv", index=False, encoding='utf-8-sig')
    print(f"サンプル競合価格データを生成: {output_dir / 'competitor_prices.csv'}")
    
    return df

def main():
    """メイン実行関数"""
    print("利益最大化価格設定システムを開始します...")
    
    # サンプルデータ生成
    if not Path("data/sample_data/cost_data.csv").exists():
        print("サンプルコストデータを生成中...")
        create_sample_cost_data()
        create_sample_competitor_data()
    
    # システム初期化
    optimizer = ProfitMaximizer()
    
    # データ読み込み
    sales_df, cost_df, competitor_df = optimizer.load_comprehensive_data(
        "data/sample_data/price_sales_data.csv",
        "data/sample_data/cost_data.csv",
        "data/sample_data/competitor_prices.csv"
    )
    
    if sales_df is None or cost_df is None:
        print("必要なデータファイルが見つかりません")
        return
    
    # ポートフォリオ価格最適化実行
    optimization_summary = optimizer.portfolio_price_optimization(
        sales_df, cost_df, competitor_df
    )
    
    if not optimization_summary:
        print("価格最適化できる商品がありませんでした")
        return
    
    # 可視化
    optimizer.visualize_price_optimization()
    
    # 推奨事項生成
    recommendations = optimizer.generate_pricing_recommendations()
    
    # レポート出力
    print(f"\n=== 利益最大化価格設定レポート ===")
    print(f"最適化対象商品数: {recommendations['summary']['total_products']}商品")
    print(f"平均利益改善率: {recommendations['summary']['avg_profit_improvement']:.1f}%")
    print(f"価格上昇推奨: {recommendations['summary']['products_with_price_increase']}商品")
    print(f"価格下降推奨: {recommendations['summary']['products_with_price_decrease']}商品")
    
    print(f"\n🎯 実装優先度トップ5:")
    for i, item in enumerate(recommendations['implementation_priority'][:5], 1):
        print(f"  {i}. {item['product_id']}: {item['current_price']:.0f}円 → {item['optimal_price']:.0f}円 ({item['profit_improvement_percent']:+.1f}%)")
    
    # 結果保存
    results_df = pd.DataFrame(list(optimizer.optimization_results.values()))
    output_dir = Path("results/price_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "profit_maximization_results.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 利益最大化価格設定が完了しました！")
    print(f"📊 結果保存先: {results_file}")
    print("💡 実装優先度の高い商品から段階的に価格変更を実施してください。")

if __name__ == "__main__":
    main()
