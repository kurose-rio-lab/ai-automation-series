"""
ABC分析による商品分類システム
売上貢献度に基づいて商品をA/B/Cランクに分類
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

class ABCAnalyzer:
    def __init__(self, config_path="config/config.yaml"):
        """ABC分析クラスの初期化"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # ABC分析の閾値設定
        self.a_threshold = 0.8  # 売上累計80%まで
        self.b_threshold = 0.95  # 売上累計95%まで
        
    def load_sales_data(self, file_path):
        """販売データの読み込み"""
        try:
            df = pd.read_csv(file_path)
            print(f"データ読み込み完了: {len(df)}件")
            return df
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return None
    
    def calculate_abc_classification(self, df):
        """ABC分類の実行"""
        # 商品別売上集計
        product_sales = df.groupby('product_id').agg({
            'sales_amount': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        # 売上順でソート
        product_sales = product_sales.sort_values('sales_amount', ascending=False)
        
        # 累積売上比率を計算
        total_sales = product_sales['sales_amount'].sum()
        product_sales['sales_ratio'] = product_sales['sales_amount'] / total_sales
        product_sales['cumulative_ratio'] = product_sales['sales_ratio'].cumsum()
        
        # ABC分類を実行
        product_sales['abc_class'] = 'C'
        product_sales.loc[product_sales['cumulative_ratio'] <= self.a_threshold, 'abc_class'] = 'A'
        product_sales.loc[
            (product_sales['cumulative_ratio'] > self.a_threshold) & 
            (product_sales['cumulative_ratio'] <= self.b_threshold), 
            'abc_class'
        ] = 'B'
        
        return product_sales
    
    def generate_analysis_report(self, abc_results):
        """分析レポートの生成"""
        print("\n" + "="*50)
        print("ABC分析結果レポート")
        print("="*50)
        
        # クラス別統計
        class_summary = abc_results.groupby('abc_class').agg({
            'product_id': 'count',
            'sales_amount': ['sum', 'mean'],
            'sales_ratio': 'sum'
        }).round(4)
        
        print("\nクラス別商品数と売上貢献度:")
        print(class_summary)
        
        # 各クラスの特徴
        for class_name in ['A', 'B', 'C']:
            class_data = abc_results[abc_results['abc_class'] == class_name]
            product_count = len(class_data)
            sales_contribution = class_data['sales_ratio'].sum()
            
            print(f"\n【{class_name}クラス】")
            print(f"  商品数: {product_count}商品 ({product_count/len(abc_results)*100:.1f}%)")
            print(f"  売上貢献: {sales_contribution:.1%}")
            print(f"  平均売上: {class_data['sales_amount'].mean():,.0f}円")
        
        return class_summary
    
    def create_visualization(self, abc_results):
        """ABC分析の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. パレート図
        ax1 = axes[0, 0]
        x_pos = range(len(abc_results))
        bars = ax1.bar(x_pos, abc_results['sales_ratio'], alpha=0.7, 
                       color=['red' if x == 'A' else 'yellow' if x == 'B' else 'green' 
                              for x in abc_results['abc_class']])
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x_pos, abc_results['cumulative_ratio'], 'ko-', linewidth=2)
        ax1_twin.axhline(y=self.a_threshold, color='red', linestyle='--', alpha=0.7, label='80%ライン')
        ax1_twin.axhline(y=self.b_threshold, color='orange', linestyle='--', alpha=0.7, label='95%ライン')
        
        ax1.set_title('ABC分析パレート図')
        ax1.set_xlabel('商品（売上順）')
        ax1.set_ylabel('売上比率')
        ax1_twin.set_ylabel('累積売上比率')
        ax1_twin.legend()
        
        # 2. クラス別商品数
        ax2 = axes[0, 1]
        class_counts = abc_results['abc_class'].value_counts()
        colors = ['red', 'yellow', 'green']
        ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax2.set_title('クラス別商品数分布')
        
        # 3. クラス別売上分布
        ax3 = axes[1, 0]
        class_sales = abc_results.groupby('abc_class')['sales_ratio'].sum()
        bars = ax3.bar(class_sales.index, class_sales.values, color=colors, alpha=0.7)
        ax3.set_title('クラス別売上貢献度')
        ax3.set_ylabel('売上比率')
        
        # 数値ラベル追加
        for bar, value in zip(bars, class_sales.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.1%}', ha='center', va='bottom')
        
        # 4. 売上分布ヒストグラム
        ax4 = axes[1, 1]
        for class_name, color in zip(['A', 'B', 'C'], colors):
            class_data = abc_results[abc_results['abc_class'] == class_name]
            ax4.hist(np.log10(class_data['sales_amount']), bins=20, alpha=0.5, 
                    label=f'{class_name}クラス', color=color)
        
        ax4.set_title('売上分布（対数スケール）')
        ax4.set_xlabel('log10(売上金額)')
        ax4.set_ylabel('商品数')
        ax4.legend()
        
        plt.tight_layout()
        
        # 結果保存
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "abc_analysis_visualization.png", dpi=300, bbox_inches='tight')
        print(f"\n可視化結果を保存: {output_dir / 'abc_analysis_visualization.png'}")
        
        return fig
    
    def save_results(self, abc_results):
        """結果の保存"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        # CSV出力
        output_file = output_dir / "abc_analysis_results.csv"
        abc_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"ABC分析結果を保存: {output_file}")
        
        # 管理用サマリー
        summary_data = []
        for class_name in ['A', 'B', 'C']:
            class_data = abc_results[abc_results['abc_class'] == class_name]
            summary_data.append({
                'class': class_name,
                'product_count': len(class_data),
                'sales_contribution': class_data['sales_ratio'].sum(),
                'avg_sales': class_data['sales_amount'].mean()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "abc_summary.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"ABC分析サマリーを保存: {summary_file}")
        
        return output_file, summary_file

def main():
    """メイン実行関数"""
    print("ABC分析を開始します...")
    
    # 分析器の初期化
    analyzer = ABCAnalyzer()
    
    # サンプルデータの生成（実際のデータがない場合）
    if not Path("data/sample_data/sales_data.csv").exists():
        print("サンプルデータを生成中...")
        create_sample_sales_data()
    
    # データ読み込み
    df = analyzer.load_sales_data("data/sample_data/sales_data.csv")
    if df is None:
        return
    
    # ABC分析実行
    abc_results = analyzer.calculate_abc_classification(df)
    
    # レポート生成
    summary = analyzer.generate_analysis_report(abc_results)
    
    # 可視化
    analyzer.create_visualization(abc_results)
    
    # 結果保存
    analyzer.save_results(abc_results)
    
    print("\n✅ ABC分析が完了しました！")
    print("📊 results/ フォルダに結果ファイルが保存されています。")

def create_sample_sales_data():
    """サンプル販売データの生成"""
    np.random.seed(42)
    
    # 商品データ生成
    n_products = 1000
    product_ids = [f"PROD_{i:04d}" for i in range(1, n_products + 1)]
    
    # パレート分布に従う売上データ生成
    # 20%の商品が80%の売上を占める分布
    sales_data = []
    
    for i, product_id in enumerate(product_ids):
        # パレート分布での基本売上
        if i < n_products * 0.2:  # 上位20%商品
            base_sales = np.random.exponential(scale=500000) + 100000
        elif i < n_products * 0.5:  # 中位30%商品
            base_sales = np.random.exponential(scale=50000) + 10000
        else:  # 下位50%商品
            base_sales = np.random.exponential(scale=5000) + 1000
        
        # 月次販売データを生成
        for month in range(1, 13):
            monthly_sales = base_sales * np.random.uniform(0.8, 1.2)
            quantity = int(monthly_sales / np.random.uniform(1000, 5000))
            
            sales_data.append({
                'product_id': product_id,
                'month': month,
                'sales_amount': monthly_sales,
                'quantity': quantity
            })
    
    # DataFrame作成
    df = pd.DataFrame(sales_data)
    
    # データ保存
    output_dir = Path("data/sample_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "sales_data.csv", index=False, encoding='utf-8-sig')
    print(f"サンプルデータを生成: {output_dir / 'sales_data.csv'}")

if __name__ == "__main__":
    main()
