"""
金融・保険業界向けAI統合システム
メインシステムファイル

このファイルは、リスク評価、ポートフォリオ最適化、不正検知、保険料計算の
4つの主要機能を統合したシステムの中核部分です。
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 各エンジンのインポート
from risk_engine import RiskAssessmentEngine
from portfolio_optimizer import PortfolioOptimizer  
from fraud_detector import FraudDetector
from insurance_calculator import InsuranceCalculator

class FinancialAISystem:
    """
    金融AI統合システムのメインクラス
    
    このクラスは4つの主要エンジンを統合し、
    包括的な金融サービスを提供します。
    """
    
    def __init__(self):
        """
        システムの初期化
        各エンジンのインスタンスを作成し、
        必要なデータを読み込みます。
        """
        print("🏦 金融AI統合システムを初期化中...")
        
        # 各エンジンの初期化
        self.risk_engine = RiskAssessmentEngine()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.fraud_detector = FraudDetector()
        self.insurance_calculator = InsuranceCalculator()
        
        # サンプルデータの読み込み
        self.load_sample_data()
        
        print("✅ システム初期化完了")
    
    def load_sample_data(self):
        """
        サンプルデータの読み込み
        
        customer_data.csv: 顧客基本情報
        market_data.csv: 市場データ
        transaction_history.csv: 取引履歴
        """
        try:
            # 顧客データの読み込み
            self.customer_data = pd.read_csv('data/customer_data.csv')
            print(f"📊 顧客データ読み込み完了: {len(self.customer_data)}件")
            
            # 市場データの読み込み
            self.market_data = pd.read_csv('data/market_data.csv')
            print(f"📈 市場データ読み込み完了: {len(self.market_data)}件")
            
            # 取引履歴の読み込み
            self.transaction_history = pd.read_csv('data/transaction_history.csv')
            print(f"💳 取引履歴読み込み完了: {len(self.transaction_history)}件")
            
        except FileNotFoundError as e:
            print(f"⚠️ データファイルが見つかりません: {e}")
            print("サンプルデータを生成します...")
            self.generate_sample_data()
    
    def generate_sample_data(self):
        """
        サンプルデータの自動生成
        実際のシステムではAPI経由でリアルタイムデータを取得しますが、
        デモ用にサンプルデータを生成します。
        """
        # 顧客データの生成
        np.random.seed(42)  # 再現性のための固定シード
        
        customer_ids = [f"CUST_{i:06d}" for i in range(1000, 1101)]
        ages = np.random.normal(45, 15, 101).astype(int)
        ages = np.clip(ages, 20, 80)  # 20-80歳の範囲に制限
        
        incomes = np.random.lognormal(np.log(5000000), 0.5, 101).astype(int)
        occupations = np.random.choice(['会社員', '公務員', '自営業', '医師', '弁護士'], 101)
        genders = np.random.choice(['男性', '女性'], 101)
        
        self.customer_data = pd.DataFrame({
            'customer_id': customer_ids,
            'age': ages,
            'income': incomes,
            'occupation': occupations,
            'gender': genders
        })
        
        print("✅ サンプル顧客データ生成完了")
    
    def comprehensive_analysis(self, customer_id):
        """
        包括的分析の実行
        
        指定された顧客に対して、4つのエンジンすべてを使用した
        総合的な分析を実行します。
        
        Args:
            customer_id (str): 分析対象の顧客ID
            
        Returns:
            dict: 分析結果の統合レポート
        """
        print(f"\n🔍 顧客ID {customer_id} の包括的分析を開始...")
        
        # 顧客データの取得
        customer_info = self.get_customer_info(customer_id)
        if customer_info is None:
            return {"error": "顧客データが見つかりません"}
        
        results = {
            'customer_id': customer_id,
            'timestamp': datetime.now().isoformat(),
            'customer_info': customer_info
        }
        
        # 1. リスク評価の実行
        print("📊 リスク評価を実行中...")
        risk_results = self.risk_engine.assess_risk(customer_info)
        results['risk_assessment'] = risk_results
        
        # 2. ポートフォリオ最適化の実行
        print("📈 ポートフォリオ最適化を実行中...")
        portfolio_results = self.portfolio_optimizer.optimize_portfolio(
            customer_info, risk_results['total_risk_score']
        )
        results['portfolio_optimization'] = portfolio_results
        
        # 3. 不正検知分析の実行
        print("🛡️ 不正検知分析を実行中...")
        fraud_results = self.fraud_detector.analyze_customer_pattern(customer_id)
        results['fraud_analysis'] = fraud_results
        
        # 4. 保険料計算の実行
        print("🏥 保険料計算を実行中...")
        insurance_results = self.insurance_calculator.calculate_premium(customer_info)
        results['insurance_calculation'] = insurance_results
        
        print("✅ 包括的分析完了")
        return results
    
    def get_customer_info(self, customer_id):
        """
        顧客情報の取得
        
        Args:
            customer_id (str): 顧客ID
            
        Returns:
            dict: 顧客情報、または None（見つからない場合）
        """
        customer_row = self.customer_data[
            self.customer_data['customer_id'] == customer_id
        ]
        
        if customer_row.empty:
            return None
        
        return customer_row.iloc[0].to_dict()
    
    def display_comprehensive_results(self, results):
        """
        分析結果の表示
        
        Args:
            results (dict): comprehensive_analysis()の実行結果
        """
        if 'error' in results:
            print(f"❌ エラー: {results['error']}")
            return
        
        print("\n" + "="*60)
        print(f"📋 包括的分析レポート")
        print("="*60)
        
        # 顧客情報の表示
        customer_info = results['customer_info']
        print(f"\n👤 顧客情報:")
        print(f"   ID: {results['customer_id']}")
        print(f"   年齢: {customer_info['age']}歳")
        print(f"   年収: {customer_info['income']:,}円")
        print(f"   職業: {customer_info['occupation']}")
        print(f"   性別: {customer_info['gender']}")
        
        # リスク評価結果
        risk_info = results['risk_assessment']
        print(f"\n📊 リスク評価結果:")
        print(f"   総合リスクスコア: {risk_info['total_risk_score']:.1f}/100")
        print(f"   年齢リスク: {risk_info['age_risk']:.1f}")
        print(f"   職業リスク: {risk_info['occupation_risk']:.1f}")
        print(f"   収入リスク: {risk_info['income_risk']:.1f}")
        print(f"   リスクレベル: {risk_info['risk_level']}")
        
        # ポートフォリオ最適化結果
        portfolio_info = results['portfolio_optimization']
        print(f"\n📈 最適ポートフォリオ:")
        print(f"   期待リターン: {portfolio_info['expected_return']:.2%}")
        print(f"   リスク: {portfolio_info['risk']:.2%}")
        print(f"   資産配分:")
        for asset, weight in portfolio_info['allocation'].items():
            print(f"     {asset}: {weight:.1%}")
        
        # 不正検知結果
        fraud_info = results['fraud_analysis']
        print(f"\n🛡️ 不正検知分析:")
        print(f"   異常スコア: {fraud_info['anomaly_score']:.1f}/100")
        print(f"   リスクレベル: {fraud_info['risk_level']}")
        if fraud_info['risk_level'] != '低':
            print(f"   ⚠️ 注意事項: {fraud_info['recommendation']}")
        
        # 保険料計算結果
        insurance_info = results['insurance_calculation']
        print(f"\n🏥 保険料計算結果:")
        for insurance_type, premium in insurance_info.items():
            if insurance_type != 'total':
                print(f"   {insurance_type}: 年額{premium:,}円")
        print(f"   合計年額: {insurance_info['total']:,}円")
        
        print("\n" + "="*60)
        print(f"📅 分析実行時刻: {results['timestamp']}")
        print("="*60)

def main():
    """
    メイン実行関数
    システムのデモンストレーションを実行します。
    """
    print("🚀 金融AI統合システム デモンストレーション")
    print("="*50)
    
    # システムの初期化
    system = FinancialAISystem()
    
    # サンプル顧客での分析実行
    sample_customer_id = "CUST_001000"
    
    print(f"\n📋 サンプル顧客 {sample_customer_id} での分析を実行します...")
    
    # 包括的分析の実行
    results = system.comprehensive_analysis(sample_customer_id)
    
    # 結果の表示
    system.display_comprehensive_results(results)
    
    # 複数顧客での比較分析（オプション）
    print("\n🔄 複数顧客での比較分析も可能です...")
    customer_ids = ["CUST_001001", "CUST_001002", "CUST_001003"]
    
    comparison_results = []
    for cid in customer_ids:
        result = system.comprehensive_analysis(cid)
        if 'error' not in result:
            comparison_results.append(result)
    
    # 比較結果の簡易表示
    if comparison_results:
        print("\n📊 顧客比較サマリー:")
        print("-" * 80)
        print(f"{'顧客ID':<12} {'リスクスコア':<12} {'期待リターン':<12} {'年額保険料':<12}")
        print("-" * 80)
        
        for result in comparison_results:
            customer_id = result['customer_id']
            risk_score = result['risk_assessment']['total_risk_score']
            expected_return = result['portfolio_optimization']['expected_return']
            insurance_total = result['insurance_calculation']['total']
            
            print(f"{customer_id:<12} {risk_score:<12.1f} {expected_return:<12.1%} {insurance_total:<12,}")
    
    print("\n✅ デモンストレーション完了")
    print("\n💡 詳細な機能については、各エンジンのファイルを参照してください:")
    print("   - risk_engine.py: リスク評価の詳細")
    print("   - portfolio_optimizer.py: ポートフォリオ最適化の詳細")  
    print("   - fraud_detector.py: 不正検知の詳細")
    print("   - insurance_calculator.py: 保険料計算の詳細")

if __name__ == "__main__":
    main()
