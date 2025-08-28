"""
リスク評価エンジン

このモジュールは、顧客の様々な属性から総合的なリスクスコアを算出します。
年齢、職業、収入、市場環境などを総合的に評価し、
金融サービス提供時の参考情報を提供します。
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json

class RiskAssessmentEngine:
    """
    リスク評価エンジンクラス
    
    顧客情報から多角的なリスク評価を行い、
    総合的なリスクスコアを算出します。
    """
    
    def __init__(self):
        """
        リスク評価エンジンの初期化
        各種リスク評価パラメータの設定を行います。
        """
        print("📊 リスク評価エンジンを初期化中...")
        
        # 年齢別リスク係数の設定
        # 統計データに基づく年齢別のリスク傾向
        self.age_risk_factors = {
            (20, 30): 0.15,  # 20-30歳：低リスク
            (30, 40): 0.12,  # 30-40歳：最低リスク
            (40, 50): 0.18,  # 40-50歳：やや低リスク
            (50, 60): 0.25,  # 50-60歳：中リスク
            (60, 70): 0.35,  # 60-70歳：やや高リスク
            (70, 80): 0.45   # 70-80歳：高リスク
        }
        
        # 職業別リスク係数の設定
        # 業界統計に基づく職業別の安定性評価
        self.occupation_risk_factors = {
            '公務員': 0.10,      # 最も安定
            '医師': 0.12,        # 高収入・安定
            '弁護士': 0.15,      # 高収入だが変動あり
            '会社員': 0.20,      # 一般的な安定性
            '自営業': 0.35,      # 収入変動が大きい
            'その他': 0.25       # 平均的なリスク
        }
        
        # 市場リスク要因の設定
        self.market_volatility = 0.15  # 現在の市場ボラティリティ
        
        print("✅ リスク評価エンジン初期化完了")
    
    def assess_risk(self, customer_info):
        """
        包括的リスク評価の実行
        
        Args:
            customer_info (dict): 顧客情報
                - age: 年齢
                - occupation: 職業
                - income: 年収
                - gender: 性別
                
        Returns:
            dict: リスク評価結果
        """
        print("🔍 リスク評価を実行中...")
        
        # 各要素のリスク評価
        age_risk = self.calculate_age_risk(customer_info['age'])
        occupation_risk = self.calculate_occupation_risk(customer_info['occupation'])
        income_risk = self.calculate_income_risk(customer_info['income'])
        market_risk = self.calculate_market_risk()
        
        # 重み付けによる総合リスクスコアの算出
        # 各要素の重要度を考慮した加重平均
        weights = {
            'age': 0.20,        # 年齢の影響度
            'occupation': 0.35, # 職業の影響度（最重要）
            'income': 0.30,     # 収入の影響度
            'market': 0.15      # 市場環境の影響度
        }
        
        total_risk_score = (
            age_risk * weights['age'] +
            occupation_risk * weights['occupation'] +
            income_risk * weights['income'] +
            market_risk * weights['market']
        ) * 100  # 0-100スケールに変換
        
        # リスクレベルの判定
        risk_level = self.determine_risk_level(total_risk_score)
        
        # 推奨事項の生成
        recommendations = self.generate_recommendations(
            total_risk_score, age_risk, occupation_risk, income_risk
        )
        
        result = {
            'total_risk_score': total_risk_score,
            'age_risk': age_risk * 100,
            'occupation_risk': occupation_risk * 100,
            'income_risk': income_risk * 100,
            'market_risk': market_risk * 100,
            'risk_level': risk_level,
            'recommendations': recommendations,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ リスク評価完了 - 総合スコア: {total_risk_score:.1f}")
        return result
    
    def calculate_age_risk(self, age):
        """
        年齢リスクの計算
        
        年齢統計データに基づいて、年齢に応じたリスク係数を算出します。
        一般的に、若年層と高齢層でリスク傾向が異なります。
        
        Args:
            age (int): 年齢
            
        Returns:
            float: 年齢リスク係数 (0.0-1.0)
        """
        for age_range, risk_factor in self.age_risk_factors.items():
            if age_range[0] <= age < age_range[1]:
                return risk_factor
        
        # 範囲外の場合は高リスクとして扱う
        if age < 20:
            return 0.30  # 若年層の特別リスク
        else:
            return 0.50  # 高齢層の特別リスク
    
    def calculate_occupation_risk(self, occupation):
        """
        職業リスクの計算
        
        職業統計データに基づいて、職業の安定性を評価します。
        収入の安定性、業界の将来性、経済変動への耐性を考慮します。
        
        Args:
            occupation (str): 職業
            
        Returns:
            float: 職業リスク係数 (0.0-1.0)
        """
        return self.occupation_risk_factors.get(occupation, 
                                               self.occupation_risk_factors['その他'])
    
    def calculate_income_risk(self, income):
        """
        収入リスクの計算
        
        収入水準から金融リスクを評価します。
        高収入は一般的に低リスクですが、収入源の多様性も重要です。
        
        Args:
            income (int): 年収（円）
            
        Returns:
            float: 収入リスク係数 (0.0-1.0)
        """
        # 収入レベル別のリスク評価
        if income >= 10000000:      # 1000万円以上
            return 0.10
        elif income >= 8000000:     # 800万円以上
            return 0.15
        elif income >= 6000000:     # 600万円以上
            return 0.20
        elif income >= 4000000:     # 400万円以上
            return 0.25
        elif income >= 3000000:     # 300万円以上
            return 0.35
        else:                       # 300万円未満
            return 0.45
    
    def calculate_market_risk(self):
        """
        市場リスクの計算
        
        現在の経済環境、市場ボラティリティ、金利環境などから
        市場全体のリスクレベルを評価します。
        
        Returns:
            float: 市場リスク係数 (0.0-1.0)
        """
        # 実際のシステムでは、リアルタイムの経済指標を取得
        # ここではサンプル値を使用
        
        base_market_risk = self.market_volatility
        
        # 経済指標による調整（サンプル）
        # GDP成長率、失業率、インフレ率などを考慮
        economic_adjustment = 0.05  # 現在の経済状況による調整
        
        return min(base_market_risk + economic_adjustment, 1.0)
    
    def determine_risk_level(self, risk_score):
        """
        リスクレベルの判定
        
        総合リスクスコアから、わかりやすいリスクレベルを判定します。
        
        Args:
            risk_score (float): 総合リスクスコア (0-100)
            
        Returns:
            str: リスクレベル
        """
        if risk_score <= 20:
            return '非常に低'
        elif risk_score <= 35:
            return '低'
        elif risk_score <= 50:
            return '中'
        elif risk_score <= 70:
            return '高'
        else:
            return '非常に高'
    
    def generate_recommendations(self, total_risk, age_risk, occupation_risk, income_risk):
        """
        リスク評価に基づく推奨事項の生成
        
        各リスク要因を分析し、適切な推奨事項を生成します。
        
        Args:
            total_risk (float): 総合リスクスコア
            age_risk (float): 年齢リスク
            occupation_risk (float): 職業リスク  
            income_risk (float): 収入リスク
            
        Returns:
            list: 推奨事項のリスト
        """
        recommendations = []
        
        # 総合リスクに基づく基本推奨事項
        if total_risk <= 25:
            recommendations.append("低リスクプロファイル：積極的な投資戦略が可能です")
            recommendations.append("長期的な資産形成に適した商品をお勧めします")
        elif total_risk <= 50:
            recommendations.append("中リスクプロファイル：バランス型の投資戦略をお勧めします")
            recommendations.append("リスク分散を重視したポートフォリオが適しています")
        else:
            recommendations.append("高リスクプロファイル：保守的な投資戦略をお勧めします")
            recommendations.append("安全性を重視した商品選択が重要です")
        
        # 個別リスク要因に基づく詳細推奨事項
        if age_risk > 0.3:
            recommendations.append("年齢要因：将来の収入減少に備えた保険商品の検討をお勧めします")
        
        if occupation_risk > 0.3:
            recommendations.append("職業要因：収入安定化のための緊急資金の確保をお勧めします")
        
        if income_risk > 0.3:
            recommendations.append("収入要因：収入源の多様化や副収入の検討をお勧めします")
        
        return recommendations
    
    def batch_risk_assessment(self, customers_df):
        """
        バッチリスク評価
        
        複数の顧客に対して一括でリスク評価を実行します。
        大量処理や定期的な評価更新に使用します。
        
        Args:
            customers_df (pandas.DataFrame): 顧客データのDataFrame
            
        Returns:
            pandas.DataFrame: リスク評価結果を含むDataFrame
        """
        print(f"📊 {len(customers_df)}件の顧客のバッチリスク評価を開始...")
        
        results = []
        for _, customer in customers_df.iterrows():
            customer_dict = customer.to_dict()
            risk_result = self.assess_risk(customer_dict)
            
            # 結果をフラット化してDataFrameに追加しやすくする
            flat_result = {
                'customer_id': customer_dict.get('customer_id', ''),
                'total_risk_score': risk_result['total_risk_score'],
                'risk_level': risk_result['risk_level'],
                'age_risk': risk_result['age_risk'],
                'occupation_risk': risk_result['occupation_risk'],
                'income_risk': risk_result['income_risk'],
                'market_risk': risk_result['market_risk']
            }
            results.append(flat_result)
        
        results_df = pd.DataFrame(results)
        print(f"✅ バッチリスク評価完了")
        
        return results_df

# テスト実行用の関数
def test_risk_engine():
    """
    リスク評価エンジンのテスト実行
    """
    print("🧪 リスク評価エンジンのテストを開始...")
    
    # エンジンの初期化
    risk_engine = RiskAssessmentEngine()
    
    # テスト用の顧客データ
    test_customers = [
        {
            'customer_id': 'TEST_001',
            'age': 35,
            'occupation': '会社員',
            'income': 5000000,
            'gender': '男性'
        },
        {
            'customer_id': 'TEST_002', 
            'age': 50,
            'occupation': '医師',
            'income': 12000000,
            'gender': '女性'
        },
        {
            'customer_id': 'TEST_003',
            'age': 28,
            'occupation': '自営業',
            'income': 3500000,
            'gender': '男性'
        }
    ]
    
    # 各顧客のリスク評価
    for customer in test_customers:
        print(f"\n--- {customer['customer_id']} のリスク評価 ---")
        result = risk_engine.assess_risk(customer)
        
        print(f"総合リスクスコア: {result['total_risk_score']:.1f}")
        print(f"リスクレベル: {result['risk_level']}")
        print("推奨事項:")
        for rec in result['recommendations']:
            print(f"  - {rec}")
    
    print("\n✅ リスク評価エンジンのテスト完了")

if __name__ == "__main__":
    test_risk_engine()
