"""
保険料計算エンジン

このモジュールは、顧客の個人属性、健康状態、ライフスタイルなどから
適正な保険料を計算します。AIを活用してリスクを精密に評価し、
公平で個別最適化された保険料を提供します。
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
import json

class InsuranceCalculator:
    """
    保険料計算エンジンクラス
    
    様々な保険商品（生命保険、医療保険、自動車保険など）の
    保険料を顧客の個別リスクに基づいて計算します。
    """
    
    def __init__(self):
        """
        保険料計算エンジンの初期化
        各種保険の基準料率とリスク係数を設定します。
        """
        print("🏥 保険料計算エンジンを初期化中...")
        
        # 生命保険の基準料率（年齢別・性別別）
        self.life_insurance_base_rates = {
            '男性': {
                (20, 30): 15000,   # 20-30歳の年間基準保険料
                (30, 40): 18000,   # 30-40歳
                (40, 50): 25000,   # 40-50歳
                (50, 60): 40000,   # 50-60歳
                (60, 70): 60000,   # 60-70歳
                (70, 80): 90000    # 70-80歳
            },
            '女性': {
                (20, 30): 12000,   # 女性の方が統計的に長寿のため低料率
                (30, 40): 15000,
                (40, 50): 20000,
                (50, 60): 32000,
                (60, 70): 48000,
                (70, 80): 72000
            }
        }
        
        # 医療保険の基準料率
        self.medical_insurance_base_rates = {
            '男性': {
                (20, 30): 8000,
                (30, 40): 12000,
                (40, 50): 18000,
                (50, 60): 28000,
                (60, 70): 42000,
                (70, 80): 65000
            },
            '女性': {
                (20, 30): 10000,   # 女性特有の疾患リスクを考慮
                (30, 40): 14000,
                (40, 50): 20000,
                (50, 60): 30000,
                (60, 70): 45000,
                (70, 80): 70000
            }
        }
        
        # 職業別リスク係数
        self.occupation_risk_factors = {
            '公務員': 0.85,      # 安定職業は低リスク
            '会社員': 1.00,      # 基準値
            '医師': 0.90,        # 高収入だが高ストレス
            '弁護士': 0.95,      # 高収入だが高ストレス
            '教師': 0.90,        # 比較的安定
            '自営業': 1.15,      # 収入不安定、健康管理が不規則
            '建設業': 1.25,      # 身体的リスクが高い
            '運輸業': 1.20,      # 事故リスクが高い
            'IT関係': 1.05,      # 長時間労働のリスク
            '販売業': 1.00,      # 標準的リスク
            '製造業': 1.10,      # 労働災害リスク
            'その他': 1.00       # デフォルト値
        }
        
        # 健康状態リスク係数
        self.health_risk_factors = {
            '非常に良好': 0.80,
            '良好': 0.90,
            '普通': 1.00,
            'やや不安': 1.20,
            '要注意': 1.50
        }
        
        # 生活習慣リスク係数
        self.lifestyle_risk_factors = {
            '喫煙': 1.30,        # 喫煙は大幅なリスク増
            '飲酒_適量': 0.95,   # 適量の飲酒は健康に良い場合もある
            '飲酒_過度': 1.25,   # 過度の飲酒はリスク増
            '運動習慣': 0.85,    # 定期的な運動はリスク減
            '不規則生活': 1.15,  # 不規則な生活はリスク増
            '健康管理': 0.90     # 定期健診等の健康管理
        }
        
        # 地域別リスク係数（自然災害、医療環境等を考慮）
        self.regional_risk_factors = {
            '東京': 1.05,        # 都市部は医療環境良好だが生活ストレス
            '大阪': 1.03,
            '愛知': 1.00,
            '神奈川': 1.02,
            '北海道': 0.95,      # 自然環境良好
            '沖縄': 0.90,        # 長寿県
            'その他': 1.00       # デフォルト値
        }
        
        print("✅ 保険料計算エンジン初期化完了")
    
    def calculate_premium(self, customer_info):
        """
        包括的保険料計算
        
        顧客情報から各種保険の適正保険料を計算します。
        
        Args:
            customer_info (dict): 顧客情報
                - age: 年齢
                - gender: 性別
                - occupation: 職業
                - health_status: 健康状態
                - lifestyle: 生活習慣（リスト）
                - region: 居住地域
                - income: 年収
                
        Returns:
            dict: 各種保険料の計算結果
        """
        print("💰 保険料計算を実行中...")
        
        # 基本情報の取得
        age = customer_info.get('age', 40)
        gender = customer_info.get('gender', '男性')
        occupation = customer_info.get('occupation', '会社員')
        health_status = customer_info.get('health_status', '普通')
        lifestyle = customer_info.get('lifestyle', [])
        region = customer_info.get('region', 'その他')
        income = customer_info.get('income', 5000000)
        
        # 各種リスク係数の計算
        risk_factors = self.calculate_risk_factors(
            age, gender, occupation, health_status, lifestyle, region
        )
        
        # 各保険料の計算
        life_insurance = self.calculate_life_insurance_premium(
            age, gender, risk_factors
        )
        
        medical_insurance = self.calculate_medical_insurance_premium(
            age, gender, risk_factors
        )
        
        auto_insurance = self.calculate_auto_insurance_premium(
            age, gender, risk_factors, customer_info
        )
        
        disability_insurance = self.calculate_disability_insurance_premium(
            age, gender, risk_factors, income
        )
        
        # 割引制度の適用
        discounts = self.calculate_discounts(customer_info, risk_factors)
        
        # 最終保険料の計算（割引適用後）
        final_premiums = {
            '生命保険': int(life_insurance * (1 - discounts.get('life', 0))),
            '医療保険': int(medical_insurance * (1 - discounts.get('medical', 0))),
            '自動車保険': int(auto_insurance * (1 - discounts.get('auto', 0))),
            '就業不能保険': int(disability_insurance * (1 - discounts.get('disability', 0)))
        }
        
        # 合計保険料
        total_premium = sum(final_premiums.values())
        
        result = {
            **final_premiums,
            'total': total_premium,
            'risk_analysis': risk_factors,
            'applied_discounts': discounts,
            'recommendations': self.generate_recommendations(
                customer_info, final_premiums, risk_factors
            ),
            'calculation_timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ 保険料計算完了 - 年間合計: {total_premium:,}円")
        return result
    
    def calculate_risk_factors(self, age, gender, occupation, health_status, lifestyle, region):
        """
        各種リスク係数の計算
        
        Returns:
            dict: リスク係数の詳細
        """
        # 職業リスク
        occupation_risk = self.occupation_risk_factors.get(occupation, 1.00)
        
        # 健康状態リスク
        health_risk = self.health_risk_factors.get(health_status, 1.00)
        
        # 生活習慣リスク（複数要因の複合計算）
        lifestyle_risk = 1.00
        for habit in lifestyle:
            factor = self.lifestyle_risk_factors.get(habit, 1.00)
            lifestyle_risk *= factor
        
        # 地域リスク
        regional_risk = self.regional_risk_factors.get(region, 1.00)
        
        # 年齢リスク（基準からの乖離）
        if age < 30:
            age_risk = 0.90  # 若年層はリスク低
        elif age < 50:
            age_risk = 1.00  # 中年層は標準
        elif age < 65:
            age_risk = 1.15  # 高年層はリスク増
        else:
            age_risk = 1.30  # 高齢層はリスク高
        
        # 総合リスク係数
        total_risk = (
            occupation_risk * 
            health_risk * 
            lifestyle_risk * 
            regional_risk * 
            age_risk
        )
        
        return {
            'occupation_risk': occupation_risk,
            'health_risk': health_risk,
            'lifestyle_risk': lifestyle_risk,
            'regional_risk': regional_risk,
            'age_risk': age_risk,
            'total_risk': total_risk
        }
    
    def calculate_life_insurance_premium(self, age, gender, risk_factors):
        """
        生命保険料の計算
        """
        # 年齢・性別による基準保険料の取得
        base_rate = self.get_base_rate(age, gender, self.life_insurance_base_rates)
        
        # リスク係数の適用
        premium = base_rate * risk_factors['total_risk']
        
        return premium
    
    def calculate_medical_insurance_premium(self, age, gender, risk_factors):
        """
        医療保険料の計算
        """
        # 年齢・性別による基準保険料の取得
        base_rate = self.get_base_rate(age, gender, self.medical_insurance_base_rates)
        
        # 医療保険は健康状態の影響が大きい
        health_weight = 1.5
        adjusted_risk = (
            risk_factors['health_risk'] ** health_weight *
            risk_factors['occupation_risk'] *
            risk_factors['lifestyle_risk'] *
            risk_factors['regional_risk']
        )
        
        premium = base_rate * adjusted_risk
        
        return premium
    
    def calculate_auto_insurance_premium(self, age, gender, risk_factors, customer_info):
        """
        自動車保険料の計算
        """
        # 基準保険料（年齢による）
        if age < 25:
            base_rate = 120000  # 若年層は高額
        elif age < 35:
            base_rate = 80000   # 中堅層
        elif age < 50:
            base_rate = 60000   # 最も安全な年齢層
        elif age < 65:
            base_rate = 70000   # やや増加
        else:
            base_rate = 90000   # 高齢層は増加
        
        # 性別による調整
        gender_factor = 0.95 if gender == '女性' else 1.00
        
        # 職業・地域による調整
        occupation_factor = self.occupation_risk_factors.get(
            customer_info.get('occupation', '会社員'), 1.00
        )
        
        # 運転歴による調整（サンプル）
        driving_experience = customer_info.get('driving_years', age - 18)
        if driving_experience < 3:
            experience_factor = 1.30
        elif driving_experience < 10:
            experience_factor = 1.10
        else:
            experience_factor = 0.90
        
        premium = (
            base_rate * 
            gender_factor * 
            occupation_factor * 
            experience_factor * 
            risk_factors['regional_risk']
        )
        
        return premium
    
    def calculate_disability_insurance_premium(self, age, gender, risk_factors, income):
        """
        就業不能保険料の計算
        """
        # 年収に基づく基準保険料（年収の0.5-2%程度）
        income_rate = min(0.02, max(0.005, 100000 / income))  # 年収に反比例
        base_rate = income * income_rate
        
        # 職業リスクの影響が大きい
        occupation_weight = 1.8
        adjusted_risk = (
            (risk_factors['occupation_risk'] ** occupation_weight) *
            risk_factors['health_risk'] *
            risk_factors['age_risk']
        )
        
        premium = base_rate * adjusted_risk
        
        return premium
    
    def get_base_rate(self, age, gender, rate_table):
        """
        年齢・性別による基準料率の取得
        """
        gender_rates = rate_table.get(gender, rate_table['男性'])
        
        for age_range, rate in gender_rates.items():
            if age_range[0] <= age < age_range[1]:
                return rate
        
        # 範囲外の場合は最高年齢層の料率を適用
        return max(gender_rates.values())
    
    def calculate_discounts(self, customer_info, risk_factors):
        """
        各種割引制度の計算
        """
        discounts = {}
        
        # 健康優良割引
        if risk_factors['health_risk'] <= 0.90:
            discounts['life'] = 0.10  # 生命保険10%割引
            discounts['medical'] = 0.15  # 医療保険15%割引
        
        # 無事故割引（自動車保険）
        accident_history = customer_info.get('accident_count', 0)
        if accident_history == 0:
            discounts['auto'] = 0.20  # 20%割引
        elif accident_history == 1:
            discounts['auto'] = 0.10  # 10%割引
        
        # 職業安定割引
        stable_occupations = ['公務員', '医師', '教師']
        if customer_info.get('occupation') in stable_occupations:
            discounts['disability'] = 0.15  # 就業不能保険15%割引
        
        # セット割引（複数保険加入）
        if len(discounts) >= 2:
            for key in discounts:
                discounts[key] = min(discounts[key] + 0.05, 0.30)  # 最大30%まで
        
        return discounts
    
    def generate_recommendations(self, customer_info, premiums, risk_factors):
        """
        保険料最適化の推奨事項生成
        """
        recommendations = []
        
        # 高リスク要因に対する推奨事項
        if risk_factors['health_risk'] > 1.20:
            recommendations.append(
                "健康状態の改善により保険料を削減できる可能性があります。定期健診の受診をお勧めします。"
            )
        
        if risk_factors['lifestyle_risk'] > 1.20:
            recommendations.append(
                "生活習慣の改善により保険料削減が期待できます。禁煙・運動習慣の開始を検討してください。"
            )
        
        # 保険料負担に関する推奨事項
        total_premium = premiums.get('total', 0)
        income = customer_info.get('income', 5000000)
        premium_ratio = total_premium / income
        
        if premium_ratio > 0.15:  # 年収の15%超
            recommendations.append(
                "保険料負担が年収の15%を超えています。保障内容の見直しを推奨します。"
            )
        elif premium_ratio < 0.05:  # 年収の5%未満
            recommendations.append(
                "保険料負担に余裕があります。保障の充実を検討されてはいかがでしょうか。"
            )
        
        # 年齢に応じた推奨事項
        age = customer_info.get('age', 40)
        if age < 30:
            recommendations.append(
                "若年層の方は将来の保険料上昇に備え、終身保険の検討をお勧めします。"
            )
        elif age > 50:
            recommendations.append(
                "医療保険の充実や介護保険の検討をお勧めします。"
            )
        
        return recommendations
    
    def compare_insurance_plans(self, customer_info, plan_variations):
        """
        複数の保険プランの比較
        
        Args:
            customer_info (dict): 顧客情報
            plan_variations (list): プランのバリエーション
            
        Returns:
            dict: プラン比較結果
        """
        print("📊 保険プランの比較分析を実行中...")
        
        comparison_results = []
        
        for plan in plan_variations:
            # 顧客情報にプラン固有の設定を適用
            modified_customer_info = {**customer_info, **plan}
            
            # 保険料計算
            premium_result = self.calculate_premium(modified_customer_info)
            
            # プラン情報を追加
            plan_result = {
                'plan_name': plan.get('plan_name', 'カスタムプラン'),
                'total_premium': premium_result['total'],
                'individual_premiums': {
                    k: v for k, v in premium_result.items() 
                    if k not in 
