"""
患者リスク評価システム
患者の基本情報と既往歴から、医療リスクを総合的に評価

機能:
- 年齢、BMI、生活習慣などの多角的リスク評価
- 既往歴や家族歴の考慮
- リスクスコアの数値化と分類
- 個別化されたリスク軽減推奨事項の提供
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, date
import logging

class PatientRiskAssessment:
    """患者リスク評価システムのメインクラス"""
    
    def __init__(self):
        """リスク評価システムの初期化"""
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        # リスクファクター定義の読み込み
        self._load_risk_factors()
        self._load_disease_risk_weights()
        self._load_demographic_adjustments()
        
        self.logger.info("リスク評価システムが初期化されました")
    
    def _load_risk_factors(self):
        """リスクファクターの定義を読み込み"""
        self.age_risk_categories = {
            'very_low': (0, 30),      # 極低リスク: 0-30歳
            'low': (31, 45),          # 低リスク: 31-45歳
            'moderate': (46, 60),     # 中リスク: 46-60歳
            'high': (61, 75),         # 高リスク: 61-75歳
            'very_high': (76, 120)    # 極高リスク: 76歳以上
        }
        
        self.bmi_risk_categories = {
            'underweight': (0, 18.4),      # 低体重
            'normal': (18.5, 24.9),        # 正常
            'overweight': (25.0, 29.9),    # 過体重
            'obese_1': (30.0, 34.9),       # 肥満度1
            'obese_2': (35.0, 39.9),       # 肥満度2
            'obese_3': (40.0, 60.0)        # 肥満度3（重度肥満）
        }
        
        self.smoking_risk_levels = {
            'never_smoker': 0.0,           # 非喫煙者
            'ex_smoker_recent': 0.4,       # 禁煙1年未満
            'ex_smoker_long': 0.2,         # 禁煙1年以上
            'light_smoker': 0.5,           # 軽度喫煙（10本/日未満）
            'moderate_smoker': 0.7,        # 中程度喫煙（10-20本/日）
            'heavy_smoker': 0.9            # 重度喫煙（20本/日以上）
        }
        
        self.alcohol_risk_levels = {
            'none': 0.0,                   # 飲酒なし
            'light': 0.1,                  # 軽度（週1-2日）
            'moderate': 0.2,               # 中程度（週3-4日）
            'heavy': 0.6,                  # 重度（毎日）
            'excessive': 0.8               # 過度（大量毎日）
        }
        
        self.exercise_risk_levels = {
            'very_active': -0.2,           # 非常に活発（週5日以上）
            'active': -0.1,                # 活発（週3-4日）
            'moderate': 0.0,               # 中程度（週1-2日）
            'sedentary': 0.3,              # 運動不足
            'inactive': 0.5                # 運動なし
        }
    
    def _load_disease_risk_weights(self):
        """疾患のリスク重み付けを定義"""
        self.chronic_disease_weights = {
            # 心血管系疾患（高リスク）
            'hypertension': 0.25,          # 高血圧
            'diabetes': 0.30,              # 糖尿病
            'heart_disease': 0.35,         # 心疾患
            'stroke': 0.40,                # 脳卒中
            'peripheral_artery_disease': 0.28,  # 末梢動脈疾患
            
            # 呼吸器系疾患
            'asthma': 0.15,                # 喘息
            'copd': 0.30,                  # 慢性閉塞性肺疾患
            'sleep_apnea': 0.20,           # 睡眠時無呼吸症候群
            
            # 消化器系疾患
            'liver_disease': 0.25,         # 肝疾患
            'kidney_disease': 0.35,        # 腎疾患
            'inflammatory_bowel_disease': 0.18,  # 炎症性腸疾患
            
            # 内分泌系疾患
            'thyroid_disease': 0.12,       # 甲状腺疾患
            'adrenal_disease': 0.20,       # 副腎疾患
            
            # 精神系疾患
            'depression': 0.15,            # うつ病
            'anxiety_disorder': 0.10,      # 不安障害
            
            # がん関連
            'cancer_history': 0.40,        # がん既往歴
            'cancer_active': 0.60,         # 現在治療中のがん
            
            # 自己免疫疾患
            'rheumatoid_arthritis': 0.22,  # 関節リウマチ
            'lupus': 0.28,                 # 全身性エリテマトーデス
            
            # その他
            'osteoporosis': 0.12,          # 骨粗鬆症
            'obesity': 0.20                # 肥満症
        }
        
        self.family_history_weights = {
            'heart_disease': 0.15,         # 心疾患の家族歴
            'diabetes': 0.12,              # 糖尿病の家族歴
            'cancer': 0.18,                # がんの家族歴
            'stroke': 0.15,                # 脳卒中の家族歴
            'hypertension': 0.10,          # 高血圧の家族歴
            'mental_illness': 0.08         # 精神疾患の家族歴
        }
    
    def _load_demographic_adjustments(self):
        """人口統計学的調整因子を定義"""
        self.gender_risk_adjustments = {
            'male': {
                'heart_disease': 1.2,      # 男性は心疾患リスクが高い
                'stroke': 1.1,
                'diabetes': 1.0,
                'cancer': 1.1
            },
            'female': {
                'heart_disease': 0.8,      # 女性は閉経前は心疾患リスクが低い
                'stroke': 0.9,
                'diabetes': 1.0,
                'cancer': 0.9,
                'osteoporosis': 1.3        # 女性は骨粗鬆症リスクが高い
            }
        }
        
        # 年齢による疾患リスク調整
        self.age_disease_multipliers = {
            'heart_disease': {
                (0, 40): 0.3,
                (41, 55): 0.7,
                (56, 70): 1.2,
                (71, 120): 1.8
            },
            'diabetes': {
                (0, 30): 0.4,
                (31, 50): 0.8,
                (51, 65): 1.3,
                (66, 120): 1.6
            },
            'cancer': {
                (0, 35): 0.2,
                (36, 55): 0.6,
                (56, 70): 1.4,
                (71, 120): 2.0
            }
        }
    
    def calculate_risk_score(self, patient_data: Dict[str, Any]) -> float:
        """
        患者の総合リスクスコアを計算
        
        Args:
            patient_data: 患者情報辞書
                - age: 年齢
                - gender: 性別 ('male'/'female')
                - bmi: BMI値
                - smoking_status: 喫煙状況
                - alcohol_consumption: 飲酒状況
                - exercise_level: 運動レベル
                - chronic_diseases: 既往歴リスト
                - family_history: 家族歴リスト
                - medications: 服薬リスト（オプション）
        
        Returns:
            リスクスコア（0-100）
        """
        try:
            total_risk_score = 0.0
            
            # 1. 年齢リスク（重み: 25%）
            age_risk = self._calculate_age_risk(patient_data.get('age', 0))
            total_risk_score += age_risk * 0.25
            
            # 2. BMIリスク（重み: 15%）
            bmi_risk = self._calculate_bmi_risk(patient_data.get('bmi', 22))
            total_risk_score += bmi_risk * 0.15
            
            # 3. 生活習慣リスク（重み: 30%）
            lifestyle_risk = self._calculate_lifestyle_risk(patient_data)
            total_risk_score += lifestyle_risk * 0.30
            
            # 4. 既往歴リスク（重み: 20%）
            medical_history_risk = self._calculate_medical_history_risk(
                patient_data.get('chronic_diseases', [])
            )
            total_risk_score += medical_history_risk * 0.20
            
            # 5. 家族歴リスク（重み: 10%）
            family_risk = self._calculate_family_history_risk(
                patient_data.get('family_history', [])
            )
            total_risk_score += family_risk * 0.10
            
            # 6. 性別・年齢による調整
            adjusted_risk = self._apply_demographic_adjustments(
                total_risk_score, patient_data
            )
            
            # 最終スコアを0-100の範囲に正規化
            final_score = max(0.0, min(100.0, adjusted_risk))
            
            self.logger.info(f"リスクスコア計算完了: {final_score:.1f}")
            return final_score
            
        except Exception as e:
            self.logger.error(f"リスクスコア計算エラー: {str(e)}")
            return 50.0  # エラー時はデフォルト値を返す
    
    def _calculate_age_risk(self, age: int) -> float:
        """年齢リスクを計算"""
        for risk_level, (min_age, max_age) in self.age_risk_categories.items():
            if min_age <= age <= max_age:
                risk_mapping = {
                    'very_low': 5.0,
                    'low': 15.0,
                    'moderate': 35.0,
                    'high': 60.0,
                    'very_high': 85.0
                }
                return risk_mapping.get(risk_level, 35.0)
        
        return 35.0  # デフォルト値
    
    def _calculate_bmi_risk(self, bmi: float) -> float:
        """BMIリスクを計算"""
        for risk_level, (min_bmi, max_bmi) in self.bmi_risk_categories.items():
            if min_bmi <= bmi <= max_bmi:
                risk_mapping = {
                    'underweight': 30.0,    # 低体重もリスク
                    'normal': 10.0,
                    'overweight': 25.0,
                    'obese_1': 45.0,
                    'obese_2': 65.0,
                    'obese_3': 85.0
                }
                return risk_mapping.get(risk_level, 25.0)
        
        return 25.0  # デフォルト値
    
    def _calculate_lifestyle_risk(self, patient_data: Dict) -> float:
        """生活習慣リスクを計算"""
        lifestyle_risk = 0.0
        
        # 喫煙リスク（重み: 40%）
        smoking_status = patient_data.get('smoking_status', 'never_smoker')
        smoking_risk = self.smoking_risk_levels.get(smoking_status, 0.0)
        lifestyle_risk += smoking_risk * 40.0
        
        # 飲酒リスク（重み: 25%）
        alcohol_consumption = patient_data.get('alcohol_consumption', 'none')
        alcohol_risk = self.alcohol_risk_levels.get(alcohol_consumption, 0.0)
        lifestyle_risk += alcohol_risk * 25.0
        
        # 運動不足リスク（重み: 35%）
        exercise_level = patient_data.get('exercise_level', 'moderate')
        exercise_risk = self.exercise_risk_levels.get(exercise_level, 0.0)
        # 運動は負のリスク（保護因子）なので、正の値に変換
        if exercise_risk < 0:
            lifestyle_risk += abs(exercise_risk) * 35.0 * (-1)  # 運動によるリスク軽減
        else:
            lifestyle_risk += exercise_risk * 35.0
        
        return max(0.0, lifestyle_risk)
    
    def _calculate_medical_history_risk(self, chronic_diseases: List[str]) -> float:
        """既往歴リスクを計算"""
        if not chronic_diseases:
            return 0.0
        
        medical_risk = 0.0
        disease_count = len(chronic_diseases)
        
        for disease in chronic_diseases:
            disease_weight = self.chronic_disease_weights.get(disease, 0.1)
            medical_risk += disease_weight * 100
        
        # 複数疾患による相乗効果を考慮
        if disease_count > 1:
            multiplier = 1.0 + (disease_count - 1) * 0.2  # 疾患が増えるごとに20%増加
            medical_risk *= multiplier
        
        return min(medical_risk, 80.0)  # 上限80点
    
    def _calculate_family_history_risk(self, family_history: List[str]) -> float:
        """家族歴リスクを計算"""
        if not family_history:
            return 0.0
        
        family_risk = 0.0
        
        for condition in family_history:
            condition_weight = self.family_history_weights.get(condition, 0.05)
            family_risk += condition_weight * 100
        
        return min(family_risk, 30.0)  # 上限30点
    
    def _apply_demographic_adjustments(self, base_risk: float, patient_data: Dict) -> float:
        """性別・年齢による人口統計学的調整を適用"""
        try:
            gender = patient_data.get('gender', 'unknown')
            age = patient_data.get('age', 0)
            
            if gender not in ['male', 'female']:
                return base_risk
            
            # 主要疾患リスクの性別調整（簡略化）
            gender_adjustments = self.gender_risk_adjustments.get(gender, {})
            
            # 心疾患リスクの調整例
            heart_disease_adjustment = gender_adjustments.get('heart_disease', 1.0)
            
            # 年齢と性別の複合調整（簡略化）
            demographic_multiplier = 1.0
            
            if gender == 'male' and age > 45:
                demographic_multiplier = 1.1  # 中年男性のリスク増加
            elif gender == 'female' and age > 55:
                demographic_multiplier = 1.15  # 閉経後女性のリスク増加
            
            adjusted_risk = base_risk * demographic_multiplier
            
            return adjusted_risk
            
        except Exception as e:
            self.logger.error(f"人口統計学的調整エラー: {str(e)}")
            return base_risk
    
    def get_risk_breakdown(self, patient_data: Dict[str, Any]) -> Dict[str, float]:
        """リスクの内訳詳細を取得"""
        try:
            breakdown = {}
            
            # 各要因別のリスクスコア
            breakdown['age_risk'] = self._calculate_age_risk(patient_data.get('age', 0))
            breakdown['bmi_risk'] = self._calculate_bmi_risk(patient_data.get('bmi', 22))
            breakdown['lifestyle_risk'] = self._calculate_lifestyle_risk(patient_data)
            breakdown['medical_history_risk'] = self._calculate_medical_history_risk(
                patient_data.get('chronic_diseases', [])
            )
            breakdown['family_history_risk'] = self._calculate_family_history_risk(
                patient_data.get('family_history', [])
            )
            
            # 各要因の重み付き寄与度
            breakdown['weighted_contributions'] = {
                'age': breakdown['age_risk'] * 0.25,
                'bmi': breakdown['bmi_risk'] * 0.15,
                'lifestyle': breakdown['lifestyle_risk'] * 0.30,
                'medical_history': breakdown['medical_history_risk'] * 0.20,
                'family_history': breakdown['family_history_risk'] * 0.10
            }
            
            return breakdown
            
        except Exception as e:
            self.logger.error(f"リスク内訳計算エラー: {str(e)}")
            return {}
    
    def categorize_risk_level(self, risk_score: float) -> Dict[str, Any]:
        """リスクスコアをカテゴリに分類"""
        if risk_score < 20:
            level = 'very_low'
            description = '極低リスク'
            color = 'green'
        elif risk_score < 35:
            level = 'low'
            description = '低リスク'
            color = 'light_green'
        elif risk_score < 50:
            level = 'moderate'
            description = '中リスク'
            color = 'yellow'
        elif risk_score < 70:
            level = 'high'
            description = '高リスク'
            color = 'orange'
        else:
            level = 'very_high'
            description = '極高リスク'
            color = 'red'
        
        return {
            'level': level,
            'description': description,
            'color_code': color,
            'score_range': self._get_score_range(level),
            'percentile': self._calculate_percentile(risk_score)
        }
    
    def _get_score_range(self, level: str) -> Tuple[int, int]:
        """リスクレベルに対応するスコア範囲を取得"""
        ranges = {
            'very_low': (0, 19),
            'low': (20, 34),
            'moderate': (35, 49),
            'high': (50, 69),
            'very_high': (70, 100)
        }
        return ranges.get(level, (0, 100))
    
    def _calculate_percentile(self, risk_score: float) -> int:
        """同年代との比較パーセンタイルを計算（簡略化）"""
        # 実際の実装では年齢・性別別の統計データを使用
        # ここでは簡略化した計算
        percentile = min(100, max(1, int(risk_score * 1.2)))
        return percentile
    
    def generate_risk_recommendations(self, patient_data: Dict[str, Any], 
                                    risk_score: float) -> List[Dict[str, Any]]:
        """個別化されたリスク軽減推奨事項を生成"""
        try:
            recommendations = []
            risk_breakdown = self.get_risk_breakdown(patient_data)
            
            # 年齢リスクに対する推奨事項
            if risk_breakdown.get('age_risk', 0) > 50:
                recommendations.append({
                    'category': 'age_related',
                    'priority': 'high',
                    'recommendation': '年齢に伴うリスク増加に対して、定期的な健康診断と予防医療を重視してください',
                    'specific_actions': [
                        '年1回以上の包括的健康診断',
                        '専門医による定期的な検査',
                        '予防接種の適切な接種'
                    ]
                })
            
            # BMIリスクに対する推奨事項
            bmi = patient_data.get('bmi', 22)
            if bmi > 30:
                recommendations.append({
                    'category': 'weight_management',
                    'priority': 'high',
                    'recommendation': '肥満による健康リスクを軽減するため、体重管理が重要です',
                    'specific_actions': [
                        '栄養士による食事指導',
                        '段階的な運動プログラム',
                        '体重減少目標：月1-2kg'
                    ]
                })
            elif bmi < 18.5:
                recommendations.append({
                    'category': 'weight_management',
                    'priority': 'medium',
                    'recommendation': '低体重による健康リスクがあります。適正体重の維持を目指してください',
                    'specific_actions': [
                        '栄養バランスの改善',
                        '筋力トレーニングの導入',
                        '定期的な体重モニタリング'
                    ]
                })
            
            # 喫煙リスクに対する推奨事項
            smoking_status = patient_data.get('smoking_status', 'never_smoker')
            if smoking_status in ['light_smoker', 'moderate_smoker', 'heavy_smoker']:
                recommendations.append({
                    'category': 'smoking_cessation',
                    'priority': 'critical',
                    'recommendation': '喫煙は多くの疾患の最大のリスクファクターです。禁煙を強く推奨します',
                    'specific_actions': [
                        '禁煙外来の受診',
                        'ニコチン代替療法の検討',
                        '禁煙支援プログラムへの参加'
                    ]
                })
            
            # 運動不足に対する推奨事項
            exercise_level = patient_data.get('exercise_level', 'moderate')
            if exercise_level in ['sedentary', 'inactive']:
                recommendations.append({
                    'category': 'physical_activity',
                    'priority': 'high',
                    'recommendation': '定期的な運動は多くの健康リスクを軽減します',
                    'specific_actions': [
                        '週150分以上の中強度運動',
                        '段階的な運動量増加',
                        '筋力トレーニング週2回以上'
                    ]
                })
            
            # 既往歴に基づく推奨事項
            chronic_diseases = patient_data.get('chronic_diseases', [])
            if 'diabetes' in chronic_diseases:
                recommendations.append({
                    'category': 'diabetes_management',
                    'priority': 'critical',
                    'recommendation': '糖尿病の適切な管理が重要です',
                    'specific_actions': [
                        '血糖値の定期的なモニタリング',
                        '糖尿病専門医による管理',
                        '合併症予防のための定期検査'
                    ]
                })
            
            if 'hypertension' in chronic_diseases:
                recommendations.append({
                    'category': 'hypertension_management',
                    'priority': 'high',
                    'recommendation': '高血圧の適切な管理が必要です',
                    'specific_actions': [
                        '家庭血圧測定の実施',
                        '減塩食の実践',
                        '降圧薬の適切な服用'
                    ]
                })
            
            # 家族歴に基づく推奨事項
            family_history = patient_data.get('family_history', [])
            if 'heart_disease' in family_history:
                recommendations.append({
                    'category': 'cardiovascular_prevention',
                    'priority': 'medium',
                    'recommendation': '心疾患の家族歴があるため、心血管疾患の予防に注意してください',
                    'specific_actions': [
                        '年1回の心電図検査',
                        'コレステロール値の定期チェック',
                        '心臓に優しい食事の実践'
                    ]
                })
            
            # リスクレベル全体に基づく推奨事項
            if risk_score > 70:
                recommendations.append({
                    'category': 'general_high_risk',
                    'priority': 'critical',
                    'recommendation': '総合的なリスクが高いため、包括的な健康管理が必要です',
                    'specific_actions': [
                        '月1回以上の医師との相談',
                        '緊急時連絡先の整備',
                        '家族・周囲への状況共有'
                    ]
                })
            
            # 優先度順にソート
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"推奨事項生成エラー: {str(e)}")
            return []
    
    def generate_risk_report(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """包括的なリスク評価レポートを生成"""
        try:
            # リスクスコア計算
            risk_score = self.calculate_risk_score(patient_data)
            
            # リスク分類
            risk_category = self.categorize_risk_level(risk_score)
            
            # リスク内訳
            risk_breakdown = self.get_risk_breakdown(patient_data)
            
            # 推奨事項生成
            recommendations = self.generate_risk_recommendations(patient_data, risk_score)
            
            # レポート作成
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'patient_id': patient_data.get('patient_id', 'unknown'),
                'risk_assessment': {
                    'total_risk_score': risk_score,
                    'risk_category': risk_category,
                    'risk_breakdown': risk_breakdown
                },
                'recommendations': recommendations,
                'follow_up_schedule': self._generate_follow_up_schedule(risk_score),
                'emergency_indicators': self._identify_emergency_indicators(patient_data, risk_score),
                'report_summary': self._generate_risk_summary(risk_score, risk_category, patient_data)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"リスクレポート生成エラー: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _generate_follow_up_schedule(self, risk_score: float) -> Dict[str, str]:
        """フォローアップスケジュールを生成"""
        if risk_score > 70:
            return {
                'next_assessment': '1ヶ月後',
                'routine_checkup': '3ヶ月ごと',
                'emergency_review': '症状変化時は即座に'
            }
        elif risk_score > 50:
            return {
                'next_assessment': '3ヶ月後',
                'routine_checkup': '6ヶ月ごと',
                'emergency_review': '重大な症状変化時'
            }
        else:
            return {
                'next_assessment': '6ヶ月後',
                'routine_checkup': '年1回',
                'emergency_review': '緊急症状時のみ'
            }
    
    def _identify_emergency_indicators(self, patient_data: Dict, risk_score: float) -> List[str]:
        """緊急対応が必要な指標を特定"""
        emergency_indicators = []
        
        chronic_diseases = patient_data.get('chronic_diseases', [])
        
        if risk_score > 80:
            emergency_indicators.append('極高リスクレベル - 即座の医学的評価が必要')
        
        if 'cancer_active' in chronic_diseases:
            emergency_indicators.append('活動性がん - 腫瘍専門医との密接な連携が必要')
        
        if 'heart_disease' in chronic_diseases and risk_score > 60:
            emergency_indicators.append('心疾患既往+高リスク - 心血管系緊急事態に注意')
        
        if patient_data.get('age', 0) > 75 and risk_score > 50:
            emergency_indicators.append('高齢+中等度以上リスク - 急速な健康状態変化に注意')
        
        return emergency_indicators
    
    def _generate_risk_summary(self, risk_score: float, risk_category: Dict, 
                             patient_data: Dict) -> str:
        """リスク評価のサマリーを生成"""
        patient_age = patient_data.get('age', 0)
        risk_level = risk_category['description']
        
        summary = f"患者（{patient_age}歳）の総合医療リスクスコアは{risk_score:.1f}点で、"
        summary += f"{risk_level}に分類されます。"
        
        # 主要なリスクファクターを特定
        risk_breakdown = self.get_risk_breakdown(patient_data)
        max_risk_factor = max(risk_breakdown['weighted_contributions'].items(), 
                             key=lambda x: x[1])
        
        factor_names = {
            'age': '年齢',
            'bmi': 'BMI',
            'lifestyle': '生活習慣',
            'medical_history': '既往歴',
            'family_history': '家族歴'
        }
        
        primary_factor = factor_names.get(max_risk_factor[0], '不明')
        summary += f" 主要なリスクファクターは{primary_factor}です。"
        
        if risk_score > 50:
            summary += " 積極的なリスク管理と定期的な医学的監視が推奨されます。"
        else:
            summary += " 現在のライフスタイルの維持と定期的な健康チェックを継続してください。"
        
        return summary

# 使用例とテスト用のメイン関数
if __name__ == "__main__":
    # リスク評価システムの初期化
    risk_assessor = PatientRiskAssessment()
    
    # テスト用患者データ
    test_patient = {
        'patient_id': 'P001',
        'age': 58,
        'gender': 'male',
        'bmi': 29.5,
        'smoking_status': 'ex_smoker_long',
        'alcohol_consumption': 'moderate',
        'exercise_level': 'sedentary',
        'chronic_diseases': ['hypertension', 'diabetes'],
        'family_history': ['heart_disease', 'stroke']
    }
    
    print("=== 患者リスク評価システム デモ ===")
    
    # 包括的リスクレポートの生成
    report = risk_assessor.generate_risk_report(test_patient)
    
    print(f"\nレポート生成時刻: {report['report_timestamp']}")
    print(f"患者ID: {report['patient_id']}")
    
    # リスク評価結果
    risk_assessment = report['risk_assessment']
    print(f"\n【リスク評価結果】")
    print(f"総合リスクスコア: {risk_assessment['total_risk_score']:.1f}")
    print(f"リスクカテゴリ: {risk_assessment['risk_category']['description']}")
    
    # リスク内訳
    print(f"\n【リスク内訳】")
    breakdown = risk_assessment['risk_breakdown']['weighted_contributions']
    for factor, score in breakdown.items():
        print(f"  - {factor}: {score:.1f}点")
    
    # 推奨事項
    print(f"\n【推奨事項】（上位3件）")
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"{i}. {rec['recommendation']} (優先度: {rec['priority']})")
    
    # サマリー
    print(f"\n【サマリー】")
    print(report['report_summary'])
    
    # フォローアップスケジュール
    print(f"\n【フォローアップスケジュール】")
    schedule = report['follow_up_schedule']
    for item, timing in schedule.items():
        print(f"  - {item}: {timing}")
