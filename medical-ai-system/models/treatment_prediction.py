"""
治療効果予測システム
患者の診断結果とリスクプロファイルに基づいて最適な治療法を予測

機能:
- 疾患別治療選択肢データベース
- 個別患者のリスクを考慮した治療効果予測
- 治療期間と成功率の推定
- 治療法の推奨度ランキング
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

class TreatmentEffectPredictor:
    """治療効果予測システムのメインクラス"""
    
    def __init__(self):
        """治療効果予測システムの初期化"""
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        # 治療データベースの読み込み
        self._load_treatment_database()
        self._load_drug_interactions()
        self._load_contraindications()
        
        self.logger.info("治療効果予測システムが初期化されました")
    
    def _load_treatment_database(self):
        """治療法データベースの読み込み"""
        self.treatment_database = {
            # 風邪 (Common Cold)
            'common_cold': {
                'rest_and_fluids': {
                    'success_rate': 0.85,
                    'typical_duration': 7,
                    'side_effects_risk': 0.0,
                    'cost_level': 'low',
                    'monitoring_required': False,
                    'description': '安静と水分補給'
                },
                'symptomatic_treatment': {
                    'success_rate': 0.78,
                    'typical_duration': 5,
                    'side_effects_risk': 0.1,
                    'cost_level': 'low',
                    'monitoring_required': False,
                    'description': '症状緩和薬'
                },
                'vitamin_c_zinc': {
                    'success_rate': 0.72,
                    'typical_duration': 6,
                    'side_effects_risk': 0.05,
                    'cost_level': 'low',
                    'monitoring_required': False,
                    'description': 'ビタミンC・亜鉛補給'
                }
            },
            
            # インフルエンザ (Influenza)
            'influenza': {
                'antiviral_therapy': {
                    'success_rate': 0.88,
                    'typical_duration': 7,
                    'side_effects_risk': 0.15,
                    'cost_level': 'medium',
                    'monitoring_required': True,
                    'description': '抗ウイルス薬療法'
                },
                'supportive_care': {
                    'success_rate': 0.75,
                    'typical_duration': 10,
                    'side_effects_risk': 0.05,
                    'cost_level': 'low',
                    'monitoring_required': False,
                    'description': '対症療法'
                },
                'oseltamivir': {
                    'success_rate': 0.85,
                    'typical_duration': 6,
                    'side_effects_risk': 0.2,
                    'cost_level': 'medium',
                    'monitoring_required': True,
                    'description': 'オセルタミビル'
                }
            },
            
            # 肺炎 (Pneumonia)
            'pneumonia': {
                'antibiotic_therapy': {
                    'success_rate': 0.92,
                    'typical_duration': 14,
                    'side_effects_risk': 0.25,
                    'cost_level': 'medium',
                    'monitoring_required': True,
                    'description': '抗生物質療法'
                },
                'hospitalization': {
                    'success_rate': 0.95,
                    'typical_duration': 10,
                    'side_effects_risk': 0.1,
                    'cost_level': 'high',
                    'monitoring_required': True,
                    'description': '入院治療'
                },
                'oxygen_therapy': {
                    'success_rate': 0.88,
                    'typical_duration': 12,
                    'side_effects_risk': 0.05,
                    'cost_level': 'medium',
                    'monitoring_required': True,
                    'description': '酸素療法'
                }
            },
            
            # 高血圧 (Hypertension)
            'hypertension': {
                'ace_inhibitors': {
                    'success_rate': 0.82,
                    'typical_duration': 365,  # 慢性疾患のため長期間
                    'side_effects_risk': 0.15,
                    'cost_level': 'medium',
                    'monitoring_required': True,
                    'description': 'ACE阻害薬'
                },
                'lifestyle_modification': {
                    'success_rate': 0.65,
                    'typical_duration': 180,
                    'side_effects_risk': 0.0,
                    'cost_level': 'low',
                    'monitoring_required': True,
                    'description': '生活習慣改善'
                },
                'diuretics': {
                    'success_rate': 0.78,
                    'typical_duration': 365,
                    'side_effects_risk': 0.2,
                    'cost_level': 'low',
                    'monitoring_required': True,
                    'description': '利尿薬'
                },
                'calcium_channel_blockers': {
                    'success_rate': 0.80,
                    'typical_duration': 365,
                    'side_effects_risk': 0.18,
                    'cost_level': 'medium',
                    'monitoring_required': True,
                    'description': 'カルシウム拮抗薬'
                }
            },
            
            # 糖尿病 (Diabetes)
            'diabetes': {
                'metformin': {
                    'success_rate': 0.85,
                    'typical_duration': 365,
                    'side_effects_risk': 0.2,
                    'cost_level': 'low',
                    'monitoring_required': True,
                    'description': 'メトホルミン'
                },
                'insulin_therapy': {
                    'success_rate': 0.92,
                    'typical_duration': 365,
                    'side_effects_risk': 0.3,
                    'cost_level': 'high',
                    'monitoring_required': True,
                    'description': 'インスリン療法'
                },
                'lifestyle_diet_control': {
                    'success_rate': 0.70,
                    'typical_duration': 365,
                    'side_effects_risk': 0.0,
                    'cost_level': 'low',
                    'monitoring_required': True,
                    'description': '食事・運動療法'
                },
                'combination_therapy': {
                    'success_rate': 0.88,
                    'typical_duration': 365,
                    'side_effects_risk': 0.25,
                    'cost_level': 'medium',
                    'monitoring_required': True,
                    'description': '併用療法'
                }
            },
            
            # 偏頭痛 (Migraine)
            'migraine': {
                'nsaid_therapy': {
                    'success_rate': 0.75,
                    'typical_duration': 1,  # 発作時のみ
                    'side_effects_risk': 0.15,
                    'cost_level': 'low',
                    'monitoring_required': False,
                    'description': 'NSAID療法'
                },
                'triptan_therapy': {
                    'success_rate': 0.85,
                    'typical_duration': 1,
                    'side_effects_risk': 0.2,
                    'cost_level': 'high',
                    'monitoring_required': True,
                    'description': 'トリプタン療法'
                },
                'preventive_medication': {
                    'success_rate': 0.65,
                    'typical_duration': 180,
                    'side_effects_risk': 0.25,
                    'cost_level': 'medium',
                    'monitoring_required': True,
                    'description': '予防薬'
                },
                'lifestyle_modification': {
                    'success_rate': 0.60,
                    'typical_duration': 90,
                    'side_effects_risk': 0.0,
                    'cost_level': 'low',
                    'monitoring_required': False,
                    'description': '生活習慣改善'
                }
            }
        }
        
        self.logger.info(f"治療データベースを読み込み完了: {len(self.treatment_database)}種類の疾患")
    
    def _load_drug_interactions(self):
        """薬物相互作用データベースの読み込み"""
        self.drug_interactions = {
            'ace_inhibitors': {
                'contraindicated_with': ['pregnancy', 'hyperkalemia'],
                'caution_with': ['kidney_disease', 'diabetes'],
                'interaction_drugs': ['potassium_supplements', 'nsaids']
            },
            'metformin': {
                'contraindicated_with': ['kidney_disease', 'liver_disease'],
                'caution_with': ['heart_failure', 'elderly'],
                'interaction_drugs': ['contrast_agents', 'alcohol']
            },
            'oseltamivir': {
                'contraindicated_with': ['severe_kidney_disease'],
                'caution_with': ['kidney_impairment'],
                'interaction_drugs': ['live_vaccines']
            },
            'triptan_therapy': {
                'contraindicated_with': ['heart_disease', 'stroke_history'],
                'caution_with': ['hypertension', 'elderly'],
                'interaction_drugs': ['maoi', 'ssri']
            }
        }
    
    def _load_contraindications(self):
        """禁忌データベースの読み込み"""
        self.contraindications = {
            'age_related': {
                'elderly': {  # 65歳以上
                    'high_risk_drugs': ['benzodiazepines', 'tricyclic_antidepressants'],
                    'dose_adjustments': ['kidney_excreted_drugs', 'sedatives']
                },
                'pediatric': {  # 18歳未満
                    'contraindicated': ['aspirin', 'tetracyclines'],
                    'special_consideration': ['dosing_by_weight']
                }
            },
            'condition_related': {
                'pregnancy': {
                    'contraindicated': ['ace_inhibitors', 'statins', 'warfarin'],
                    'preferred': ['methyldopa', 'insulin', 'acetaminophen']
                },
                'kidney_disease': {
                    'contraindicated': ['metformin', 'nsaids'],
                    'dose_adjustment': ['all_kidney_excreted_drugs']
                },
                'liver_disease': {
                    'contraindicated': ['acetaminophen_high_dose', 'statins'],
                    'monitoring_required': ['hepatotoxic_drugs']
                }
            }
        }
    
    def predict_treatment_outcome(self, disease: str, patient_risk_score: float, 
                                treatment_option: str, patient_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        治療効果を予測
        
        Args:
            disease: 疾患名
            patient_risk_score: 患者のリスクスコア
            treatment_option: 治療選択肢
            patient_data: 患者の詳細情報（オプション）
        
        Returns:
            予測結果辞書
        """
        try:
            if disease not in self.treatment_database:
                self.logger.warning(f"未知の疾患: {disease}")
                return None
            
            if treatment_option not in self.treatment_database[disease]:
                self.logger.warning(f"未知の治療法: {treatment_option}")
                return None
            
            base_data = self.treatment_database[disease][treatment_option]
            
            # 基本治療効果データを取得
            base_success_rate = base_data['success_rate']
            base_duration = base_data['typical_duration']
            side_effects_risk = base_data['side_effects_risk']
            cost_level = base_data['cost_level']
            monitoring_required = base_data['monitoring_required']
            
            # 患者リスクによる調整
            risk_adjustment = self._calculate_risk_adjustment(patient_risk_score)
            
            # 年齢による調整
            age_adjustment = 1.0
            if patient_data and 'age' in patient_data:
                age_adjustment = self._calculate_age_adjustment(
                    patient_data['age'], treatment_option
                )
            
            # 既往歴による調整
            comorbidity_adjustment = 1.0
            if patient_data and 'chronic_diseases' in patient_data:
                comorbidity_adjustment = self._calculate_comorbidity_adjustment(
                    patient_data['chronic_diseases'], treatment_option
                )
            
                       # 薬物相互作用チェック
            interaction_risk = 0.0
            if patient_data and 'medications' in patient_data:
                interaction_risk = self._check_drug_interactions(
                    treatment_option, patient_data['medications']
                )
            
            # 禁忌チェック
            contraindication_found = False
            if patient_data:
                contraindication_found = self._check_contraindications(
                    treatment_option, patient_data
                )
            
            # 最終的な成功率計算
            adjusted_success_rate = (
                base_success_rate * 
                risk_adjustment * 
                age_adjustment * 
                comorbidity_adjustment
            )
            
            # 副作用リスクの調整
            adjusted_side_effects_risk = side_effects_risk + interaction_risk
            
            # 治療期間の調整
            adjusted_duration = base_duration * (2.0 - risk_adjustment)
            
            # 推奨度スコアの計算
            recommendation_score = self._calculate_recommendation_score(
                adjusted_success_rate,
                adjusted_duration,
                adjusted_side_effects_risk,
                cost_level,
                contraindication_found
            )
            
            # 信頼度の計算
            confidence = self._calculate_prediction_confidence(
                patient_risk_score, patient_data
            )
            
            result = {
                'success_rate': round(max(0.1, min(1.0, adjusted_success_rate)), 3),
                'estimated_duration': round(max(1, adjusted_duration), 1),
                'side_effects_risk': round(min(1.0, adjusted_side_effects_risk), 3),
                'recommendation_score': round(max(0.0, min(1.0, recommendation_score)), 3),
                'confidence': confidence,
                'cost_level': cost_level,
                'monitoring_required': monitoring_required,
                'contraindications': contraindication_found,
                'interaction_warnings': interaction_risk > 0.1,
                'description': base_data['description'],
                'adjustments_applied': {
                    'risk_adjustment': risk_adjustment,
                    'age_adjustment': age_adjustment,
                    'comorbidity_adjustment': comorbidity_adjustment
                }
            }
            
            self.logger.info(f"治療予測完了: {treatment_option} - スコア: {recommendation_score:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"治療効果予測エラー: {str(e)}")
            return None
    
    def _calculate_risk_adjustment(self, risk_score: float) -> float:
        """リスクスコアに基づく治療効果調整係数を計算"""
        # リスクが高いほど治療効果が下がる
        # リスクスコア0-100を調整係数0.6-1.0に変換
        adjustment = 1.0 - (risk_score / 250)  # 最低0.6まで下がる
        return max(0.6, min(1.0, adjustment))
    
    def _calculate_age_adjustment(self, age: int, treatment_option: str) -> float:
        """年齢による治療効果調整係数を計算"""
        # 年齢による薬物代謝の変化を考慮
        if age < 18:
            # 小児: 代謝が早い
            return 0.9
        elif age < 65:
            # 成人: 標準
            return 1.0
        elif age < 80:
            # 高齢者: 代謝が遅い、副作用リスク増加
            return 0.85
        else:
            # 超高齢者: さらに代謝が遅い
            return 0.75
    
    def _calculate_comorbidity_adjustment(self, chronic_diseases: List[str], 
                                        treatment_option: str) -> float:
        """併存疾患による治療効果調整係数を計算"""
        adjustment = 1.0
        
        # 各併存疾患による影響
        disease_impacts = {
            'diabetes': 0.95,          # 糖尿病: 治癒遅延
            'hypertension': 0.98,      # 高血圧: 軽度影響
            'heart_disease': 0.90,     # 心疾患: 中等度影響
            'kidney_disease': 0.85,    # 腎疾患: 薬物排泄影響
            'liver_disease': 0.80,     # 肝疾患: 薬物代謝影響
            'cancer_active': 0.70,     # 活動性がん: 免疫低下
            'autoimmune_disease': 0.85 # 自己免疫疾患: 免疫系影響
        }
        
        for disease in chronic_diseases:
            if disease in disease_impacts:
                adjustment *= disease_impacts[disease]
        
        return max(0.5, adjustment)  # 最低0.5まで
    
    def _check_drug_interactions(self, treatment_option: str, 
                               current_medications: List[str]) -> float:
        """薬物相互作用による追加リスクを計算"""
        if treatment_option not in self.drug_interactions:
            return 0.0
        
        interaction_data = self.drug_interactions[treatment_option]
        additional_risk = 0.0
        
        # 相互作用薬物のチェック
        interaction_drugs = interaction_data.get('interaction_drugs', [])
        for medication in current_medications:
            if medication in interaction_drugs:
                additional_risk += 0.1  # 各相互作用につき10%リスク増加
        
        return min(0.5, additional_risk)  # 最大50%まで
    
    def _check_contraindications(self, treatment_option: str, 
                               patient_data: Dict) -> bool:
        """禁忌条件をチェック"""
        if treatment_option not in self.drug_interactions:
            return False
        
        interaction_data = self.drug_interactions[treatment_option]
        contraindicated_conditions = interaction_data.get('contraindicated_with', [])
        
        # 年齢による禁忌チェック
        age = patient_data.get('age', 0)
        if age >= 65 and treatment_option in ['benzodiazepines', 'tricyclic_antidepressants']:
            return True
        
        # 既往歴による禁忌チェック
        chronic_diseases = patient_data.get('chronic_diseases', [])
        for condition in contraindicated_conditions:
            if condition in chronic_diseases:
                return True
        
        # 妊娠による禁忌チェック
        if (patient_data.get('pregnancy', False) and 
            treatment_option in ['ace_inhibitors', 'statins']):
            return True
        
        return False
    
    def _calculate_recommendation_score(self, success_rate: float, duration: float,
                                      side_effects_risk: float, cost_level: str,
                                      contraindication: bool) -> float:
        """治療法の推奨度スコアを計算"""
        if contraindication:
            return 0.0  # 禁忌がある場合は推奨しない
        
        # 成功率の寄与（50%）
        success_component = success_rate * 0.5
        
        # 治療期間の寄与（20%）
        # 短期間ほど良い（ただし慢性疾患は除く）
        if duration > 300:  # 慢性疾患
            duration_component = 0.8 * 0.2  # 慢性疾患は期間評価を緩く
        else:
            duration_component = max(0.2, (30 - min(30, duration)) / 30) * 0.2
        
        # 副作用リスクの寄与（20%）
        side_effects_component = (1.0 - side_effects_risk) * 0.2
        
        # コストの寄与（10%）
        cost_scores = {'low': 1.0, 'medium': 0.7, 'high': 0.4}
        cost_component = cost_scores.get(cost_level, 0.7) * 0.1
        
        total_score = (success_component + duration_component + 
                      side_effects_component + cost_component)
        
        return total_score
    
    def _calculate_prediction_confidence(self, risk_score: float, 
                                       patient_data: Optional[Dict]) -> float:
        """予測の信頼度を計算"""
        base_confidence = 0.7
        
        # リスクスコアが中程度の範囲にあると信頼度が高い
        if 20 <= risk_score <= 60:
            risk_confidence_bonus = 0.1
        else:
            risk_confidence_bonus = -0.05
        
        # 患者データの完全性による信頼度調整
        data_completeness_bonus = 0.0
        if patient_data:
            required_fields = ['age', 'chronic_diseases', 'medications']
            available_fields = sum(1 for field in required_fields if field in patient_data)
            data_completeness_bonus = (available_fields / len(required_fields)) * 0.15
        
        final_confidence = base_confidence + risk_confidence_bonus + data_completeness_bonus
        return max(0.3, min(0.95, final_confidence))
    
    def get_available_treatments(self, disease: str) -> List[str]:
        """指定疾患で利用可能な治療選択肢を取得"""
        if disease in self.treatment_database:
            return list(self.treatment_database[disease].keys())
        return []
    
    def compare_treatments(self, disease: str, patient_risk_score: float,
                         patient_data: Optional[Dict] = None) -> List[Dict]:
        """疾患に対するすべての治療選択肢を比較"""
        try:
            if disease not in self.treatment_database:
                return []
            
            treatment_comparisons = []
            
            for treatment_option in self.treatment_database[disease]:
                prediction = self.predict_treatment_outcome(
                    disease, patient_risk_score, treatment_option, patient_data
                )
                
                if prediction:
                    treatment_comparisons.append({
                        'treatment': treatment_option,
                        'prediction': prediction
                    })
            
            # 推奨度スコア順にソート
            treatment_comparisons.sort(
                key=lambda x: x['prediction']['recommendation_score'], 
                reverse=True
            )
            
            return treatment_comparisons
            
        except Exception as e:
            self.logger.error(f"治療比較エラー: {str(e)}")
            return []
    
    def generate_treatment_plan(self, disease: str, patient_risk_score: float,
                              patient_data: Dict) -> Dict:
        """個別化された治療計画を生成"""
        try:
            # 治療選択肢の比較
            treatment_comparisons = self.compare_treatments(
                disease, patient_risk_score, patient_data
            )
            
            if not treatment_comparisons:
                return {'error': '適用可能な治療法が見つかりません'}
            
            # 最適な治療法を選択
            primary_treatment = treatment_comparisons[0]
            
            # 代替治療法を提案
            alternative_treatments = treatment_comparisons[1:3]  # 上位3つまで
            
            # 治療計画の詳細を生成
            treatment_plan = {
                'plan_timestamp': datetime.now().isoformat(),
                'patient_id': patient_data.get('patient_id', 'unknown'),
                'disease': disease,
                'primary_treatment': {
                    'name': primary_treatment['treatment'],
                    'details': primary_treatment['prediction'],
                    'rationale': self._generate_treatment_rationale(
                        primary_treatment, patient_data
                    )
                },
                'alternative_treatments': [
                    {
                        'name': alt['treatment'],
                        'details': alt['prediction'],
                        'reason_for_alternative': self._explain_alternative_reason(
                            alt, primary_treatment
                        )
                    }
                    for alt in alternative_treatments
                ],
                'monitoring_schedule': self._create_monitoring_schedule(
                    primary_treatment['treatment'], 
                    primary_treatment['prediction']
                ),
                'patient_education': self._generate_patient_education(
                    disease, primary_treatment['treatment']
                ),
                'follow_up_plan': self._create_follow_up_plan(
                    primary_treatment['prediction'], patient_risk_score
                ),
                'emergency_indicators': self._identify_emergency_indicators(
                    disease, primary_treatment['treatment']
                )
            }
            
            return treatment_plan
            
        except Exception as e:
            self.logger.error(f"治療計画生成エラー: {str(e)}")
            return {'error': str(e)}
    
    def _generate_treatment_rationale(self, primary_treatment: Dict, patient_data: Dict) -> str:
        """治療選択の根拠を生成"""
        treatment_name = primary_treatment['treatment']
        prediction = primary_treatment['prediction']
        
        rationale = f"{treatment_name}を第一選択とした理由："
        
        reasons = []
        
        if prediction['success_rate'] > 0.8:
            reasons.append(f"高い成功率（{prediction['success_rate']*100:.1f}%）")
        
        if prediction['side_effects_risk'] < 0.2:
            reasons.append("副作用リスクが低い")
        
        if prediction['cost_level'] == 'low':
            reasons.append("経済的負担が軽い")
        
        if not prediction['contraindications']:
            reasons.append("患者に禁忌事項がない")
        
        if prediction['confidence'] > 0.8:
            reasons.append("予測の信頼度が高い")
        
        # 患者固有の要因
        age = patient_data.get('age', 0)
        if age > 65 and prediction['side_effects_risk'] < 0.15:
            reasons.append("高齢者に適した安全性プロファイル")
        
        chronic_diseases = patient_data.get('chronic_diseases', [])
        if chronic_diseases and prediction['success_rate'] > 0.75:
            reasons.append("併存疾患があっても効果が期待できる")
        
        rationale += "、".join(reasons) + "。"
        
        return rationale
    
    def _explain_alternative_reason(self, alternative: Dict, primary: Dict) -> str:
        """代替治療を提案する理由を説明"""
        alt_pred = alternative['prediction']
        primary_pred = primary['prediction']
        
        reasons = []
        
        if alt_pred['side_effects_risk'] < primary_pred['side_effects_risk']:
            reasons.append("副作用がより少ない")
        
        if alt_pred['cost_level'] == 'low' and primary_pred['cost_level'] != 'low':
            reasons.append("コストが安い")
        
        if alt_pred['estimated_duration'] < primary_pred['estimated_duration']:
            reasons.append("治療期間が短い")
        
        if not alt_pred['monitoring_required'] and primary_pred['monitoring_required']:
            reasons.append("定期的なモニタリングが不要")
        
        if not reasons:
            reasons.append("第一選択が適さない場合の選択肢")
        
        return "、".join(reasons)
    
    def _create_monitoring_schedule(self, treatment: str, prediction: Dict) -> Dict:
        """モニタリングスケジュールを作成"""
        if not prediction['monitoring_required']:
            return {'monitoring_required': False}
        
        schedule = {'monitoring_required': True}
        
        # 治療の種類に応じたモニタリング間隔
        if treatment in ['antibiotic_therapy', 'antiviral_therapy']:
            schedule.update({
                'initial_assessment': '3日後',
                'follow_up_interval': '1週間ごと',
                'parameters_to_monitor': ['症状改善', '副作用', '体温']
            })
        elif treatment in ['ace_inhibitors', 'metformin']:
            schedule.update({
                'initial_assessment': '2週間後',
                'follow_up_interval': '1ヶ月ごと',
                'parameters_to_monitor': ['血圧', '腎機能', '電解質', '副作用']
            })
        elif treatment in ['insulin_therapy']:
            schedule.update({
                'initial_assessment': '1週間後',
                'follow_up_interval': '2週間ごと',
                'parameters_to_monitor': ['血糖値', 'HbA1c', '体重', '低血糖症状']
            })
        else:
            schedule.update({
                'initial_assessment': '1週間後',
                'follow_up_interval': '2週間ごと',
                'parameters_to_monitor': ['症状改善', '副作用']
            })
        
        return schedule
    
    def _generate_patient_education(self, disease: str, treatment: str) -> List[str]:
        """患者教育内容を生成"""
        education_points = []
        
        # 疾患別の基本教育
        disease_education = {
            'common_cold': [
                '十分な休息と水分摂取が重要です',
                '症状が1週間以上続く場合は再受診してください'
            ],
            'influenza': [
                '他人への感染を防ぐため、マスク着用と手洗いを徹底してください',
                '高熱が続く場合や呼吸困難がある場合は緊急受診してください'
            ],
            'pneumonia': [
                '処方された抗生物質は必ず最後まで服用してください',
                '呼吸困難や胸痛が悪化した場合は即座に受診してください'
            ],
            'hypertension': [
                '毎日同じ時間に血圧を測定し、記録してください',
                '塩分制限と適度な運動を継続してください'
            ],
            'diabetes': [
                '血糖値の自己測定を習慣化してください',
                '低血糖症状（冷汗、手の震え、意識障害）に注意してください'
            ]
        }
        
        education_points.extend(disease_education.get(disease, []))
        
        # 治療法別の教育
        treatment_education = {
            'antibiotic_therapy': [
                '抗生物質は医師の指示通り、最後まで服用してください',
                '下痢や発疹などの副作用があれば連絡してください'
            ],
            'insulin_therapy': [
                'インスリンの注射方法と保存方法について理解してください',
                '食事時間と注射のタイミングを一定に保ってください'
            ],
            'ace_inhibitors': [
                '起立時のめまいに注意してください',
                '空咳が続く場合は医師に相談してください'
            ]
        }
        
        education_points.extend(treatment_education.get(treatment, []))
        
        # 一般的な注意事項
        education_points.extend([
            '処方された薬は用法・用量を守って服用してください',
            '気になる症状や副作用があれば遠慮なく相談してください'
        ])
        
        return education_points
    
    def _create_follow_up_plan(self, prediction: Dict, risk_score: float) -> Dict:
        """フォローアップ計画を作成"""
        follow_up = {}
        
        # 治療期間に基づくフォローアップ
        duration = prediction['estimated_duration']
        
        if duration <= 7:  # 短期治療
            follow_up['next_appointment'] = '1週間後'
            follow_up['treatment_completion_check'] = '治療終了後3日以内'
        elif duration <= 30:  # 中期治療
            follow_up['next_appointment'] = '2週間後'
            follow_up['treatment_completion_check'] = '治療終了後1週間以内'
        else:  # 長期治療
            follow_up['next_appointment'] = '1ヶ月後'
            follow_up['routine_follow_up'] = '3ヶ月ごと'
        
        # リスクスコアに基づく調整
        if risk_score > 70:
            follow_up['high_risk_monitoring'] = '週1回の状況確認'
            follow_up['emergency_contact'] = '24時間対応可能'
        elif risk_score > 50:
            follow_up['moderate_risk_monitoring'] = '2週間ごとの状況確認'
        
        # 成功率に基づく調整
        if prediction['success_rate'] < 0.7:
            follow_up['alternative_treatment_review'] = '2週間後に効果判定'
        
        return follow_up
    
    def _identify_emergency_indicators(self, disease: str, treatment: str) -> List[str]:
        """緊急受診が必要な症状を特定"""
        emergency_indicators = []
        
        # 疾患別の緊急症状
        disease_emergencies = {
            'pneumonia': [
                '呼吸困難の悪化',
                '持続する高熱（39度以上）',
                '意識レベルの低下',
                '胸痛の増強'
            ],
            'influenza': [
                '呼吸困難',
                '持続する高熱（39度以上が3日以上）',
                '脱水症状',
                '意識障害'
            ],
            'hypertension': [
                '収縮期血圧180mmHg以上',
                '激しい頭痛',
                '視野異常',
                '胸痛や呼吸困難'
            ],
            'diabetes': [
                '血糖値400mg/dl以上',
                '意識障害',
                '激しい嘔吐',
                '脱水症状'
            ]
        }
        
        emergency_indicators.extend(disease_emergencies.get(disease, []))
        
        # 治療法別の緊急症状
        treatment_emergencies = {
            'antibiotic_therapy': [
                '重篤なアレルギー反応（発疹、呼吸困難）',
                '激しい下痢（血便）'
            ],
            'insulin_therapy': [
                '重度の低血糖（意識障害、けいれん）',
                'ケトアシドーシスの症状'
            ],
            'ace_inhibitors': [
                '血管浮腫（顔面・喉頭の腫れ）',
                '重度の低血圧'
            ]
        }
        
        emergency_indicators.extend(treatment_emergencies.get(treatment, []))
        
        return emergency_indicators

# 使用例とテスト用のメイン関数
if __name__ == "__main__":
    # 治療効果予測システムの初期化
    treatment_predictor = TreatmentEffectPredictor()
    
    # テスト用患者データ
    test_patient = {
        'patient_id': 'P001',
        'age': 58,
        'chronic_diseases': ['hypertension', 'diabetes'],
        'medications': ['metformin', 'lisinopril'],
        'pregnancy': False
    }
    
    print("=== 治療効果予測システム デモ ===")
    
    # 肺炎に対する治療計画生成
    disease = 'pneumonia'
    risk_score = 45.5
    
    treatment_plan = treatment_predictor.generate_treatment_plan(
        disease, risk_score, test_patient
    )
    
    print(f"\n治療計画生成時刻: {treatment_plan['plan_timestamp']}")
    print(f"患者ID: {treatment_plan['patient_id']}")
    print(f"疾患: {treatment_plan['disease']}")
    
    # 第一選択治療
    primary = treatment_plan['primary_treatment']
    print(f"\n【第一選択治療】")
    print(f"治療法: {primary['name']}")
    print(f"成功率: {primary['details']['success_rate']*100:.1f}%")
    print(f"推定期間: {primary['details']['estimated_duration']:.1f}日")
    print(f"推奨度: {primary['details']['recommendation_score']:.3f}")
    print(f"選択理由: {primary['rationale']}")
    
    # 代替治療
    print(f"\n【代替治療選択肢】")
    for i, alt in enumerate(treatment_plan['alternative_treatments'], 1):
        print(f"{i}. {alt['name']} - {alt['reason_for_alternative']}")
    
    # モニタリング
    monitoring = treatment_plan['monitoring_schedule']
    if monitoring['monitoring_required']:
        print(f"\n【モニタリング計画】")
        print(f"初回評価: {monitoring['initial_assessment']}")
        print(f"フォローアップ: {monitoring['follow_up_interval']}")
        print(f"監視項目: {', '.join(monitoring['parameters_to_monitor'])}")
    
    # 患者教育
    print(f"\n【患者教育ポイント】")
    for i, point in enumerate(treatment_plan['patient_education'], 1):
        print(f"{i}. {point}")
    
    # 緊急受診指標
    print(f"\n【緊急受診が必要な症状】")
    for i, indicator in enumerate(treatment_plan['emergency_indicators'], 1):
        print(f"{i}. {indicator}")
