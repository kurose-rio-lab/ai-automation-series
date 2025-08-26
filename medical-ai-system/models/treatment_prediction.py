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
                interaction_risk
