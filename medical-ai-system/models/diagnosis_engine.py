"""
医療症状診断エンジン
患者の症状から可能性の高い疾患を予測するシステム

機能:
- 症状データベースに基づく疾患予測
- 症状の重症度を考慮した診断
- 診断結果の信頼度計算
"""

import json
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

class MedicalDiagnosisSystem:
    """医療診断システムのメインクラス"""
    
    def __init__(self):
        """診断システムの初期化"""
        self.symptom_disease_db = {}
        self.disease_info_db = {}
        self.symptom_weights = {}
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        # データベースの読み込み
        self._load_symptom_database()
        self._load_disease_information()
        self._calculate_symptom_weights()
        
        self.logger.info("診断システムが初期化されました")
    
    def _load_symptom_database(self):
        """症状-疾患関連データベースの読み込み"""
        self.symptom_disease_db = {
            'fever': {  # 発熱
                'common_cold': 0.7,      # 風邪
                'influenza': 0.85,       # インフルエンザ
                'pneumonia': 0.6,        # 肺炎
                'bronchitis': 0.5,       # 気管支炎
                'gastroenteritis': 0.4   # 胃腸炎
            },
            'cough': {  # 咳
                'common_cold': 0.8,
                'pneumonia': 0.9,
                'bronchitis': 0.95,
                'influenza': 0.7,
                'asthma': 0.85
            },
            'headache': {  # 頭痛
                'migraine': 0.9,
                'tension_headache': 0.8,
                'cluster_headache': 0.6,
                'influenza': 0.6,
                'hypertension': 0.4
            },
            'fatigue': {  # 疲労感
                'common_cold': 0.6,
                'influenza': 0.8,
                'anemia': 0.9,
                'depression': 0.7,
                'diabetes': 0.5
            },
            'nausea': {  # 悪心
                'gastroenteritis': 0.8,
                'food_poisoning': 0.9,
                'migraine': 0.6,
                'pregnancy': 0.7,
                'medication_side_effect': 0.5
            },
            'shortness_of_breath': {  # 呼吸困難
                'asthma': 0.9,
                'pneumonia': 0.7,
                'heart_failure': 0.8,
                'anxiety': 0.4,
                'anemia': 0.3
            },
            'chest_pain': {  # 胸痛
                'heart_disease': 0.8,
                'pneumonia': 0.4,
                'anxiety': 0.3,
                'muscle_strain': 0.6,
                'gastroesophageal_reflux': 0.5
            },
            'abdominal_pain': {  # 腹痛
                'gastroenteritis': 0.8,
                'appendicitis': 0.7,
                'gallstones': 0.6,
                'irritable_bowel_syndrome': 0.9,
                'food_poisoning': 0.7
            },
            'dizziness': {  # めまい
                'vertigo': 0.9,
                'hypertension': 0.5,
                'anemia': 0.6,
                'inner_ear_infection': 0.8,
                'medication_side_effect': 0.4
            },
            'skin_rash': {  # 皮疹
                'allergic_reaction': 0.8,
                'eczema': 0.7,
                'viral_infection': 0.5,
                'contact_dermatitis': 0.9,
                'medication_side_effect': 0.4
            }
        }
        
        self.logger.info(f"症状データベースを読み込み完了: {len(self.symptom_disease_db)}種類の症状")
    
    def _load_disease_information(self):
        """疾患情報データベースの読み込み"""
        self.disease_info_db = {
            'common_cold': {
                'name_jp': '風邪',
                'severity': 'mild',
                'typical_duration': 7,
                'contagious': True,
                'common_age_groups': ['all']
            },
            'influenza': {
                'name_jp': 'インフルエンザ',
                'severity': 'moderate',
                'typical_duration': 10,
                'contagious': True,
                'common_age_groups': ['all']
            },
            'pneumonia': {
                'name_jp': '肺炎',
                'severity': 'severe',
                'typical_duration': 14,
                'contagious': False,
                'common_age_groups': ['elderly', 'children']
            },
            'migraine': {
                'name_jp': '偏頭痛',
                'severity': 'moderate',
                'typical_duration': 24,
                'contagious': False,
                'common_age_groups': ['adults']
            },
            'gastroenteritis': {
                'name_jp': '胃腸炎',
                'severity': 'mild_to_moderate',
                'typical_duration': 5,
                'contagious': True,
                'common_age_groups': ['all']
            }
        }
        
        self.logger.info(f"疾患情報データベースを読み込み完了: {len(self.disease_info_db)}種類の疾患")
    
    def _calculate_symptom_weights(self):
        """症状の重要度重みを計算"""
        # 各症状が関連する疾患数に基づいて重みを計算
        for symptom, diseases in self.symptom_disease_db.items():
            # 関連疾患数が少ない症状ほど診断的価値が高い
            disease_count = len(diseases)
            avg_probability = sum(diseases.values()) / len(diseases)
            
            # 重みの計算（関連疾患が少なく、確率が高いほど重要）
            weight = (1.0 / disease_count) * avg_probability
            self.symptom_weights[symptom] = weight
        
        # 正規化（最大値を1.0に）
        max_weight = max(self.symptom_weights.values())
        for symptom in self.symptom_weights:
            self.symptom_weights[symptom] /= max_weight
        
        self.logger.info("症状重みの計算が完了しました")
    
    def diagnose_symptoms(self, symptoms_list: List[str], 
                         severity_scores: List[int]) -> List[Tuple[str, float]]:
        """
        症状から疾患を診断
        
        Args:
            symptoms_list: 症状名のリスト
            severity_scores: 各症状の重症度（1-10の整数）
        
        Returns:
            診断結果のリスト（疾患名, スコア）の組み合わせ
        """
        try:
            if len(symptoms_list) != len(severity_scores):
                self.logger.warning("症状数と重症度スコア数が一致しません")
                # 不足分は中程度の重症度（5）で補完
                while len(severity_scores) < len(symptoms_list):
                    severity_scores.append(5)
            
            disease_scores = {}
            total_weight = 0
            
            # 各症状について疾患スコアを計算
            for i, symptom in enumerate(symptoms_list):
                severity = severity_scores[i]
                
                # 重症度の正規化（1-10 → 0.1-1.0）
                normalized_severity = max(0.1, min(1.0, severity / 10.0))
                
                if symptom in self.symptom_disease_db:
                    # 症状の重要度重みを取得
                    symptom_weight = self.symptom_weights.get(symptom, 0.5)
                    total_weight += symptom_weight
                    
                    for disease, base_probability in self.symptom_disease_db[symptom].items():
                        if disease not in disease_scores:
                            disease_scores[disease] = 0
                        
                        # 重症度と症状重みを考慮したスコア計算
                        weighted_score = (
                            base_probability * 
                            normalized_severity * 
                            symptom_weight
                        )
                        
                        disease_scores[disease] += weighted_score
                else:
                    self.logger.warning(f"不明な症状: {symptom}")
            
            if not disease_scores:
                self.logger.info("該当する疾患が見つかりませんでした")
                return []
            
            # スコアの正規化
            if total_weight > 0:
                for disease in disease_scores:
                    disease_scores[disease] = (disease_scores[disease] / total_weight) * 100
            
            # スコアの上限を100%に制限
            max_score = max(disease_scores.values())
            if max_score > 100:
                for disease in disease_scores:
                    disease_scores[disease] = (disease_scores[disease] / max_score) * 100
            
            # スコア順に並び替えて上位5位まで返す
            sorted_results = sorted(
                disease_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            self.logger.info(f"診断完了: {len(sorted_results)}個の疾患候補を特定")
            return sorted_results
            
        except Exception as e:
            self.logger.error(f"診断処理中にエラーが発生: {str(e)}")
            return []
    
    def calculate_confidence(self, symptoms_count: int, 
                           severity_avg: float, 
                           database_matches: int) -> float:
        """
        診断の信頼度を計算
        
        Args:
            symptoms_count: 入力された症状の数
            severity_avg: 症状の平均重症度
            database_matches: データベースでマッチした症状の数
        
        Returns:
            信頼度スコア（0.0-1.0）
        """
        try:
            base_confidence = 0.3  # 基本信頼度30%
            
            # 症状数による信頼度向上（多いほど良い、ただし上限あり）
            symptom_bonus = min(symptoms_count * 0.08, 0.25)
            
            # 重症度による信頼度調整
            # 適度な重症度（5-7）で最高、極端に低いか高いと信頼度低下
            if 5 <= severity_avg <= 7:
                severity_bonus = 0.15
            elif 3 <= severity_avg <= 8:
                severity_bonus = 0.10
            else:
                severity_bonus = 0.05
            
            # データベースマッチ率による信頼度向上
            if symptoms_count > 0:
                match_rate = database_matches / symptoms_count
                match_bonus = match_rate * 0.20
            else:
                match_bonus = 0
            
            # 症状の組み合わせボーナス
            # 複数の症状が同じ疾患を指している場合の信頼度向上
            combination_bonus = self._calculate_combination_bonus(symptoms_count, database_matches)
            
            # 最終信頼度計算
            final_confidence = (
                base_confidence + 
                symptom_bonus + 
                severity_bonus + 
                match_bonus + 
                combination_bonus
            )
            
            # 0.0-1.0の範囲に制限（最大95%まで）
            final_confidence = max(0.0, min(0.95, final_confidence))
            
            self.logger.info(f"診断信頼度: {final_confidence:.1%}")
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"信頼度計算エラー: {str(e)}")
            return 0.0
    
    def _calculate_combination_bonus(self, symptoms_count: int, database_matches: int) -> float:
        """症状の組み合わせによる信頼度ボーナスを計算"""
        if symptoms_count < 2 or database_matches < 2:
            return 0.0
        
        # 多くの症状がデータベースにマッチし、それらが相互に関連している場合にボーナス
        match_rate = database_matches / symptoms_count
        
        if match_rate >= 0.8:  # 80%以上の症状がマッチ
            return 0.10
        elif match_rate >= 0.6:  # 60%以上の症状がマッチ
            return 0.05
        else:
            return 0.02
    
    def get_disease_information(self, disease_name: str) -> Optional[Dict]:
        """疾患の詳細情報を取得"""
        return self.disease_info_db.get(disease_name)
    
    def get_differential_diagnosis(self, primary_symptoms: List[str]) -> Dict[str, List[str]]:
        """鑑別診断のための関連症状を提案"""
        suggested_questions = {}
        
        for symptom in primary_symptoms:
            if symptom in self.symptom_disease_db:
                # この症状に関連する疾患を取得
                related_diseases = list(self.symptom_disease_db[symptom].keys())
                
                # 各疾患の他の典型症状を調べる
                additional_symptoms = set()
                for disease in related_diseases:
                    for other_symptom, disease_dict in self.symptom_disease_db.items():
                        if disease in disease_dict and other_symptom not in primary_symptoms:
                            additional_symptoms.add(other_symptom)
                
                suggested_questions[symptom] = list(additional_symptoms)[:5]  # 上位5個まで
        
        return suggested_questions
    
    def validate_symptom_input(self, symptoms_list: List[str]) -> Dict[str, List[str]]:
        """入力された症状の妥当性をチェック"""
        valid_symptoms = []
        invalid_symptoms = []
        
        known_symptoms = set(self.symptom_disease_db.keys())
        
        for symptom in symptoms_list:
            if symptom.lower() in known_symptoms:
                valid_symptoms.append(symptom.lower())
            else:
                invalid_symptoms.append(symptom)
        
        return {
            'valid_symptoms': valid_symptoms,
            'invalid_symptoms': invalid_symptoms,
            'suggestions': self._suggest_similar_symptoms(invalid_symptoms)
        }
    
    def _suggest_similar_symptoms(self, invalid_symptoms: List[str]) -> Dict[str, List[str]]:
        """無効な症状に対して類似症状を提案"""
        suggestions = {}
        known_symptoms = list(self.symptom_disease_db.keys())
        
        for invalid_symptom in invalid_symptoms:
            # 簡単な文字列類似度による提案（より高度なアルゴリズムも可能）
            similar_symptoms = []
            for known_symptom in known_symptoms:
                if (invalid_symptom.lower() in known_symptom.lower() or 
                    known_symptom.lower() in invalid_symptom.lower()):
                    similar_symptoms.append(known_symptom)
            
            suggestions[invalid_symptom] = similar_symptoms[:3]  # 上位3個まで
        
        return suggestions
    
    def generate_diagnosis_report(self, symptoms_list: List[str], 
                                severity_scores: List[int]) -> Dict:
        """包括的な診断レポートを生成"""
        try:
            # 入力検証
            validation_result = self.validate_symptom_input(symptoms_list)
            
            # 診断実行
            diagnosis_results = self.diagnose_symptoms(
                validation_result['valid_symptoms'], 
                severity_scores[:len(validation_result['valid_symptoms'])]
            )
            
            # 信頼度計算
            if validation_result['valid_symptoms']:
                avg_severity = sum(severity_scores[:len(validation_result['valid_symptoms'])]) / len(validation_result['valid_symptoms'])
                confidence = self.calculate_confidence(
                    len(validation_result['valid_symptoms']),
                    avg_severity,
                    len(validation_result['valid_symptoms'])
                )
            else:
                confidence = 0.0
            
            # 鑑別診断提案
            differential_suggestions = self.get_differential_diagnosis(validation_result['valid_symptoms'])
            
            # レポート作成
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'input_validation': validation_result,
                'diagnosis_results': diagnosis_results,
                'confidence_score': confidence,
                'primary_diagnosis': diagnosis_results[0][0] if diagnosis_results else None,
                'differential_diagnosis_suggestions': differential_suggestions,
                'recommendations': self._generate_recommendations(diagnosis_results, confidence),
                'report_summary': self._generate_report_summary(diagnosis_results, confidence)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"診断レポート生成エラー: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _generate_recommendations(self, diagnosis_results: List[Tuple[str, float]], 
                                confidence: float) -> List[str]:
        """診断結果に基づく推奨事項を生成"""
        recommendations = []
        
        if not diagnosis_results:
            recommendations.append("症状が特定の疾患パターンと一致しません。医師の診察を受けてください。")
            return recommendations
        
        if confidence < 0.4:
            recommendations.append("診断の信頼度が低いため、追加の検査や専門医への相談を強く推奨します。")
        elif confidence < 0.7:
            recommendations.append("診断の信頼度が中程度です。医師の確認を推奨します。")
        
        # 最上位診断に基づく推奨事項
        primary_disease = diagnosis_results[0][0]
        disease_info = self.get_disease_information(primary_disease)
        
        if disease_info:
            severity = disease_info.get('severity', 'unknown')
            if severity == 'severe':
                recommendations.append("重篤な疾患の可能性があります。速やかに医療機関を受診してください。")
            elif severity == 'moderate':
                recommendations.append("中程度の疾患の可能性があります。医師の診察を受けてください。")
            
            if disease_info.get('contagious', False):
                recommendations.append("感染性の疾患の可能性があります。他者への感染予防に注意してください。")
        
        return recommendations
    
    def _generate_report_summary(self, diagnosis_results: List[Tuple[str, float]], 
                               confidence: float) -> str:
        """診断レポートのサマリーを生成"""
        if not diagnosis_results:
            return "入力された症状から特定の疾患を診断することができませんでした。"
        
        primary_disease = diagnosis_results[0][0]
        primary_score = diagnosis_results[0][1]
        
        disease_info = self.get_disease_information(primary_disease)
        disease_name_jp = disease_info.get('name_jp', primary_disease) if disease_info else primary_disease
        
        summary = f"最も可能性の高い診断は{disease_name_jp}です（確率: {primary_score:.1f}%）。"
        summary += f" この診断の信頼度は{confidence:.1%}です。"
        
        if len(diagnosis_results) > 1:
            summary += f" 他の可能性として{len(diagnosis_results)-1}つの疾患も考慮する必要があります。"
        
        return summary

# 使用例とテスト用のメイン関数
if __name__ == "__main__":
    # 診断システムの初期化
    diagnosis_system = MedicalDiagnosisSystem()
    
    # テスト用症状データ
    test_symptoms = ['fever', 'cough', 'fatigue']
    test_severity = [8, 6, 5]
    
    print("=== 医療診断システム デモ ===")
    
    # 包括的診断レポートの生成
    report = diagnosis_system.generate_diagnosis_report(test_symptoms, test_severity)
    
    print(f"\n診断レポート生成時刻: {report['report_timestamp']}")
    print(f"診断信頼度: {report['confidence_score']:.1%}")
    print(f"主診断: {report['primary_diagnosis']}")
    
    print("\n診断結果:")
    for disease, score in report['diagnosis_results']:
        disease_info = diagnosis_system.get_disease_information(disease)
        disease_name = disease_info.get('name_jp', disease) if disease_info else disease
        print(f"  - {disease_name}: {score:.1f}%")
    
    print("\n推奨事項:")
    for recommendation in report['recommendations']:
        print(f"  - {recommendation}")
    
    print(f"\nサマリー: {report['report_summary']}")
