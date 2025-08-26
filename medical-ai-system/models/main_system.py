"""
医療AI統合システム - メインクラス
患者の総合的な医療情報を分析し、診断支援を行う統合システム

使用方法:
    medical_ai = MedicalAISystem()
    result = medical_ai.comprehensive_analysis(patient_data)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from diagnosis_engine import MedicalDiagnosisSystem
from risk_assessment import PatientRiskAssessment
from treatment_prediction import TreatmentEffectPredictor
from vital_monitoring import VitalSignsMonitor

class MedicalAISystem:
    """医療AI統合システムのメインクラス"""
    
    def __init__(self):
        """システム初期化"""
        # 各専門システムの初期化
        self.diagnosis_system = MedicalDiagnosisSystem()
        self.risk_assessor = PatientRiskAssessment()
        self.treatment_predictor = TreatmentEffectPredictor()
        self.vital_monitor = VitalSignsMonitor()
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("医療AIシステムが初期化されました")
    
    def comprehensive_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        患者の包括的医療分析を実行
        
        Args:
            patient_data (dict): 患者の全情報を含む辞書
                - patient_id: 患者ID
                - age: 年齢
                - bmi: BMI値
                - smoking_status: 喫煙状況
                - chronic_diseases: 既往歴リスト
                - symptoms: 症状リスト
                - symptom_severity: 症状の重症度リスト
                - vital_signs: バイタルサインの辞書
        
        Returns:
            dict: 統合分析結果
        """
        try:
            analysis_start_time = datetime.now()
            self.logger.info(f"患者 {patient_data.get('patient_id', 'unknown')} の分析を開始")
            
            # 1. 症状診断分析
            diagnosis_results = self._perform_diagnosis_analysis(patient_data)
            
            # 2. リスク評価分析
            risk_assessment = self._perform_risk_analysis(patient_data)
            
            # 3. 治療効果予測分析
            treatment_predictions = self._perform_treatment_analysis(
                diagnosis_results, risk_assessment, patient_data
            )
            
            # 4. バイタルサイン分析
            vital_analysis = self._perform_vital_analysis(patient_data)
            
            # 5. 統合判断の実行
            integrated_recommendation = self._generate_integrated_recommendation(
                diagnosis_results, risk_assessment, treatment_predictions, vital_analysis
            )
            
            # 最終結果の構築
            final_result = {
                'patient_id': patient_data.get('patient_id', 'unknown'),
                'analysis_timestamp': analysis_start_time.isoformat(),
                'diagnosis_analysis': diagnosis_results,
                'risk_assessment': risk_assessment,
                'treatment_predictions': treatment_predictions,
                'vital_analysis': vital_analysis,
                'integrated_recommendation': integrated_recommendation,
                'system_confidence': self._calculate_overall_confidence(
                    diagnosis_results, risk_assessment, vital_analysis
                ),
                'processing_time_seconds': (datetime.now() - analysis_start_time).total_seconds()
            }
            
            self.logger.info(f"分析完了 - 処理時間: {final_result['processing_time_seconds']:.2f}秒")
            return final_result
            
        except Exception as e:
            self.logger.error(f"包括分析中にエラーが発生: {str(e)}")
            return {
                'error': str(e),
                'patient_id': patient_data.get('patient_id', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
    
    def _perform_diagnosis_analysis(self, patient_data: Dict) -> Dict:
        """症状診断分析の実行"""
        try:
            symptoms = patient_data.get('symptoms', [])
            severity = patient_data.get('symptom_severity', [])
            
            if not symptoms:
                return {'status': 'no_symptoms', 'message': '症状データがありません'}
            
            # 症状リストと重症度の長さを合わせる
            if len(severity) < len(symptoms):
                severity.extend([5] * (len(symptoms) - len(severity)))  # デフォルト重症度5
            
            # 診断実行
            diagnosis_results = self.diagnosis_system.diagnose_symptoms(symptoms, severity)
            
            # 信頼度計算
            confidence = self.diagnosis_system.calculate_confidence(
                len(symptoms),
                sum(severity) / len(severity),
                len([s for s in symptoms if s in self.diagnosis_system.symptom_disease_db])
            )
            
            return {
                'status': 'completed',
                'diagnosis_candidates': diagnosis_results,
                'primary_diagnosis': diagnosis_results[0][0] if diagnosis_results else None,
                'confidence_score': confidence,
                'symptoms_analyzed': len(symptoms),
                'database_matches': len([s for s in symptoms if s in self.diagnosis_system.symptom_disease_db])
            }
            
        except Exception as e:
            self.logger.error(f"診断分析エラー: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _perform_risk_analysis(self, patient_data: Dict) -> Dict:
        """リスク評価分析の実行"""
        try:
            risk_score = self.risk_assessor.calculate_risk_score(patient_data)
            risk_level = self._categorize_risk_level(risk_score)
            
            # 詳細なリスク内訳を計算
            risk_breakdown = self.risk_assessor.get_risk_breakdown(patient_data)
            
            return {
                'status': 'completed',
                'risk_score': round(risk_score, 1),
                'risk_level': risk_level,
                'risk_breakdown': risk_breakdown,
                'recommendations': self._generate_risk_recommendations(risk_score, risk_level)
            }
            
        except Exception as e:
            self.logger.error(f"リスク分析エラー: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _perform_treatment_analysis(self, diagnosis_results: Dict, 
                                  risk_assessment: Dict, patient_data: Dict) -> Dict:
        """治療効果予測分析の実行"""
        try:
            if diagnosis_results.get('status') != 'completed':
                return {'status': 'no_diagnosis', 'message': '診断結果が必要です'}
            
            primary_diagnosis = diagnosis_results.get('primary_diagnosis')
            if not primary_diagnosis:
                return {'status': 'no_diagnosis', 'message': '主診断が特定されていません'}
            
            risk_score = risk_assessment.get('risk_score', 50)  # デフォルトリスク50
            
            # 利用可能な治療選択肢を取得
            available_treatments = self.treatment_predictor.get_available_treatments(primary_diagnosis)
            
            treatment_predictions = {}
            for treatment in available_treatments:
                prediction = self.treatment_predictor.predict_treatment_outcome(
                    primary_diagnosis, risk_score, treatment
                )
                if prediction:
                    treatment_predictions[treatment] = prediction
            
            # 最も推奨される治療法を決定
            best_treatment = self._select_best_treatment(treatment_predictions)
            
            return {
                'status': 'completed',
                'available_treatments': list(treatment_predictions.keys()),
                'treatment_predictions': treatment_predictions,
                'recommended_treatment': best_treatment,
                'treatment_count': len(treatment_predictions)
            }
            
        except Exception as e:
            self.logger.error(f"治療分析エラー: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _perform_vital_analysis(self, patient_data: Dict) -> Dict:
        """バイタルサイン分析の実行"""
        try:
            vital_data = patient_data.get('vital_signs', {})
            
            if not vital_data:
                return {'status': 'no_vital_data', 'message': 'バイタルデータがありません'}
            
            # バイタルサイン分析を実行
            vital_analysis = self.vital_monitor.analyze_vital_signs(vital_data)
            
            return {
                'status': 'completed',
                'vital_analysis': vital_analysis,
                'alert_count': len(vital_analysis.get('alerts', [])),
                'overall_vital_status': vital_analysis.get('overall_status', 'unknown')
            }
            
        except Exception as e:
            self.logger.error(f"バイタル分析エラー: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _generate_integrated_recommendation(self, diagnosis_results: Dict, 
                                         risk_assessment: Dict, 
                                         treatment_predictions: Dict, 
                                         vital_analysis: Dict) -> Dict:
        """統合的な推奨事項を生成"""
        try:
            recommendations = []
            priority_level = 'routine'  # routine, urgent, critical
            
            # バイタルサインの緊急度をチェック
            vital_status = vital_analysis.get('overall_vital_status', 'normal')
            if vital_status == 'critical':
                priority_level = 'critical'
                recommendations.append("緊急治療が必要です。バイタルサインに重篤な異常があります。")
            elif vital_status == 'warning':
                priority_level = 'urgent'
                recommendations.append("注意深い監視が必要です。バイタルサインに異常があります。")
            
            # 診断の信頼度をチェック
            diagnosis_confidence = diagnosis_results.get('confidence_score', 0)
            if diagnosis_confidence < 0.7:
                recommendations.append("診断の信頼度が低いため、追加の検査を推奨します。")
            
            # リスクレベルに応じた推奨事項
            risk_level = risk_assessment.get('risk_level', 'low')
            if risk_level == 'high':
                if priority_level == 'routine':
                    priority_level = 'urgent'
                recommendations.append("患者の高リスクにより、より慎重なアプローチが必要です。")
            
            # 治療推奨事項
            recommended_treatment = treatment_predictions.get('recommended_treatment')
            if recommended_treatment:
                treatment_info = treatment_predictions['treatment_predictions'].get(recommended_treatment)
                if treatment_info:
                    success_rate = treatment_info.get('success_rate', 0) * 100
                    duration = treatment_info.get('estimated_duration', 0)
                    recommendations.append(
                        f"推奨治療: {recommended_treatment} "
                        f"(成功率: {success_rate:.1f}%, 推定期間: {duration:.1f}日)"
                    )
            
            # 総合的な推奨事項
            if not recommendations:
                recommendations.append("現在の状態は安定しています。定期的な経過観察を継続してください。")
            
            return {
                'priority_level': priority_level,
                'recommendations': recommendations,
                'follow_up_required': priority_level in ['urgent', 'critical'],
                'estimated_recovery_time': self._estimate_recovery_time(
                    diagnosis_results, risk_assessment, treatment_predictions
                ),
                'next_evaluation_date': self._calculate_next_evaluation_date(priority_level)
            }
            
        except Exception as e:
            self.logger.error(f"統合推奨事項生成エラー: {str(e)}")
            return {'error': str(e)}
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """リスクスコアをレベル分類"""
        if risk_score >= 70:
            return 'high'
        elif risk_score >= 40:
            return 'medium'
        else:
            return 'low'
    
    def _generate_risk_recommendations(self, risk_score: float, risk_level: str) -> List[str]:
        """リスクレベルに応じた推奨事項を生成"""
        recommendations = []
        
        if risk_level == 'high':
            recommendations.extend([
                "定期的な専門医での診察を推奨します",
                "生活習慣の改善に積極的に取り組んでください",
                "緊急時の連絡先を常に携帯してください"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "年1-2回の定期健診を受診してください",
                "健康的な生活習慣を心がけてください"
            ])
        else:
            recommendations.append("現在のライフスタイルを維持してください")
        
        return recommendations
    
    def _select_best_treatment(self, treatment_predictions: Dict) -> str:
        """最適な治療法を選択"""
        if not treatment_predictions:
            return None
        
        best_treatment = None
        best_score = 0
        
        for treatment, prediction in treatment_predictions.items():
            score = prediction.get('recommendation_score', 0)
            if score > best_score:
                best_score = score
                best_treatment = treatment
        
        return best_treatment
    
    def _calculate_overall_confidence(self, diagnosis_results: Dict, 
                                    risk_assessment: Dict, vital_analysis: Dict) -> float:
        """システム全体の信頼度を計算"""
        confidence_scores = []
        
        # 診断の信頼度
        if diagnosis_results.get('status') == 'completed':
            confidence_scores.append(diagnosis_results.get('confidence_score', 0.5))
        
        # リスク評価の信頼度（固定値、実際は計算ロジックを実装）
        if risk_assessment.get('status') == 'completed':
            confidence_scores.append(0.85)
        
        # バイタル分析の信頼度
        if vital_analysis.get('status') == 'completed':
            confidence_scores.append(0.90)
        
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        else:
            return 0.0
    
    def _estimate_recovery_time(self, diagnosis_results: Dict, 
                              risk_assessment: Dict, treatment_predictions: Dict) -> Optional[float]:
        """回復時間を推定"""
        recommended_treatment = treatment_predictions.get('recommended_treatment')
        if recommended_treatment:
            treatment_info = treatment_predictions['treatment_predictions'].get(recommended_treatment)
            if treatment_info:
                base_duration = treatment_info.get('estimated_duration', 7)
                risk_score = risk_assessment.get('risk_score', 50)
                
                # リスクに応じて回復時間を調整
                risk_multiplier = 1.0 + (risk_score / 200)  # 高リスクほど回復時間が長い
                
                return base_duration * risk_multiplier
        
        return None
    
    def _calculate_next_evaluation_date(self, priority_level: str) -> str:
        """次回評価日を計算"""
        from datetime import timedelta
        
        if priority_level == 'critical':
            next_date = datetime.now() + timedelta(hours=4)
        elif priority_level == 'urgent':
            next_date = datetime.now() + timedelta(days=1)
        else:
            next_date = datetime.now() + timedelta(days=7)
        
        return next_date.strftime('%Y-%m-%d %H:%M')
    
    def get_system_status(self) -> Dict[str, Any]:
        """システムステータスを取得"""
        return {
            'system_name': 'Medical AI Integrated System',
            'version': '1.0.0',
            'status': 'operational',
            'modules': {
                'diagnosis_system': 'active',
                'risk_assessor': 'active',
                'treatment_predictor': 'active',
                'vital_monitor': 'active'
            },
            'last_updated': datetime.now().isoformat()
        }

# 使用例とテスト用のメイン関数
if __name__ == "__main__":
    # システムの初期化
    medical_ai = MedicalAISystem()
    
    # サンプル患者データ
    sample_patient = {
        'patient_id': 'P001',
        'age': 45,
        'bmi': 28.5,
        'smoking_status': 'ex_smoker',
        'chronic_diseases': ['hypertension'],
        'symptoms': ['fever', 'cough', 'fatigue'],
        'symptom_severity': [8, 6, 5],
        'vital_signs': {
            'heart_rate': 110,
            'blood_pressure_sys': 150,
            'blood_pressure_dia': 95,
            'body_temperature': 38.5,
            'oxygen_saturation': 92
        }
    }
    
    # 包括的分析の実行
    print("=== 医療AI統合システム デモ実行 ===")
    result = medical_ai.comprehensive_analysis(sample_patient)
    
    # 結果の表示
    print(f"\n患者ID: {result['patient_id']}")
    print(f"分析時刻: {result['analysis_timestamp']}")
    print(f"処理時間: {result['processing_time_seconds']:.2f}秒")
    print(f"システム信頼度: {result['system_confidence']:.1%}")
    
    # 診断結果
    diagnosis = result['diagnosis_analysis']
    if diagnosis.get('status') == 'completed':
        print(f"\n【診断結果】")
        print(f"主診断: {diagnosis['primary_diagnosis']}")
        print(f"診断信頼度: {diagnosis['confidence_score']:.1%}")
        print("診断候補:")
        for disease, score in diagnosis['diagnosis_candidates']:
            print(f"  - {disease}: {score:.1f}%")
    
    # リスク評価
    risk = result['risk_assessment']
    if risk.get('status') == 'completed':
        print(f"\n【リスク評価】")
        print(f"リスクスコア: {risk['risk_score']}")
        print(f"リスクレベル: {risk['risk_level']}")
    
    # 統合推奨事項
    recommendation = result['integrated_recommendation']
    print(f"\n【統合推奨事項】")
    print(f"優先度: {recommendation['priority_level']}")
    print("推奨事項:")
    for rec in recommendation['recommendations']:
        print(f"  - {rec}")
