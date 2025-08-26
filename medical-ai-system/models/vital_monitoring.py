"""
バイタルサイン異常検知システム
患者のバイタルサインをリアルタイムで監視し、異常を検知

機能:
- 心拍数、血圧、体温、酸素飽和度などの監視
- 年齢・性別による正常値の調整
- 重症度別のアラート生成
- トレンド分析による早期警告
"""

import math
import statistics
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

class VitalSignsMonitor:
    """バイタルサイン監視システムのメインクラス"""
    
    def __init__(self):
        """バイタルサイン監視システムの初期化"""
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        # 正常値範囲の設定
        self._load_normal_ranges()
        self._load_age_adjustments()
        self._load_condition_adjustments()
        
        self.logger.info("バイタルサイン監視システムが初期化されました")
    
    def _load_normal_ranges(self):
        """基本的な正常値範囲を定義"""
        self.base_normal_ranges = {
            'heart_rate': {
                'min': 60,
                'max': 100,
                'unit': 'bpm',
                'critical_low': 40,
                'critical_high': 150,
                'warning_low': 50,
                'warning_high': 120
            },
            'blood_pressure_systolic': {
                'min': 90,
                'max': 140,
                'unit': 'mmHg',
                'critical_low': 70,
                'critical_high': 180,
                'warning_low': 80,
                'warning_high': 160
            },
            'blood_pressure_diastolic': {
                'min': 60,
                'max': 90,
                'unit': 'mmHg',
                'critical_low': 40,
                'critical_high': 110,
                'warning_low': 50,
                'warning_high': 100
            },
            'body_temperature': {
                'min': 36.1,
                'max': 37.2,
                'unit': '°C',
                'critical_low': 35.0,
                'critical_high': 40.0,
                'warning_low': 35.5,
                'warning_high': 38.5
            },
            'oxygen_saturation': {
                'min': 95,
                'max': 100,
                'unit': '%',
                'critical_low': 85,
                'critical_high': 100,  # 上限は通常問題ない
                'warning_low': 90,
                'warning_high': 100
            },
            'respiratory_rate': {
                'min': 12,
                'max': 20,
                'unit': '/min',
                'critical_low': 8,
                'critical_high': 35,
                'warning_low': 10,
                'warning_high': 25
            },
            'blood_glucose': {
                'min': 70,
                'max': 140,
                'unit': 'mg/dL',
                'critical_low': 50,
                'critical_high': 400,
                'warning_low': 60,
                'warning_high': 200
            }
        }
    
    def _load_age_adjustments(self):
        """年齢による正常値調整を定義"""
        self.age_adjustments = {
            'heart_rate': {
                (0, 1): {'min': 80, 'max': 160},      # 新生児
                (1, 5): {'min': 75, 'max': 130},      # 幼児
                (6, 12): {'min': 70, 'max': 110},     # 学童
                (13, 17): {'min': 60, 'max': 105},    # 青少年
                (18, 64): {'min': 60, 'max': 100},    # 成人
                (65, 120): {'min': 50, 'max': 95}     # 高齢者
            },
            'blood_pressure_systolic': {
                (0, 5): {'min': 70, 'max': 110},
                (6, 12): {'min': 80, 'max': 120},
                (13, 17): {'min': 90, 'max': 130},
                (18, 64): {'min': 90, 'max': 140},
                (65, 120): {'min': 95, 'max': 150}    # 高齢者は少し高めでも許容
            },
            'respiratory_rate': {
                (0, 1): {'min': 25, 'max': 50},
                (1, 5): {'min': 20, 'max': 30},
                (6, 12): {'min': 15, 'max': 25},
                (13, 17): {'min': 12, 'max': 22},
                (18, 120): {'min': 12, 'max': 20}
            }
        }
    
    def _load_condition_adjustments(self):
        """疾患による正常値調整を定義"""
        self.condition_adjustments = {
            'hypertension': {
                'blood_pressure_systolic': {'target_max': 130},
                'blood_pressure_diastolic': {'target_max': 80}
            },
            'diabetes': {
                'blood_glucose': {'target_min': 80, 'target_max': 130}
            },
            'heart_disease': {
                'heart_rate': {'target_min': 50, 'target_max': 85},
                'blood_pressure_systolic': {'target_max': 120}
            },
            'copd': {  # 慢性閉塞性肺疾患
                'oxygen_saturation': {'target_min': 88, 'target_max': 92},
                'respiratory_rate': {'warning_high': 30}
            },
            'anemia': {
                'heart_rate': {'warning_high': 110},  # 代償性頻脈
                'oxygen_saturation': {'warning_low': 92}
            }
        }
    
    def analyze_vital_signs(self, vital_data: Dict[str, float], 
                          patient_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        バイタルサインの包括的分析
        
        Args:
            vital_data: バイタルサインのデータ
            patient_info: 患者情報（年齢、疾患など）
        
        Returns:
            分析結果辞書
        """
        try:
            analysis_timestamp = datetime.now()
            
            # 患者情報の取得
            age = patient_info.get('age', 50) if patient_info else 50
            chronic_diseases = patient_info.get('chronic_diseases', []) if patient_info else []
            
            # 個別バイタルサインの分析
            individual_analyses = {}
            alerts = []
            overall_severity = 'normal'
            
            for vital_type, value in vital_data.items():
                if vital_type in self.base_normal_ranges:
                    analysis = self._analyze_single_vital(
                        vital_type, value, age, chronic_diseases
                    )
                    individual_analyses[vital_type] = analysis
                    
                    # アラートの収集
                    if analysis['alert_level'] != 'normal':
                        alerts.append({
                            'vital_type': vital_type,
                            'value': value,
                            'alert_level': analysis['alert_level'],
                            'message': analysis['message'],
                            'recommendation': analysis['recommendation']
                        })
                        
                        # 全体の重症度更新
                        if analysis['alert_level'] == 'critical' or overall_severity != 'critical':
                            if analysis['alert_level'] == 'critical':
                                overall_severity = 'critical'
                            elif analysis['alert_level'] == 'warning' and overall_severity == 'normal':
                                overall_severity = 'warning'
            
            # バイタルサインの相関分析
            correlation_analysis = self._analyze_vital_correlations(vital_data, patient_info)
            
            # 総合的なリスク評価
            composite_risk = self._calculate_composite_risk(individual_analyses, correlation_analysis)
            
            # 最終結果の構築
            result = {
                'analysis_timestamp': analysis_timestamp.isoformat(),
                'overall_status': overall_severity,
                'composite_risk_score': composite_risk,
                'individual_analyses': individual_analyses,
                'alerts': alerts,
                'correlation_analysis': correlation_analysis,
                'recommendations': self._generate_overall_recommendations(
                    individual_analyses, correlation_analysis, overall_severity
                ),
                'requires_immediate_attention': overall_severity == 'critical',
                'monitoring_priorities': self._identify_monitoring_priorities(alerts),
                'trend_indicators': self._analyze_trend_indicators(vital_data, patient_info)
            }
            
            self.logger.info(f"バイタル分析完了 - 状態: {overall_severity}, アラート数: {len(alerts)}")
            return result
            
        except Exception as e:
            self.logger.error(f"バイタルサイン分析エラー: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_single_vital(self, vital_type: str, value: float, 
                            age: int, chronic_diseases: List[str]) -> Dict[str, Any]:
        """単一バイタルサインの詳細分析"""
        base_ranges = self.base_normal_ranges[vital_type]
        
        # 年齢調整された正常値範囲を取得
        adjusted_ranges = self._get_age_adjusted_ranges(vital_type, age)
        
        # 疾患による調整
        condition_adjusted_ranges = self._apply_condition_adjustments(
            vital_type, adjusted_ranges, chronic_diseases
        )
        
        # 異常レベルの判定
        alert_level, message = self._determine_alert_level(
            vital_type, value, condition_adjusted_ranges
        )
        
        # 推奨事項の生成
        recommendation = self._generate_vital_recommendation(
            vital_type, value, alert_level, chronic_diseases
        )
        
        # 正常範囲からの偏差計算
        deviation = self._calculate_deviation(value, condition_adjusted_ranges)
        
        return {
            'value': value,
            'unit': base_ranges['unit'],
            'normal_range': f"{condition_adjusted_ranges['min']}-{condition_adjusted_ranges['max']}",
            'alert_level': alert_level,
            'message': message,
            'recommendation': recommendation,
            'deviation_percentage': deviation,
            'adjusted_for_age': age != 50,  # デフォルト年齢と異なる場合
            'adjusted_for_conditions': len(chronic_diseases) > 0
        }
    
    def _get_age_adjusted_ranges(self, vital_type: str, age: int) -> Dict[str, float]:
        """年齢調整された正常値範囲を取得"""
        base_ranges = self.base_normal_ranges[vital_type].copy()
        
        if vital_type in self.age_adjustments:
            age_ranges = self.age_adjustments[vital_type]
            
            for (min_age, max_age), adjustments in age_ranges.items():
                if min_age <= age <= max_age:
                    base_ranges.update(adjustments)
                    break
        
        return base_ranges
    
    def _apply_condition_adjustments(self, vital_type: str, ranges: Dict[str, float], 
                                   chronic_diseases: List[str]) -> Dict[str, float]:
        """慢性疾患による正常値調整を適用"""
        adjusted_ranges = ranges.copy()
        
        for disease in chronic_diseases:
            if disease in self.condition_adjustments:
                disease_adjustments = self.condition_adjustments[disease]
                if vital_type in disease_adjustments:
                    adjustments = disease_adjustments[vital_type]
                    adjusted_ranges.update(adjustments)
        
        return adjusted_ranges
    
    def _determine_alert_level(self, vital_type: str, value: float, 
                             ranges: Dict[str, float]) -> Tuple[str, str]:
        """アラートレベルとメッセージを決定"""
        
        # 緊急レベルのチェック
        if (value <= ranges.get('critical_low', 0) or 
            value >= ranges.get('critical_high', float('inf'))):
            return 'critical', f'{vital_type}が危険レベルです: {value}{ranges["unit"]}'
        
        # 警告レベルのチェック
        if (value <= ranges.get('warning_low', ranges['min']) or 
            value >= ranges.get('warning_high', ranges['max'])):
            return 'warning', f'{vital_type}が警告レベルです: {value}{ranges["unit"]}'
        
        # 正常範囲のチェック
        if ranges['min'] <= value <= ranges['max']:
            return 'normal', f'{vital_type}は正常範囲内です: {value}{ranges["unit"]}'
        
        # 軽度異常
        return 'mild', f'{vital_type}が軽度異常です: {value}{ranges["unit"]}'
    
    def _calculate_deviation(self, value: float, ranges: Dict[str, float]) -> float:
        """正常範囲からの偏差をパーセンテージで計算"""
        normal_min = ranges['min']
        normal_max = ranges['max']
        normal_center = (normal_min + normal_max) / 2
        normal_range = normal_max - normal_min
        
        if normal_range == 0:
            return 0.0
        
        deviation = abs(value - normal_center) / (normal_range / 2) * 100
        return min(deviation, 200.0)  # 最大200%まで
    
    def _generate_vital_recommendation(self, vital_type: str, value: float, 
                                     alert_level: str, chronic_diseases: List[str]) -> str:
        """バイタルサイン別の推奨事項を生成"""
        if alert_level == 'normal':
            return "現在の値は正常範囲内です。継続的な監視を続けてください。"
        
        recommendations = {
            'heart_rate': {
                'high': "頻脈が見られます。安静にして、カフェインや刺激物を避けてください。",
                'low': "徐脈が見られます。めまいや失神の症状に注意してください。"
            },
            'blood_pressure_systolic': {
                'high': "血圧上昇が見られます。塩分制限と安静を心がけてください。",
                'low': "血圧低下が見られます。十分な水分摂取と急な体位変換を避けてください。"
            },
            'body_temperature': {
                'high': "発熱が見られます。解熱剤の使用を検討し、水分補給を行ってください。",
                'low': "体温低下が見られます。保温に努め、温かい飲み物を摂取してください。"
            },
            'oxygen_saturation': {
                'low': "酸素飽和度が低下しています。深呼吸を行い、必要に応じて酸素投与を検討してください。"
            }
        }
        
        # 基本的な推奨事項
        vital_recs = recommendations.get(vital_type, {})
        direction = 'high' if value > 100 else 'low'  # 簡略化
        base_recommendation = vital_recs.get(direction, "異常値が検出されました。医師に相談してください。")
        
        # 慢性疾患による追加推奨事項
        additional_recs = []
        if 'diabetes' in chronic_diseases and vital_type == 'blood_glucose':
            if value > 200:
                additional_recs.append("血糖値が高いため、インスリンの調整が必要な可能性があります。")
            elif value < 70:
                additional_recs.append("低血糖の可能性があります。ブドウ糖の摂取を検討してください。")
        
        if 'hypertension' in chronic_diseases and 'blood_pressure' in vital_type:
            additional_recs.append("高血圧の既往があるため、降圧薬の調整が必要な可能性があります。")
        
        final_recommendation = base_recommendation
        if additional_recs:
            final_recommendation += " " + " ".join(additional_recs)
        
        return final_recommendation
    
    def _analyze_vital_correlations(self, vital_data: Dict[str, float], 
                                  patient_info: Optional[Dict]) -> Dict[str, Any]:
        """バイタルサイン間の相関分析"""
        correlations = {}
        
        # 心拍数と血圧の関係
        if 'heart_rate' in vital_data and 'blood_pressure_systolic' in vital_data:
            hr = vital_data['heart_rate']
            bp_sys = vital_data['blood_pressure_systolic']
            
            # ショック状態の可能性（頻脈 + 低血圧）
            if hr > 100 and bp_sys < 90:
                correlations['shock_risk'] = {
                    'level': 'high',
                    'description': '頻脈と低血圧の組み合わせ：ショック状態の可能性'
                }
            
            # 高血圧性クライシス（高血圧 + 頻脈）
            elif hr > 100 and bp_sys > 160:
                correlations['hypertensive_crisis_risk'] = {
                    'level': 'moderate',
                    'description': '高血圧と頻脈の組み合わせ：高血圧クライシスの可能性'
                }
        
        # 体温と心拍数の関係
        if 'body_temperature' in vital_data and 'heart_rate' in vital_data:
            temp = vital_data['body_temperature']
            hr = vital_data['heart_rate']
            
            # 発熱に対する心拍数増加の評価
            expected_hr_increase = (temp - 37.0) * 10  # 1度上昇につき約10bpm増加
            if temp > 37.5 and hr < (70 + expected_hr_increase):
                correlations['insufficient_tachycardia'] = {
                    'level': 'moderate',
                    'description': '発熱に対して心拍数増加が不十分：重篤な感染症の可能性'
                }
        
        # 酸素飽和度と呼吸数の関係
        if 'oxygen_saturation' in vital_data and 'respiratory_rate' in vital_data:
            o2_sat = vital_data['oxygen_saturation']
            resp_rate = vital_data['respiratory_rate']
            
            # 低酸素血症と頻呼吸
            if o2_sat < 92 and resp_rate > 24:
                correlations['respiratory_distress'] = {
                    'level': 'high',
                    'description': '低酸素血症と頻呼吸：呼吸不全の可能性'
                }
        
        return correlations
    
    def _calculate_composite_risk(self, individual_analyses: Dict, 
                                correlation_analysis: Dict) -> float:
        """複合的なリスクスコアを計算"""
        total_risk = 0.0
        weight_sum = 0.0
        
        # 個別バイタルサインのリスク
        vital_weights = {
            'heart_rate': 0.2,
            'blood_pressure_systolic': 0.25,
            'body_temperature': 0.15,
            'oxygen_saturation': 0.3,
            'respiratory_rate': 0.1
        }
        
        for vital_type, analysis in individual_analyses.items():
            weight = vital_weights.get(vital_type, 0.1)
            
            # アラートレベルに基づくスコア
            level_scores = {
                'normal': 0.0,
                'mild': 0.2,
                'warning': 0.6,
                'critical': 1.0
            }
            
            vital_risk = level_scores.get(analysis['alert_level'], 0.0)
            total_risk += vital_risk * weight
            weight_sum += weight
        
        # 相関分析による追加リスク
        correlation_risk = 0.0
        for correlation_name, correlation_data in correlation_analysis.items():
            level_risk = {
                'low': 0.1,
                'moderate': 0.3,
                'high': 0.5
            }
            correlation_risk += level_risk.get(correlation_data['level'], 0.0)
        
        # 最終的なリスクスコア（0-100）
        if weight_sum > 0:
            base_risk = (total_risk / weight_sum) * 70  # 70%を上限とする基本リスク
        else:
            base_risk = 0.0
        
        final_risk = min(100.0, base_risk + correlation_risk * 30)  # 相関リスクを30%まで追加
        
        return round(final_risk, 1)
    
    def _generate_overall_recommendations(self, individual_analyses: Dict, 
                                        correlation_analysis: Dict, 
                                        overall_severity: str) -> List[str]:
        """全体的な推奨事項を生成"""
        recommendations = []
        
        # 重症度に基づく基本推奨事項
        if overall_severity == 'critical':
            recommendations.extend([
                "緊急医療対応が必要です。直ちに医師に連絡してください。",
                "バイタルサインの継続的な監視を行ってください。",
                "患者の状態変化に注意深く観察してください。"
            ])
        elif overall_severity == 'warning':
            recommendations.extend([
                "注意深い監視が必要です。医師への連絡を検討してください。",
                "バイタルサインを定期的に測定してください。"
            ])
        else:
            recommendations.append("現在の状態は安定しています。定期的な監視を継続してください。")
        
        # 特定の異常パターンに対する推奨事項
        critical_vitals = [
            vital for vital, analysis in individual_analyses.items() 
            if analysis['alert_level'] == 'critical'
        ]
        
        if critical_vitals:
            recommendations.append(
                f"特に{', '.join(critical_vitals)}の異常値に注意が必要です。"
            )
        
        # 相関分析に基づく推奨事項
        for correlation_name, correlation_data in correlation_analysis.items():
            if correlation_data['level'] in ['moderate', 'high']:
                recommendations.append(f"注意: {correlation_data['description']}")
        
        return recommendations
    
    def _identify_monitoring_priorities(self, alerts: List[Dict]) -> List[str]:
        """監視優先項目を特定"""
        priorities = []
        
        # 緊急レベルのアラートを最優先
        critical_alerts = [alert for alert in alerts if alert['alert_level'] == 'critical']
        if critical_alerts:
            priorities.extend([f"緊急監視: {alert['vital_type']}" for alert in critical_alerts])
        
        # 警告レベルのアラート
        warning_alerts = [alert for alert in alerts if alert['alert_level'] == 'warning']
        if warning_alerts:
            priorities.extend([f"重点監視: {alert['vital_type']}" for alert in warning_alerts])
        
        if not priorities:
            priorities.append("定期的な全般監視")
        
        return priorities
    
    def _analyze_trend_indicators(self, vital_data: Dict[str, float], 
                                patient_info: Optional[Dict]) -> Dict[str, str]:
        """トレンド指標の分析（簡略化版）"""
        # 実際の実装では過去のデータとの比較を行う
        # ここでは現在値に基づく傾向予測を簡略化して実装
        
        trends = {}
        
        # 体温と心拍数から感染症の進行を予測
        if 'body_temperature' in vital_data and 'heart_rate' in vital_data:
            temp = vital_data['body_temperature']
            hr = vital_data['heart_rate']
            
            if temp > 38.0 and hr > 100:
                trends['infection_progression'] = '感染症の進行の可能性 - 継続監視が必要'
        
        # 血圧と心拍数から循環動態を評価
        if all(key in vital_data for key in ['blood_pressure_systolic', 'heart_rate']):
            bp_sys = vital_data['blood_pressure_systolic']
            hr = vital_data['heart_rate']
            
            if bp_sys < 100 and hr > 100:
                trends['hemodynamic_instability'] = '循環動態不安定の兆候 - 注意深い監視が必要'
        
        return trends

    def generate_vital_report(self, vital_data: Dict[str, float], 
                            patient_info: Optional[Dict] = None, 
                            historical_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """包括的なバイタルサインレポートを生成"""
        try:
            # 基本分析
            analysis_result = self.analyze_vital_signs(vital_data, patient_info)
            
            # 履歴データがある場合のトレンド分析
            trend_analysis = {}
            if historical_data:
                trend_analysis = self._analyze_historical_trends(
                    vital_data, historical_data
                )
            
            # 予測的警告の生成
            predictive_warnings = self._generate_predictive_warnings(
                vital_data, patient_info, trend_analysis
            )
            
            # 詳細レポートの構築
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'patient_id': patient_info.get('patient_id', 'unknown') if patient_info else 'unknown',
                'current_analysis': analysis_result,
                'trend_analysis': trend_analysis,
                'predictive_warnings': predictive_warnings,
                'clinical_summary': self._generate_clinical_summary(
                    analysis_result, trend_analysis
                ),
                'action_plan': self._generate_action_plan(
                    analysis_result, predictive_warnings
                ),
                'next_monitoring_schedule': self._determine_monitoring_schedule(
                    analysis_result['overall_status']
                )
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"バイタルレポート生成エラー: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _analyze_historical_trends(self, current_data: Dict[str, float], 
                                 historical_data: List[Dict]) -> Dict[str, Any]:
        """履歴データからトレンドを分析"""
        trends = {}
        
        for vital_type in current_data:
            if len(historical_data) >= 3:  # 最低3つのデータポイントが必要
                historical_values = [
                    data.get(vital_type) for data in historical_data[-10:]  # 最新10件
                    if data.get(vital_type) is not None
                ]
                
                if len(historical_values) >= 3:
                    current_value = current_data[vital_type]
                    recent_avg = statistics.mean(historical_values[-3:])
                    older_avg = statistics.mean(historical_values[:-3]) if len(historical_values) > 3 else recent_avg
                    
                    # トレンドの方向性
                    if recent_avg > older_avg * 1.05:
                        trend_direction = 'increasing'
                    elif recent_avg < older_avg * 0.95:
                        trend_direction = 'decreasing'
                    else:
                        trend_direction = 'stable'
                    
                    # 変動性の計算
                    variability = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
                    
                    trends[vital_type] = {
                        'direction': trend_direction,
                        'current_value': current_value,
                        'recent_average': round(recent_avg, 1),
                        'variability': round(variability, 1),
                        'stability': 'stable' if variability < (recent_avg * 0.1) else 'variable'
                    }
        
        return trends
    
    def _generate_predictive_warnings(self, vital_data: Dict[str, float], 
                                    patient_info: Optional[Dict], 
                                    trend_analysis: Dict) -> List[Dict[str, str]]:
        """予測的警告を生成"""
        warnings = []
        
        # トレンドベースの警告
        for vital_type, trend_data in trend_analysis.items():
            if trend_data['direction'] == 'increasing':
                if vital_type == 'body_temperature' and trend_data['current_value'] > 37.5:
                    warnings.append({
                        'type': 'trend_warning',
                        'vital': vital_type,
                        'message': '体温上昇傾向：感染症の進行に注意',
                        'priority': 'medium'
                    })
                elif vital_type == 'heart_rate' and trend_data['current_value'] > 100:
                    warnings.append({
                        'type': 'trend_warning',
                        'vital': vital_type,
                        'message': '心拍数増加傾向：心血管系ストレスの可能性',
                        'priority': 'medium'
                    })
        
        # 患者特性に基づく警告
        if patient_info:
            age = patient_info.get('age', 0)
            chronic_diseases = patient_info.get('chronic_diseases', [])
            
            # 高齢者特有の警告
            if age > 75:
                if vital_data.get('blood_pressure_systolic', 0) > 160:
                    warnings.append({
                        'type': 'age_specific',
                        'message': '高齢者の高血圧：脳血管イベントリスク増加',
                        'priority': 'high'
                    })
            
            # 疾患特異的警告
            if 'diabetes' in chronic_diseases:
                if vital_data.get('blood_glucose', 0) > 200:
                    warnings.append({
                        'type': 'disease_specific',
                        'message': '糖尿病患者の高血糖：ケトアシドーシスリスク',
                        'priority': 'high'
                    })
        
        return warnings
    
    def _generate_clinical_summary(self, analysis_result: Dict, 
                                 trend_analysis: Dict) -> str:
        """臨床サマリーを生成"""
        overall_status = analysis_result['overall_status']
        alert_count = len(analysis_result['alerts'])
        
        summary = f"患者の現在のバイタルサイン状態は{overall_status}レベルです。"
        
        if alert_count > 0:
            summary += f" {alert_count}項目で異常値が検出されています。"
            
            critical_alerts = [
                alert for alert in analysis_result['alerts'] 
                if alert['alert_level'] == 'critical'
            ]
            if critical_alerts:
                critical_vitals = [alert['vital_type'] for alert in critical_alerts]
                summary += f" 特に{', '.join(critical_vitals)}で緊急対応が必要です。"
        
        # トレンド情報の追加
        if trend_analysis:
            increasing_trends = [
                vital for vital, data in trend_analysis.items() 
                if data['direction'] == 'increasing'
            ]
            if increasing_trends:
                summary += f" {', '.join(increasing_trends)}で上昇傾向が見られます。"
        
        return summary
    
    def _generate_action_plan(self, analysis_result: Dict, 
                            predictive_warnings: List[Dict]) -> Dict[str, List[str]]:
        """アクションプランを生成"""
        action_plan = {
            'immediate_actions': [],
            'short_term_monitoring': [],
            'long_term_considerations': []
        }
        
        overall_status = analysis_result['overall_status']
        
        # 緊急度に基づく即座のアクション
        if overall_status == 'critical':
            action_plan['immediate_actions'].extend([
                '医師への緊急連絡',
                'バイタルサインの連続監視開始',
                '必要に応じて緊急治療の準備'
            ])
        elif overall_status == 'warning':
            action_plan['immediate_actions'].extend([
                '医師への連絡',
                'バイタルサインの頻回測定'
            ])
        
        # 特定のアラートに基づくアクション
        for alert in analysis_result['alerts']:
            if alert['alert_level'] == 'critical':
                action_plan['immediate_actions'].append(
                    f"{alert['vital_type']}に対する緊急対応"
                )
        
        # 短期監視計画
        monitoring_priorities = analysis_result.get('monitoring_priorities', [])
        action_plan['short_term_monitoring'].extend([
            f"{priority}を継続" for priority in monitoring_priorities
        ])
        
        # 予測的警告に基づく長期考慮事項
        for warning in predictive_warnings:
            if warning['priority'] == 'high':
                action_plan['long_term_considerations'].append(warning['message'])
        
        return action_plan
    
    def _determine_monitoring_schedule(self, overall_status: str) -> Dict[str, str]:
        """監視スケジュールを決定"""
        schedules = {
            'critical': {
                'frequency': '連続監視',
                'next_assessment': '15分後',
                'escalation_criteria': '任意の値の更なる悪化'
            },
            'warning': {
                'frequency': '30分ごと',
                'next_assessment': '1時間後',
                'escalation_criteria': '警告レベルから緊急レベルへの変化'
            },
            'normal': {
                'frequency': '4時間ごと',
                'next_assessment': '4時間後',
                'escalation_criteria': '正常値からの逸脱'
            }
        }
        
        return schedules.get(overall_status, schedules['normal'])

# 使用例とテスト用のメイン関数
if __name__ == "__main__":
    # バイタルサイン監視システムの初期化
    vital_monitor = VitalSignsMonitor()
    
    # テスト用バイタルデータ
    test_vitals = {
        'heart_rate': 110,
        'blood_pressure_systolic': 150,
        'blood_pressure_diastolic': 95,
        'body_temperature': 38.5,
        'oxygen_saturation': 92,
        'respiratory_rate': 24
    }
    
    # テスト用患者情報
    test_patient = {
        'patient_id': 'P001',
        'age': 68,
        'chronic_diseases': ['hypertension', 'diabetes']
    }
    
    print("=== バイタルサイン監視システム デモ ===")
    
    # 包括的なバイタルレポート生成
    report = vital_monitor.generate_vital_report(test_vitals, test_patient)
    
    print(f"\nレポート生成時刻: {report['report_timestamp']}")
    print(f"患者ID: {report['patient_id']}")
    
    # 現在の分析結果
    current_analysis = report['current_analysis']
    print(f"\n【現在の状態】")
    print(f"総合状態: {current_analysis['overall_status']}")
    print(f"複合リスクスコア: {current_analysis['composite_risk_score']}")
    print(f"緊急対応必要: {'はい' if current_analysis['requires_immediate_attention'] else 'いいえ'}")
    
    # アラート情報
    alerts = current_analysis.get('alerts', [])
    if alerts:
        print(f"\n【アラート情報】({len(alerts)}件)")
        for i, alert in enumerate(alerts, 1):
            print(f"{i}. {alert['vital_type']}: {alert['message']} ({alert['alert_level']})")
    
    # 推奨事項
    recommendations = current_analysis.get('recommendations', [])
    print(f"\n【推奨事項】")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # アクションプラン
    action_plan = report.get('action_plan', {})
    print(f"\n【アクションプラン】")
    for category, actions in action_plan.items():
        if actions:
            print(f"{category.replace('_', ' ').title()}:")
            for action in actions:
                print(f"  - {action}")
    
    # 次回監視スケジュール
    monitoring = report.get('next_monitoring_schedule', {})
    if monitoring:
        print(f"\n【監視スケジュール】")
        print(f"頻度: {monitoring.get('frequency', 'unknown')}")
        print(f"次回評価: {monitoring.get('next_assessment', 'unknown')}")
