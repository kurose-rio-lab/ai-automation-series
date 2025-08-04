"""
AI分析結果処理システム
ChatGPTからの分析結果を構造化し、適切な形式で保存・配信
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
from dataclasses import dataclass, asdict

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """分析結果の構造化データクラス"""
    analysis_id: str
    timestamp: str
    analysis_type: str
    executive_summary: str
    key_insights: List[str]
    trend_analysis: Dict[str, str]
    recommendations: List[str]
    risk_factors: List[str]
    forecast: Dict[str, Any]
    confidence_score: float
    data_quality_score: float
    
class ResultProcessor:
    """分析結果処理クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.quality_threshold = config.get('quality_threshold', 0.8)
        
    def process_chatgpt_response(self, raw_response: str, metadata: Dict[str, Any]) -> AnalysisResult:
        """ChatGPT応答の処理と構造化"""
        try:
            # JSON形式の抽出
            cleaned_response = self._extract_json_from_response(raw_response)
            parsed_data = json.loads(cleaned_response)
            
            # データ品質評価
            quality_score = self._evaluate_data_quality(parsed_data)
            confidence_score = self._calculate_confidence_score(parsed_data, metadata)
            
            # 構造化データの作成
            result = AnalysisResult(
                analysis_id=self._generate_analysis_id(),
                timestamp=datetime.now().isoformat(),
                analysis_type=metadata.get('analysis_type', 'general'),
                executive_summary=self._clean_text(parsed_data.get('executive_summary', '')),
                key_insights=self._process_list_field(parsed_data.get('key_insights', [])),
                trend_analysis=self._process_trend_analysis(parsed_data.get('trend_analysis', {})),
                recommendations=self._process_list_field(parsed_data.get('recommendations', [])),
                risk_factors=self._process_list_field(parsed_data.get('risk_factors', [])),
                forecast=self._process_forecast(parsed_data.get('next_month_forecast', {})),
                confidence_score=confidence_score,
                data_quality_score=quality_score
            )
            
            logger.info(f"分析結果処理完了: {result.analysis_id}")
            return result
            
        except Exception as e:
            logger.error(f"分析結果処理エラー: {str(e)}")
            return self._create_error_result(str(e), metadata)
    
    def _extract_json_from_response(self, response: str) -> str:
        """応答からJSON部分を抽出"""
        # JSONブロックの検索パターン
        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1) if len(match.groups()) > 0 else match.group(0)
        
        # JSONパターンが見つからない場合
        raise ValueError("有効なJSONが見つかりません")
    
    def _evaluate_data_quality(self, data: Dict[str, Any]) -> float:
        """データ品質の評価"""
        score = 0.0
        max_score = 10.0
        
        # 必須フィールドの存在チェック
        required_fields = ['executive_summary', 'key_insights', 'recommendations']
        for field in required_fields:
            if field in data and data[field]:
                score += 2.0
        
        # 内容の充実度チェック
        if isinstance(data.get('key_insights'), list) and len(data['key_insights']) >= 3:
            score += 1.0
        
        if isinstance(data.get('recommendations'), list) and len(data['recommendations']) >= 2:
            score += 1.0
        
        # テキストの長さチェック
        summary_length = len(str(data.get('executive_summary', '')))
        if summary_length > 100:
            score += 1.0
        
        # 数値データの存在チェック
        if 'forecast' in data or 'trend_analysis' in data:
            score += 1.0
        
        return min(score / max_score, 1.0)
    
    def _calculate_confidence_score(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """信頼度スコアの計算"""
        base_score = 0.7
        
        # データの完全性による調整
        completeness = self._evaluate_data_quality(data)
        
        # 分析対象データの量による調整
        data_points = metadata.get('data_points', 0)
        data_score = min(data_points / 1000, 1.0) * 0.2  # 最大0.2ポイント
        
        # 分析期間による調整
        analysis_period = metadata.get('analysis_period_days', 30)
        period_score = min(analysis_period / 90, 1.0) * 0.1  # 最大0.1ポイント
        
        final_score = base_score + (completeness * 0.2) + data_score + period_score
        return min(final_score, 1.0)
    
    def _clean_text(self, text: str) -> str:
        """テキストのクリーニング"""
        if not text:
            return ""
        
        # 不要な文字の除去
        cleaned = re.sub(r'\s+', ' ', str(text))
        cleaned = cleaned.strip()
        
        # 文字数制限
        if len(cleaned) > 1000:
            cleaned = cleaned[:997] + "..."
        
        return cleaned
    
    def _process_list_field(self, field_data: Any) -> List[str]:
        """リスト形式フィールドの処理"""
        if isinstance(field_data, list):
            return [self._clean_text(item) for item in field_data if item]
        elif isinstance(field_data, str):
            # 改行区切りの文字列をリストに変換
            return [self._clean_text(item) for item in field_data.split('\n') if item.strip()]
        else:
            return []
    
    def _process_trend_analysis(self, trend_data: Any) -> Dict[str, str]:
        """トレンド分析データの処理"""
        if isinstance(trend_data, dict):
            return {
                key: self._clean_text(value) 
                for key, value in trend_data.items() 
                if value
            }
        else:
            return {}
    
    def _process_forecast(self, forecast_data: Any) -> Dict[str, Any]:
        """予測データの処理"""
        if isinstance(forecast_data, dict):
            processed = {}
            for key, value in forecast_data.items():
                if isinstance(value, (int, float)):
                    processed[key] = value
                else:
                    processed[key] = self._clean_text(str(value))
            return processed
        elif isinstance(forecast_data, str):
            return {"forecast_text": self._clean_text(forecast_data)}
        else:
            return {}
    
    def _generate_analysis_id(self) -> str:
        """分析IDの生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"analysis_{timestamp}"
    
    def _create_error_result(self, error_message: str, metadata: Dict[str, Any]) -> AnalysisResult:
        """エラー時の結果作成"""
        return AnalysisResult(
            analysis_id=self._generate_analysis_id(),
            timestamp=datetime.now().isoformat(),
            analysis_type=metadata.get('analysis_type', 'error'),
            executive_summary=f"分析エラーが発生しました: {error_message}",
            key_insights=["分析を再実行してください"],
            trend_analysis={},
            recommendations=["データを確認して再試行してください"],
            risk_factors=["分析結果の信頼性が低下しています"],
            forecast={},
            confidence_score=0.0,
            data_quality_score=0.0
        )
    
    def save_to_sheets(self, result: AnalysisResult, spreadsheet_service) -> bool:
        """Google Sheetsに結果保存"""
        try:
            # スプレッドシートの準備
            sheet_id = self.config.get('spreadsheet_id')
            worksheet_name = self.config.get('result_worksheet', '分析結果')
            
            # データの準備
            row_data = [
                result.analysis_id,
                result.timestamp,
                result.analysis_type,
                result.executive_summary,
                '\n'.join(result.key_insights),
                json.dumps(result.trend_analysis, ensure_ascii=False),
                '\n'.join(result.recommendations),
                '\n'.join(result.risk_factors),
                json.dumps(result.forecast, ensure_ascii=False),
                result.confidence_score,
                result.data_quality_score
            ]
            
            # シートに追加
            range_name = f"{worksheet_name}!A:K"
            spreadsheet_service.values().append(
                spreadsheetId=sheet_id,
                range=range_name,
                valueInputOption='USER_ENTERED',
                body={'values': [row_data]}
            ).execute()
            
            logger.info(f"Google Sheetsに保存完了: {result.analysis_id}")
            return True
            
        except Exception as e:
            logger.error(f"Google Sheets保存エラー: {str(e)}")
            return False
    
    def create_summary_report(self, results: List[AnalysisResult], period_days: int = 30) -> Dict[str, Any]:
        """サマリーレポートの作成"""
        if not results:
            return {"error": "分析結果がありません"}
        
        # 期間フィルタリング
        cutoff_date = datetime.now() - timedelta(days=period_days)
        filtered_results = [
            r for r in results 
            if datetime.fromisoformat(r.timestamp) >= cutoff_date
        ]
        
        # 統計計算
        total_analyses = len(filtered_results)
        avg_confidence = sum(r.confidence_score for r in filtered_results) / total_analyses
        avg_quality = sum(r.data_quality_score for r in filtered_results) / total_analyses
        
        # 分析タイプ別集計
        type_counts = {}
        for result in filtered_results:
            type_counts[result.analysis_type] = type_counts.get(result.analysis_type, 0) + 1
        
        # 最新の洞察集約
        recent_insights = []
        for result in sorted(filtered_results, key=lambda x: x.timestamp, reverse=True)[:5]:
            recent_insights.extend(result.key_insights[:2])  # 各結果から2つの洞察
        
        summary = {
            "period_days": period_days,
            "total_analyses": total_analyses,
            "average_confidence_score": round(avg_confidence, 3),
            "average_quality_score": round(avg_quality, 3),
            "analysis_type_distribution": type_counts,
            "recent_key_insights": recent_insights[:10],  # 最大10個
            "quality_alerts": self._generate_quality_alerts(filtered_results),
            "generated_at": datetime.now().isoformat()
        }
        
        return summary
    
    def _generate_quality_alerts(self, results: List[AnalysisResult]) -> List[str]:
        """品質アラートの生成"""
        alerts = []
        
        # 低品質結果の検出
        low_quality_count = sum(1 for r in results if r.data_quality_score < self.quality_threshold)
        if low_quality_count > len(results) * 0.2:  # 20%以上が低品質
            alerts.append(f"品質低下: {low_quality_count}件の分析で品質スコアが閾値を下回っています")
        
        # 低信頼度結果の検出
        low_confidence_count = sum(1 for r in results if r.confidence_score < self.confidence_threshold)
        if low_confidence_count > len(results) * 0.3:  # 30%以上が低信頼度
            alerts.append(f"信頼度低下: {low_confidence_count}件の分析で信頼度スコアが閾値を下回っています")
        
        # エラー結果の検出
        error_count = sum(1 for r in results if r.analysis_type == 'error')
        if error_count > 0:
            alerts.append(f"エラー発生: {error_count}件の分析でエラーが発生しています")
        
        return alerts

class ReportFormatter:
    """レポート形式変換クラス"""
    
    @staticmethod
    def to_html(result: AnalysisResult) -> str:
        """HTML形式に変換"""
        html_template = """
        <div class="analysis-report">
            <h2>分析レポート #{analysis_id}</h2>
            <div class="metadata">
                <p><strong>分析日時:</strong> {timestamp}</p>
                <p><strong>信頼度:</strong> {confidence_score:.1%}</p>
                <p><strong>品質スコア:</strong> {data_quality_score:.1%}</p>
            </div>
            
            <div class="executive-summary">
                <h3>エグゼクティブサマリー</h3>
                <p>{executive_summary}</p>
            </div>
            
            <div class="key-insights">
                <h3>主要洞察</h3>
                <ul>
                    {insights_list}
                </ul>
            </div>
            
            <div class="recommendations">
                <h3>推奨アクション</h3>
                <ul>
                    {recommendations_list}
                </ul>
            </div>
            
            <div class="risk-factors">
                <h3>リスク要因</h3>
                <ul>
                    {risks_list}
                </ul>
            </div>
        </div>
        """
        
        insights_list = ''.join(f"<li>{insight}</li>" for insight in result.key_insights)
        recommendations_list = ''.join(f"<li>{rec}</li>" for rec in result.recommendations)
        risks_list = ''.join(f"<li>{risk}</li>" for risk in result.risk_factors)
        
        return html_template.format(
            analysis_id=result.analysis_id,
            timestamp=result.timestamp,
            confidence_score=result.confidence_score,
            data_quality_score=result.data_quality_score,
            executive_summary=result.executive_summary,
            insights_list=insights_list,
            recommendations_list=recommendations_list,
            risks_list=risks_list
        )
    
    @staticmethod
    def to_markdown(result: AnalysisResult) -> str:
        """Markdown形式に変換"""
        md_content = f"""# 分析レポート #{result.analysis_id}

**分析日時**: {result.timestamp}  
**信頼度**: {result.confidence_score:.1%}  
**品質スコア**: {result.data_quality_score:.1%}

## エグゼクティブサマリー
{result.executive_summary}

## 主要洞察
"""
        for insight in result.key_insights:
            md_content += f"- {insight}\n"
        
        md_content += "\n## 推奨アクション\n"
        for rec in result.recommendations:
            md_content += f"- {rec}\n"
        
        md_content += "\n## リスク要因\n"
        for risk in result.risk_factors:
            md_content += f"- {risk}\n"
        
        return md_content

# 使用例
if __name__ == "__main__":
    # 設定
    config = {
        'confidence_threshold': 0.7,
        'quality_threshold': 0.8,
        'spreadsheet_id': 'your_spreadsheet_id',
        'result_worksheet': '分析結果'
    }
    
    # プロセッサ初期化
    processor = ResultProcessor(config)
    
    # サンプル応答の処理
    sample_response = """
    {
        "executive_summary": "今月の売上は前月比15%増加し、目標を上回る結果となりました。",
        "key_insights": [
            "新商品Aの売上が全体の30%を占める",
            "リピート顧客の購入単価が20%向上",
            "オンライン売上の成長率が店舗売上を上回る"
        ],
        "recommendations": [
            "新商品Aの在庫確保と販売チャネル拡大",
            "リピート顧客向けの特別プロモーション実施"
        ]
    }
    """
    
    metadata = {
        'analysis_type': 'monthly_sales',
        'data_points': 1500,
        'analysis_period_days': 30
    }
    
    # 処理実行
    result = processor.process_chatgpt_response(sample_response, metadata)
    print(f"処理完了: {result.analysis_id}")
    print(f"信頼度: {result.confidence_score:.2f}")
