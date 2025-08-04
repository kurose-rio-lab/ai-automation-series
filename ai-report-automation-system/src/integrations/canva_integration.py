"""
Canva API連携システム
プロフェッショナルなレポートデザインの自動生成
"""

import requests
import json
import time
import base64
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from io import BytesIO

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CanvaDesign:
    """Canvaデザインデータクラス"""
    design_id: str
    title: str
    design_type: str
    thumbnail_url: str
    export_url: str
    created_at: str

class CanvaIntegration:
    """Canva API連携クラス"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.canva.com/rest/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # デザインテンプレート設定
        self.templates = self._load_template_config()
        
    def _load_template_config(self) -> Dict[str, Any]:
        """テンプレート設定の読み込み"""
        return {
            "executive_report": {
                "template_id": "EXEC_REPORT_001",
                "dimensions": {"width": 1920, "height": 1080},
                "color_scheme": "corporate_blue",
                "font_scheme": "professional"
            },
            "sales_dashboard": {
                "template_id": "SALES_DASH_001", 
                "dimensions": {"width": 1600, "height": 900},
                "color_scheme": "success_green",
                "font_scheme": "modern"
            },
            "marketing_report": {
                "template_id": "MARKETING_001",
                "dimensions": {"width": 1920, "height": 1080},
                "color_scheme": "vibrant_orange",
                "font_scheme": "creative"
            },
            "financial_summary": {
                "template_id": "FINANCIAL_001",
                "dimensions": {"width": 1200, "height": 800},
                "color_scheme": "professional_navy",
                "font_scheme": "conservative"
            }
        }
    
    def create_design_from_template(
        self, 
        template_type: str, 
        content_data: Dict[str, Any],
        title: str = "AI Generated Report"
    ) -> CanvaDesign:
        """テンプレートからデザイン作成"""
        try:
            template_config = self.templates.get(template_type)
            if not template_config:
                raise ValueError(f"Unknown template type: {template_type}")
            
            # デザイン作成リクエスト
            create_payload = {
                "design_type": template_type,
                "title": title,
                "template_id": template_config["template_id"],
                "dimensions": template_config["dimensions"]
            }
            
            response = requests.post(
                f"{self.base_url}/designs",
                headers=self.headers,
                json=create_payload
            )
            response.raise_for_status()
            
            design_data = response.json()
            design_id = design_data["id"]
            
            logger.info(f"デザイン作成完了: {design_id}")
            
            # コンテンツの挿入
            self._populate_design_content(design_id, content_data, template_config)
            
            # デザイン情報の取得
            design_info = self._get_design_info(design_id)
            
            return CanvaDesign(
                design_id=design_id,
                title=title,
                design_type=template_type,
                thumbnail_url=design_info.get("thumbnail_url", ""),
                export_url="",  # 後で設定
                created_at=design_info.get("created_at", "")
            )
            
        except Exception as e:
            logger.error(f"デザイン作成エラー: {str(e)}")
            raise
    
    def _populate_design_content(
        self, 
        design_id: str, 
        content_data: Dict[str, Any],
        template_config: Dict[str, Any]
    ):
        """デザインにコンテンツを挿入"""
        try:
            # テキスト要素の更新
            if "text_elements" in content_data:
                self._update_text_elements(design_id, content_data["text_elements"])
            
            # 画像要素の更新
            if "image_elements" in content_data:
                self._update_image_elements(design_id, content_data["image_elements"])
            
            # グラフ・チャートの挿入
            if "chart_data" in content_data:
                self._insert_charts(design_id, content_data["chart_data"])
            
            # カラースキームの適用
            self._apply_color_scheme(design_id, template_config["color_scheme"])
            
            logger.info(f"コンテンツ挿入完了: {design_id}")
            
        except Exception as e:
            logger.error(f"コンテンツ挿入エラー: {str(e)}")
            raise
    
    def _update_text_elements(self, design_id: str, text_elements: List[Dict[str, Any]]):
        """テキスト要素の更新"""
        for element in text_elements:
            payload = {
                "element_id": element["id"],
                "text": element["content"],
                "font_size": element.get("font_size", 16),
                "font_weight": element.get("font_weight", "normal"),
                "color": element.get("color", "#000000")
            }
            
            response = requests.patch(
                f"{self.base_url}/designs/{design_id}/elements/{element['id']}",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
    
    def _update_image_elements(self, design_id: str, image_elements: List[Dict[str, Any]]):
        """画像要素の更新"""
        for element in image_elements:
            if "url" in element:
                # URL指定の画像
                payload = {
                    "element_id": element["id"],
                    "image_url": element["url"]
                }
            elif "base64" in element:
                # Base64エンコードされた画像
                payload = {
                    "element_id": element["id"],
                    "image_data": element["base64"]
                }
            else:
                continue
            
            response = requests.patch(
                f"{self.base_url}/designs/{design_id}/elements/{element['id']}",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
    
    def _insert_charts(self, design_id: str, chart_data: List[Dict[str, Any]]):
        """グラフ・チャートの挿入"""
        for chart in chart_data:
            # グラフ画像の生成（Chart.jsまたは他のライブラリを使用）
            chart_image = self._generate_chart_image(chart)
            
            # グラフ要素の追加
            payload = {
                "type": "image",
                "image_data": chart_image,
                "position": chart.get("position", {"x": 100, "y": 100}),
                "dimensions": chart.get("dimensions", {"width": 400, "height": 300})
            }
            
            response = requests.post(
                f"{self.base_url}/designs/{design_id}/elements",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
    
    def _generate_chart_image(self, chart_config: Dict[str, Any]) -> str:
        """グラフ画像の生成"""
        # この部分は実際の実装では Chart.js や matplotlib を使用
        # ここではダミー実装
        chart_type = chart_config.get("type", "bar")
        data = chart_config.get("data", [])
        
        # 実際の実装例（matplotlib使用）
        """
        import matplotlib.pyplot as plt
        import io
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if chart_type == "bar":
            ax.bar(data["labels"], data["values"])
        elif chart_type == "line":
            ax.plot(data["labels"], data["values"])
        elif chart_type == "pie":
            ax.pie(data["values"], labels=data["labels"])
        
        # 画像をBase64に変換
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
        """
        
        # ダミーデータを返す
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    def _apply_color_scheme(self, design_id: str, color_scheme: str):
        """カラースキームの適用"""
        color_schemes = {
            "corporate_blue": {
                "primary": "#1E3A8A",
                "secondary": "#3B82F6", 
                "accent": "#EBF8FF",
                "text": "#1F2937"
            },
            "success_green": {
                "primary": "#059669",
                "secondary": "#10B981",
                "accent": "#ECFDF5",
                "text": "#1F2937"
            },
            "vibrant_orange": {
                "primary": "#EA580C",
                "secondary": "#FB923C",
                "accent": "#FFF7ED",
                "text": "#1F2937"
            },
            "professional_navy": {
                "primary": "#1E293B",
                "secondary": "#475569",
                "accent": "#F8FAFC",
                "text": "#0F172A"
            }
        }
        
        colors = color_schemes.get(color_scheme, color_schemes["corporate_blue"])
        
        # カラーパレットの適用
        payload = {
            "color_palette": colors
        }
        
        response = requests.patch(
            f"{self.base_url}/designs/{design_id}/style",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
    
    def _get_design_info(self, design_id: str) -> Dict[str, Any]:
        """デザイン情報の取得"""
        response = requests.get(
            f"{self.base_url}/designs/{design_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def export_design(
        self, 
        design_id: str, 
        format: str = "pdf",
        quality: str = "high"
    ) -> str:
        """デザインのエクスポート"""
        try:
            # エクスポートリクエスト
            export_payload = {
                "format": format,
                "quality": quality,
                "pages": "all"
            }
            
            response = requests.post(
                f"{self.base_url}/designs/{design_id}/export",
                headers=self.headers,
                json=export_payload
            )
            response.raise_for_status()
            
            export_data = response.json()
            export_id = export_data["export_id"]
            
            # エクスポート完了まで待機
            export_url = self._wait_for_export_completion(export_id)
            
            logger.info(f"エクスポート完了: {export_url}")
            return export_url
            
        except Exception as e:
            logger.error(f"エクスポートエラー: {str(e)}")
            raise
    
    def _wait_for_export_completion(self, export_id: str, max_wait_time: int = 300) -> str:
        """エクスポート完了の待機"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = requests.get(
                f"{self.base_url}/exports/{export_id}",
                headers=self.headers
            )
            response.raise_for_status()
            
            export_status = response.json()
            
            if export_status["status"] == "completed":
                return export_status["download_url"]
            elif export_status["status"] == "failed":
                raise Exception(f"エクスポート失敗: {export_status.get('error', 'Unknown error')}")
            
            # 5秒待機
            time.sleep(5)
            
        raise TimeoutError("エクスポートがタイムアウトしました")
    
    def create_executive_report(self, analysis_data: Dict[str, Any]) -> CanvaDesign:
        """エグゼクティブレポートの作成"""
        content_data = {
            "text_elements": [
                {
                    "id": "title",
                    "content": f"月次業績レポート - {analysis_data.get('period', '2024年1月')}",
                    "font_size": 24,
                    "font_weight": "bold"
                },
                {
                    "id": "executive_summary",
                    "content": analysis_data.get("executive_summary", ""),
                    "font_size": 16
                },
                {
                    "id": "key_insights",
                    "content": "\n".join([f"• {insight}" for insight in analysis_data.get("key_insights", [])]),
                    "font_size": 14
                },
                {
                    "id": "recommendations",
                    "content": "\n".join([f"▶ {rec}" for rec in analysis_data.get("recommendations", [])]),
                    "font_size": 14
                }
            ],
            "chart_data": [
                {
                    "type": "bar",
                    "data": analysis_data.get("sales_data", {}),
                    "position": {"x": 50, "y": 300},
                    "dimensions": {"width": 400, "height": 250}
                },
                {
                    "type": "line",
                    "data": analysis_data.get("trend_data", {}),
                    "position": {"x": 500, "y": 300},
                    "dimensions": {"width": 400, "height": 250}
                }
            ]
        }
        
        return self.create_design_from_template(
            "executive_report",
            content_data,
            f"エグゼクティブレポート - {analysis_data.get('period', '2024年1月')}"
        )
    
    def create_sales_dashboard(self, sales_data: Dict[str, Any]) -> CanvaDesign:
        """売上ダッシュボードの作成"""
        content_data = {
            "text_elements": [
                {
                    "id": "title",
                    "content": "売上ダッシュボード",
                    "font_size": 28,
                    "font_weight": "bold"
                },
                {
                    "id": "total_sales",
                    "content": f"総売上: {sales_data.get('total_sales', '0')}万円",
                    "font_size": 20,
                    "color": "#059669"
                },
                {
                    "id": "growth_rate",
                    "content": f"成長率: {sales_data.get('growth_rate', '0')}%",
                    "font_size": 18
                }
            ],
            "chart_data": [
                {
                    "type": "pie",
                    "data": sales_data.get("product_breakdown", {}),
                    "position": {"x": 50, "y": 200},
                    "dimensions": {"width": 350, "height": 350}
                },
                {
                    "type": "bar",
                    "data": sales_data.get("monthly_trend", {}),
                    "position": {"x": 450, "y": 200},
                    "dimensions": {"width": 450, "height": 350}
                }
            ]
        }
        
        return self.create_design_from_template(
            "sales_dashboard",
            content_data,
            "売上ダッシュボード"
        )
    
    def create_marketing_report(self, marketing_data: Dict[str, Any]) -> CanvaDesign:
        """マーケティングレポートの作成"""
        content_data = {
            "text_elements": [
                {
                    "id": "title",
                    "content": "マーケティング成果レポート",
                    "font_size": 24,
                    "font_weight": "bold"
                },
                {
                    "id": "roi_summary",
                    "content": f"総ROI: {marketing_data.get('total_roi', '0')}%",
                    "font_size": 20,
                    "color": "#EA580C"
                },
                {
                    "id": "channel_performance",
                    "content": self._format_channel_performance(marketing_data.get("channels", {})),
                    "font_size": 14
                }
            ],
            "chart_data": [
                {
                    "type": "bar",
                    "data": marketing_data.get("channel_roi", {}),
                    "position": {"x": 50, "y": 250},
                    "dimensions": {"width": 400, "height": 300}
                },
                {
                    "type": "line",
                    "data": marketing_data.get("cost_trend", {}),
                    "position": {"x": 500, "y": 250},
                    "dimensions": {"width": 400, "height": 300}
                }
            ]
        }
        
        return self.create_design_from_template(
            "marketing_report",
            content_data,
            "マーケティングレポート"
        )
    
    def _format_channel_performance(self, channels: Dict[str, Any]) -> str:
        """チャネルパフォーマンスのフォーマット"""
        formatted = []
        for channel, data in channels.items():
            roi = data.get("roi", 0)
            cost = data.get("cost", 0)
            formatted.append(f"{channel}: ROI {roi}% (コスト: {cost}万円)")
        return "\n".join(formatted)
    
    def batch_create_reports(
        self, 
        analysis_results: List[Dict[str, Any]]
    ) -> List[CanvaDesign]:
        """バッチレポート作成"""
        designs = []
        
        for result in analysis_results:
            try:
                report_type = result.get("report_type", "executive_report")
                
                if report_type == "executive_report":
                    design = self.create_executive_report(result)
                elif report_type == "sales_dashboard":
                    design = self.create_sales_dashboard(result)
                elif report_type == "marketing_report":
                    design = self.create_marketing_report(result)
                else:
                    continue
                
                designs.append(design)
                logger.info(f"レポート作成完了: {design.design_id}")
                
                # API制限対応の待機
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"レポート作成エラー: {str(e)}")
                continue
        
        return designs

# 使用例
if __name__ == "__main__":
    # 設定
    api_key = "your_canva_api_key"
    canva = CanvaIntegration(api_key)
    
    # サンプルデータ
    sample_analysis = {
        "period": "2024年1月",
        "executive_summary": "今月の売上は前月比15%増加し、目標を上回る結果となりました。",
        "key_insights": [
            "新商品Aの売上が全体の30%を占める",
            "リピート顧客の購入単価が20%向上",
            "オンライン売上の成長率が店舗売上を上回る"
        ],
        "recommendations": [
            "新商品Aの在庫確保と販売チャネル拡大",
            "リピート顧客向けの特別プロモーション実施"
        ],
        "sales_data": {
            "labels": ["商品A", "商品B", "商品C"],
            "values": [300, 250, 200]
        },
        "trend_data": {
            "labels": ["1月", "2月", "3月"],
            "values": [1000, 1150, 1320]
        }
    }
    
    # エグゼクティブレポート作成
    design = canva.create_executive_report(sample_analysis)
    print(f"デザイン作成完了: {design.design_id}")
    
    # PDF出力
    export_url = canva.export_design(design.design_id, "pdf")
    print(f"PDF出力完了: {export_url}")
