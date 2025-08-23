"""
AI小売システム統合モジュール
各AIエンジンを統合し、リアルタイムでの意思決定を支援
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

from models.inventory_optimization import InventoryOptimizer
from models.demand_forecasting import DemandForecaster
from models.price_optimization import PriceOptimizer
from models.recommendation_engine import RecommendationEngine

@dataclass
class SystemDecision:
    """システム判断結果"""
    product_id: str
    recommended_action: str
    priority: int
    expected_impact: float
    reasoning: str
    timestamp: datetime

class AIRetailSystemIntegrator:
    """AI小売システム統合クラス"""
    
    def __init__(self):
        self.inventory_optimizer = InventoryOptimizer()
        self.demand_forecaster = DemandForecaster()
        self.price_optimizer = PriceOptimizer()
        self.recommendation_engine = RecommendationEngine()
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def comprehensive_analysis(self, product_id: str) -> Dict:
        """製品に対する包括的分析"""
        try:
            # 各エンジンからの分析結果を取得
            inventory_analysis = self._get_inventory_analysis(product_id)
            demand_analysis = self._get_demand_analysis(product_id)
            price_analysis = self._get_price_analysis(product_id)
            recommendation_analysis = self._get_recommendation_analysis(product_id)
            
            # 統合判断
            integrated_decision = self._make_integrated_decision(
                product_id,
                inventory_analysis,
                demand_analysis,
                price_analysis,
                recommendation_analysis
            )
            
            return {
                'product_id': product_id,
                'timestamp': datetime.now(),
                'inventory_analysis': inventory_analysis,
                'demand_analysis': demand_analysis,
                'price_analysis': price_analysis,
                'recommendation_analysis': recommendation_analysis,
                'integrated_decision': integrated_decision,
                'confidence_score': self._calculate_confidence_score(integrated_decision)
            }
            
        except Exception as e:
            self.logger.error(f"包括的分析でエラー: {e}")
            return {'error': str(e)}
    
    def _get_inventory_analysis(self, product_id: str) -> Dict:
        """在庫分析の取得"""
        try:
            # サンプルデータ読み込み
            inventory_df = pd.read_csv('data/sample_data/inventory.csv')
            product_inventory = inventory_df[inventory_df['product_id'] == product_id]
            
            if product_inventory.empty:
                return {'status': 'no_data', 'message': '在庫データが見つかりません'}
            
            current_stock = product_inventory['current_stock'].iloc[0]
            min_stock = product_inventory['min_stock'].iloc[0]
            max_stock = product_inventory['max_stock'].iloc[0]
            
            # 在庫状況判定
            stock_ratio = current_stock / max_stock
            
            if current_stock <= min_stock:
                status = 'critical_low'
                action = 'immediate_reorder'
                priority = 1
            elif stock_ratio < 0.3:
                status = 'low'
                action = 'schedule_reorder'
                priority = 2
            elif stock_ratio > 0.8:
                status = 'high'
                action = 'reduce_ordering'
                priority = 3
            else:
                status = 'normal'
                action = 'maintain'
                priority = 4
            
            return {
                'status': status,
                'current_stock': current_stock,
                'stock_ratio': stock_ratio,
                'recommended_action': action,
                'priority': priority,
                'reorder_quantity': max(0, max_stock - current_stock) if status in ['critical_low', 'low'] else 0
            }
            
        except Exception as e:
            self.logger.error(f"在庫分析エラー: {e}")
            return {'error': str(e)}
    
    def _get_demand_analysis(self, product_id: str) -> Dict:
        """需要分析の取得"""
        try:
            # 過去の販売データから需要予測
            sales_df = pd.read_csv('data/sample_data/sales_history.csv')
            product_sales = sales_df[sales_df['product_id'] == product_id]
            
            if product_sales.empty:
                return {'status': 'no_data', 'predicted_demand': 0, 'trend': 'unknown'}
            
            # 簡易的な需要予測（実際はより複雑なモデルを使用）
            daily_sales = product_sales.groupby('sale_date')['quantity'].sum()
            avg_daily_demand = daily_sales.mean()
            recent_trend = 'increasing' if daily_sales.iloc[-3:].mean() > daily_sales.iloc[-7:-3].mean() else 'decreasing'
            
            # 7日間の需要予測
            predicted_weekly_demand = avg_daily_demand * 7
            if recent_trend == 'increasing':
                predicted_weekly_demand *= 1.2
            elif recent_trend == 'decreasing':
                predicted_weekly_demand *= 0.8
            
            return {
                'status': 'calculated',
                'avg_daily_demand': round(avg_daily_demand, 2),
                'predicted_weekly_demand': round(predicted_weekly_demand, 2),
                'trend': recent_trend,
                'confidence': 0.75  # 実際は予測精度に基づく
            }
            
        except Exception as e:
            self.logger.error(f"需要分析エラー: {e}")
            return {'error': str(e)}
    
    def _get_price_analysis(self, product_id: str) -> Dict:
        """価格分析の取得"""
        try:
            # 製品データと販売履歴を読み込み
            products_df = pd.read_csv('data/sample_data/products.csv')
            sales_df = pd.read_csv('data/sample_data/sales_history.csv')
            
            product_info = products_df[products_df['product_id'] == product_id]
            product_sales = sales_df[sales_df['product_id'] == product_id]
            
            if product_info.empty:
                return {'status': 'no_data', 'message': '製品データが見つかりません'}
            
            current_price = product_info['price'].iloc[0]
            cost = product_info['cost'].iloc[0]
            current_margin = (current_price - cost) / current_price * 100
            
            # 価格弾性の簡易計算
            if not product_sales.empty:
                price_performance = len(product_sales) / (datetime.now() - pd.to_datetime('2023-10-01')).days
                
                if current_margin < 20:
                    recommendation = 'increase_price'
                    suggested_price = current_price * 1.05
                elif current_margin > 40:
                    recommendation = 'decrease_price'
                    suggested_price = current_price * 0.95
                else:
                    recommendation = 'maintain_price'
                    suggested_price = current_price
            else:
                recommendation = 'maintain_price'
                suggested_price = current_price
            
            return {
                'status': 'calculated',
                'current_price': current_price,
                'current_margin': round(current_margin, 2),
                'recommended_action': recommendation,
                'suggested_price': round(suggested_price, 0),
                'expected_margin': round((suggested_price - cost) / suggested_price * 100, 2)
            }
            
        except Exception as e:
            self.logger.error(f"価格分析エラー: {e}")
            return {'error': str(e)}
    
    def _get_recommendation_analysis(self, product_id: str) -> Dict:
        """レコメンド分析の取得"""
        try:
            # 顧客の購買パターンから関連商品を分析
            sales_df = pd.read_csv('data/sample_data/sales_history.csv')
            
            # このproduct_idを購入した顧客
            customers_bought_this = sales_df[sales_df['product_id'] == product_id]['customer_id'].unique()
            
            if len(customers_bought_this) == 0:
                return {'status': 'no_data', 'related_products': []}
            
            # これらの顧客が購入した他の商品
            related_purchases = sales_df[
                (sales_df['customer_id'].isin(customers_bought_this)) & 
                (sales_df['product_id'] != product_id)
            ]
            
            # 関連商品の頻度分析
            related_products = related_purchases['product_id'].value_counts().head(3).to_dict()
            
            return {
                'status': 'calculated',
                'customers_count': len(customers_bought_this),
                'related_products': related_products,
                'cross_sell_opportunity': len(related_products) > 0
            }
            
        except Exception as e:
            self.logger.error(f"レコメンド分析エラー: {e}")
            return {'error': str(e)}
    
    def _make_integrated_decision(self, product_id: str, inventory: Dict, 
                                demand: Dict, price: Dict, recommendation: Dict) -> SystemDecision:
        """各分析結果を統合して最終判断を作成"""
        
        # 優先度スコア計算
        priority_score = 0
        actions = []
        reasoning_parts = []
        
        # 在庫状況による判断
        if inventory.get('status') == 'critical_low':
            priority_score += 10
            actions.append('緊急発注')
            reasoning_parts.append('在庫が危険レベル')
        elif inventory.get('status') == 'low':
            priority_score += 5
            actions.append('発注計画')
            reasoning_parts.append('在庫が少ない')
        
        # 需要予測による判断
        if demand.get('trend') == 'increasing':
            priority_score += 3
            actions.append('需要増加対応')
            reasoning_parts.append('需要が増加傾向')
        elif demand.get('trend') == 'decreasing':
            priority_score -= 2
            reasoning_parts.append('需要が減少傾向')
        
        # 価格最適化による判断
        if price.get('recommended_action') == 'increase_price':
            actions.append('価格上昇')
            reasoning_parts.append('マージンが低い')
        elif price.get('recommended_action') == 'decrease_price':
            actions.append('価格下降')
            reasoning_parts.append('マージンが高すぎる')
        
        # 統合アクション決定
        if priority_score >= 8:
            main_action = '緊急対応が必要'
            priority = 1
        elif priority_score >= 4:
            main_action = '注意深い監視が必要'
            priority = 2
        else:
            main_action = '現状維持'
            priority = 3
        
        # 期待効果の計算（簡易版）
        expected_impact = min(priority_score * 0.1, 1.0)
        
        return SystemDecision(
            product_id=product_id,
            recommended_action=main_action,
            priority=priority,
            expected_impact=expected_impact,
            reasoning=' | '.join(reasoning_parts) if reasoning_parts else '正常範囲内',
            timestamp=datetime.now()
        )
    
    def _calculate_confidence_score(self, decision: SystemDecision) -> float:
        """判断の信頼度スコア計算"""
        base_confidence = 0.7
        
        # 優先度が高いほど信頼度も高い
        priority_bonus = (4 - decision.priority) * 0.1
        
        return min(base_confidence + priority_bonus, 1.0)
    
    def batch_analysis(self, product_ids: List[str]) -> List[Dict]:
        """複数製品の一括分析"""
        results = []
        for product_id in product_ids:
            result = self.comprehensive_analysis(product_id)
            results.append(result)
        
        return results
    
    def get_dashboard_summary(self) -> Dict:
        """ダッシュボード用サマリー情報"""
        try:
            # 全製品の一括分析
            products_df = pd.read_csv('data/sample_data/products.csv')
            all_products = products_df['product_id'].tolist()
            
            # 主要製品（上位5製品）を分析
            key_products = all_products[:5]
            analysis_results = self.batch_analysis(key_products)
            
            # サマリー統計
            high_priority_count = sum(1 for result in analysis_results 
                                    if result.get('integrated_decision', {}).priority == 1)
            
            avg_confidence = sum(result.get('confidence_score', 0) for result in analysis_results) / len(analysis_results)
            
            return {
                'timestamp': datetime.now(),
                'total_products_analyzed': len(analysis_results),
                'high_priority_items': high_priority_count,
                'average_confidence': round(avg_confidence, 3),
                'system_status': 'operational',
                'key_insights': self._extract_key_insights(analysis_results)
            }
            
        except Exception as e:
            self.logger.error(f"ダッシュボードサマリーエラー: {e}")
            return {'error': str(e)}
    
    def _extract_key_insights(self, analysis_results: List[Dict]) -> List[str]:
        """分析結果から主要な洞察を抽出"""
        insights = []
        
        # 在庫不足の製品数
        low_stock_count = sum(1 for result in analysis_results 
                             if result.get('inventory_analysis', {}).get('status') in ['critical_low', 'low'])
        
        if low_stock_count > 0:
            insights.append(f"{low_stock_count}製品で在庫不足が発生中")
        
        # 需要増加傾向の製品数
        increasing_demand_count = sum(1 for result in analysis_results 
                                    if result.get('demand_analysis', {}).get('trend') == 'increasing')
        
        if increasing_demand_count > 0:
            insights.append(f"{increasing_demand_count}製品で需要が増加傾向")
        
        # 価格調整が推奨される製品数
        price_adjustment_count = sum(1 for result in analysis_results 
                                   if result.get('price_analysis', {}).get('recommended_action') != 'maintain_price')
        
        if price_adjustment_count > 0:
            insights.append(f"{price_adjustment_count}製品で価格調整を推奨")
        
        if not insights:
            insights.append("すべての製品が正常範囲内で運用中")
        
        return insights

if __name__ == "__main__":
    # システム統合のテスト実行
    integrator = AIRetailSystemIntegrator()
    
    # 単一製品分析のテスト
    result = integrator.comprehensive_analysis('P001')
    print("=== 単一製品分析結果 ===")
    print(f"製品ID: {result['product_id']}")
    print(f"統合判断: {result.get('integrated_decision', {})}")
    print(f"信頼度: {result.get('confidence_score', 0)}")
    
    # ダッシュボードサマリーのテスト
    dashboard = integrator.get_dashboard_summary()
    print("\n=== ダッシュボードサマリー ===")
    for key, value in dashboard.items():
        print(f"{key}: {value}")
