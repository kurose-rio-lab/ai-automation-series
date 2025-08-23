"""
システム統合のテスト
"""

import unittest
import pandas as pd
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from integration.system_integration import AIRetailSystemIntegrator

class TestSystemIntegration(unittest.TestCase):
    """システム統合テストクラス"""
    
    def setUp(self):
        """テストセットアップ"""
        self.integrator = AIRetailSystemIntegrator()
    
    def test_comprehensive_analysis(self):
        """包括的分析のテスト"""
        result = self.integrator.comprehensive_analysis('P001')
        
        # 基本構造の確認
        expected_keys = [
            'product_id', 'timestamp', 'inventory_analysis',
            'demand_analysis', 'price_analysis', 'recommendation_analysis',
            'integrated_decision', 'confidence_score'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
        
        # 製品IDの一致確認
        self.assertEqual(result['product_id'], 'P001')
        
        # 信頼度スコアの範囲確認
        self.assertGreaterEqual(result['confidence_score'], 0)
        self.assertLessEqual(result['confidence_score'], 1)
    
    def test_inventory_analysis(self):
        """在庫分析のテスト"""
        result = self.integrator._get_inventory_analysis('P001')
        
        # 基本構造の確認
        expected_keys = ['status', 'current_stock', 'stock_ratio', 'recommended_action', 'priority']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # 在庫比率の範囲確認
        if 'stock_ratio' in result:
            self.assertGreaterEqual(result['stock_ratio'], 0)
    
    def test_demand_analysis(self):
        """需要分析のテスト"""
        result = self.integrator._get_demand_analysis('P001')
        
        if result.get('status') == 'calculated':
            # 需要予測値の確認
            self.assertGreaterEqual(result['avg_daily_demand'], 0)
            self.assertGreaterEqual(result['predicted_weekly_demand'], 0)
            self.assertIn(result['trend'], ['increasing', 'decreasing'])
            
            # 信頼度の範囲確認
            self.assertGreaterEqual(result['confidence'], 0)
            self.assertLessEqual(result['confidence'], 1)
    
    def test_price_analysis(self):
        """価格分析のテスト"""
        result = self.integrator._get_price_analysis('P001')
        
        if result.get('status') == 'calculated':
            # 価格情報の確認
            self.assertGreater(result['current_price'], 0)
            self.assertGreaterEqual(result['current_margin'], 0)
            self.assertGreater(result['suggested_price'], 0)
            
            # 推奨アクションの確認
            valid_actions = ['increase_price', 'decrease_price', 'maintain_price']
            self.assertIn(result['recommended_action'], valid_actions)
    
    def test_batch_analysis(self):
        """一括分析のテスト"""
        product_ids = ['P001', 'P002', 'P003']
        results = self.integrator.batch_analysis(product_ids)
        
        # 結果数の確認
        self.assertEqual(len(results), len(product_ids))
        
        # 各結果の構造確認
        for i, result in enumerate(results):
            self.assertEqual(result['product_id'], product_ids[i])
            self.assertIn('integrated_decision', result)
    
    def test_dashboard_summary(self):
        """ダッシュボードサマリーのテスト"""
        summary = self.integrator.get_dashboard_summary()
        
        # 基本構造の確認
        expected_keys = [
            'timestamp', 'total_products_analyzed', 'high_priority_items',
            'average_confidence', 'system_status', 'key_insights'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # 数値の妥当性確認
        self.assertGreaterEqual(summary['total_products_analyzed'], 0)
        self.assertGreaterEqual(summary['high_priority_items'], 0)
        self.assertGreaterEqual(summary['average_confidence'], 0)
        self.assertLessEqual(summary['average_confidence'], 1)
    
    def test_integrated_decision_logic(self):
        """統合判断ロジックのテスト"""
        # テスト用の分析結果
        mock_inventory = {'status': 'critical_low', 'priority': 1}
        mock_demand = {'trend': 'increasing'}
        mock_price = {'recommended_action': 'increase_price'}
        mock_recommendation = {'cross_sell_opportunity': True}
        
        decision = self.integrator._make_integrated_decision(
            'P001', mock_inventory, mock_demand, mock_price, mock_recommendation
        )
        
        # 判断結果の確認
        self.assertEqual(decision.product_id, 'P001')
        self.assertIn(decision.priority, [1, 2, 3])
        self.assertGreaterEqual(decision.expected_impact, 0)
        self.assertLessEqual(decision.expected_impact, 1)
        self.assertIsNotNone(decision.reasoning)
    
    def test_confidence_score_calculation(self):
        """信頼度スコア計算のテスト"""
        from integration.system_integration import SystemDecision
        from datetime import datetime
        
        # 高優先度の判断
        high_priority_decision = SystemDecision(
            product_id='P001',
            recommended_action='緊急対応が必要',
            priority=1,
            expected_impact=0.8,
            reasoning='在庫が危険レベル',
            timestamp=datetime.now()
        )
        
        confidence = self.integrator._calculate_confidence_score(high_priority_decision)
        
        # 信頼度の範囲確認
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
        
        # 高優先度では高い信頼度が期待される
        self.assertGreater(confidence, 0.7)
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 存在しない製品IDでのテスト
        result = self.integrator.comprehensive_analysis('INVALID_ID')
        
        # エラーが適切に処理されていることを確認
        # （存在しない製品でも基本構造は維持されるべき）
        self.assertIn('product_id', result)
    
    def test_performance(self):
        """パフォーマンステスト"""
        import time
        
        # 複数製品の一括処理時間測定
        product_ids = ['P001', 'P002', 'P003', 'P004', 'P005']
        
        start_time = time.time()
        results = self.integrator.batch_analysis(product_ids)
        end_time = time.time()
        
        # 処理時間の確認（5製品で30秒以内）
        self.assertLess(end_time - start_time, 30)
        self.assertEqual(len(results), len(product_ids))

if __name__ == '__main__':
    unittest.main()
