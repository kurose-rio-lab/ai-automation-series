"""
在庫最適化システムのユニットテスト
"""

import unittest
import pandas as pd
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.inventory_optimization import InventoryOptimizer

class TestInventoryOptimization(unittest.TestCase):
    """在庫最適化テストクラス"""
    
    def setUp(self):
        """テストセットアップ"""
        self.optimizer = InventoryOptimizer()
        
        # テストデータの作成
        self.test_inventory_data = pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003'],
            'current_stock': [45, 78, 123],
            'min_stock': [10, 20, 30],
            'max_stock': [100, 150, 200],
            'warehouse_location': ['WH_Tokyo', 'WH_Osaka', 'WH_Tokyo'],
            'last_updated': ['2023-10-08', '2023-10-08', '2023-10-08']
        })
        
        self.test_sales_data = pd.DataFrame({
            'product_id': ['P001', 'P001', 'P002', 'P002', 'P003'],
            'quantity': [1, 1, 2, 1, 3],
            'sale_date': ['2023-10-01', '2023-10-02', '2023-10-01', '2023-10-03', '2023-10-07'],
            'customer_id': ['C001', 'C004', 'C002', 'C010', 'C012']
        })
    
    def test_calculate_reorder_point(self):
        """再発注点計算のテスト"""
        # 正常ケース
        avg_demand = 2.5
        lead_time = 7
        safety_stock = 10
        
        reorder_point = self.optimizer.calculate_reorder_point(
            avg_demand, lead_time, safety_stock
        )
        
        expected = avg_demand * lead_time + safety_stock
        self.assertEqual(reorder_point, expected)
    
    def test_calculate_safety_stock(self):
        """安全在庫計算のテスト"""
        # テスト用の需要データ
        demand_data = [10, 12, 8, 15, 11, 9, 13]
        service_level = 0.95
        
        safety_stock = self.optimizer.calculate_safety_stock(
            demand_data, service_level
        )
        
        # 安全在庫が正の値であることを確認
        self.assertGreater(safety_stock, 0)
        self.assertIsInstance(safety_stock, (int, float))
    
    def test_optimize_inventory_levels(self):
        """在庫レベル最適化のテスト"""
        result = self.optimizer.optimize_inventory_levels(
            self.test_inventory_data,
            self.test_sales_data
        )
        
        # 結果の構造確認
        self.assertIn('optimized_levels', result)
        self.assertIn('total_cost_reduction', result)
        self.assertIn('recommendations', result)
        
        # 最適化レベルが製品数と一致することを確認
        self.assertEqual(
            len(result['optimized_levels']),
            len(self.test_inventory_data)
        )
    
    def test_abc_analysis(self):
        """ABC分析のテスト"""
        result = self.optimizer.abc_analysis(self.test_sales_data)
        
        # 結果の構造確認
        self.assertIn('abc_classification', result)
        self.assertIn('category_summary', result)
        
        # すべての製品が分類されていることを確認
        classified_products = set(result['abc_classification']['product_id'])
        original_products = set(self.test_sales_data['product_id'])
        self.assertEqual(classified_products, original_products)
    
    def test_stockout_risk_analysis(self):
        """欠品リスク分析のテスト"""
        result = self.optimizer.stockout_risk_analysis(
            self.test_inventory_data,
            self.test_sales_data
        )
        
        # 結果の構造確認
        self.assertIn('risk_scores', result)
        self.assertIn('high_risk_products', result)
        self.assertIn('recommendations', result)
        
        # リスクスコアが0-1の範囲内であることを確認
        for score in result['risk_scores']:
            self.assertGreaterEqual(score['risk_score'], 0)
            self.assertLessEqual(score['risk_score'], 1)
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        # 空のデータでの処理
        empty_df = pd.DataFrame()
        
        with self.assertRaises(Exception):
            self.optimizer.optimize_inventory_levels(empty_df, empty_df)
        
        # 無効な値での安全在庫計算
        with self.assertRaises(ValueError):
            self.optimizer.calculate_safety_stock([], 0.95)
    
    def test_performance(self):
        """パフォーマンステスト"""
        import time
        
        # 大きなデータセットでのテスト
        large_inventory = pd.concat([self.test_inventory_data] * 100, ignore_index=True)
        large_sales = pd.concat([self.test_sales_data] * 100, ignore_index=True)
        
        start_time = time.time()
        result = self.optimizer.optimize_inventory_levels(large_inventory, large_sales)
        end_time = time.time()
        
        # 10秒以内に処理が完了することを確認
        self.assertLess(end_time - start_time, 10)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
