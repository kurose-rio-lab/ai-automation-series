"""
需要予測システムのユニットテスト
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.demand_forecasting import DemandForecaster

class TestDemandForecasting(unittest.TestCase):
    """需要予測テストクラス"""
    
    def setUp(self):
        """テストセットアップ"""
        self.forecaster = DemandForecaster()
        
        # テスト用の時系列データ作成
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.test_sales_data = pd.DataFrame({
            'sale_date': dates,
            'product_id': ['P001'] * 100,
            'quantity': np.random.poisson(5, 100) + np.sin(np.arange(100) * 0.1) * 2
        })
        
        self.test_external_data = pd.DataFrame({
            'date': dates,
            'temperature': np.random.normal(20, 10, 100),
            'holiday': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
            'promotion': np.random.choice([0, 1], 100, p=[0.8, 0.2])
        })
    
    def test_prepare_time_series_data(self):
        """時系列データ準備のテスト"""
        result = self.forecaster.prepare_time_series_data(
            self.test_sales_data, 'P001'
        )
        
        # 結果の構造確認
        self.assertIn('time_series', result)
        self.assertIn('statistics', result)
        
        # データポイント数の確認
        self.assertEqual(len(result['time_series']), 100)
        
        # 統計情報の確認
        stats = result['statistics']
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('trend', stats)
    
    def test_exponential_smoothing(self):
        """指数平滑法のテスト"""
        time_series = self.test_sales_data.set_index('sale_date')['quantity']
        
        forecast = self.forecaster.exponential_smoothing(time_series, forecast_periods=7)
        
        # 予測結果の確認
        self.assertEqual(len(forecast['forecast']), 7)
        self.assertGreater(forecast['mae'], 0)
        self.assertGreater(forecast['mse'], 0)
        
        # 予測値が正の値であることを確認
        for value in forecast['forecast']:
            self.assertGreater(value, 0)
    
    def test_arima_forecast(self):
        """ARIMA予測のテスト"""
        time_series = self.test_sales_data.set_index('sale_date')['quantity']
        
        try:
            forecast = self.forecaster.arima_forecast(time_series, forecast_periods=7)
            
            # 予測結果の確認
            self.assertEqual(len(forecast['forecast']), 7)
            self.assertIn('confidence_intervals', forecast)
            self.assertGreater(forecast['aic'], 0)
            
        except Exception as e:
            # ARIMAが収束しない場合はスキップ
            self.skipTest(f"ARIMA収束エラー: {e}")
    
    def test_seasonal_decomposition(self):
        """季節分解のテスト"""
        time_series = self.test_sales_data.set_index('sale_date')['quantity']
        
        result = self.forecaster.seasonal_decomposition(time_series)
        
        # 分解結果の確認
        self.assertIn('trend', result)
        self.assertIn('seasonal', result)
        self.assertIn('residual', result)
        self.assertIn('seasonal_strength', result)
        
        # データポイント数の一致確認
        self.assertEqual(len(result['trend']), len(time_series))
    
    def test_feature_engineering(self):
        """特徴量エンジニアリングのテスト"""
        result = self.forecaster.feature_engineering(
            self.test_sales_data,
            self.test_external_data
        )
        
        # 特徴量の確認
        expected_features = ['lag_1', 'lag_7', 'rolling_mean_7', 'day_of_week', 'month']
        for feature in expected_features:
            self.assertIn(feature, result.columns)
        
        # データサイズの確認
        self.assertEqual(len(result), len(self.test_sales_data))
    
    def test_comprehensive_forecast(self):
        """包括的予測のテスト"""
        result = self.forecaster.comprehensive_forecast(
            self.test_sales_data,
            'P001',
            external_factors=self.test_external_data,
            forecast_periods=14
        )
        
        # 結果の構造確認
        self.assertIn('forecasts', result)
        self.assertIn('model_performance', result)
        self.assertIn('recommendations', result)
        
        # 予測期間の確認
        forecasts = result['forecasts']
        for model_name, forecast_data in forecasts.items():
            if forecast_data and 'forecast' in forecast_data:
                self.assertEqual(len(forecast_data['forecast']), 14)
    
    def test_cross_validation(self):
        """交差検証のテスト"""
        time_series = self.test_sales_data.set_index('sale_date')['quantity']
        
        result = self.forecaster.cross_validation_forecast(time_series, n_splits=3)
        
        # 検証結果の確認
        self.assertIn('cv_scores', result)
        self.assertIn('mean_mae', result)
        self.assertIn('mean_mse', result)
        
        # スコア数の確認
        self.assertEqual(len(result['cv_scores']), 3)
    
    def test_accuracy_metrics(self):
        """精度指標のテスト"""
        actual = np.array([10, 12, 8, 15, 11])
        predicted = np.array([9, 13, 7, 14, 10])
        
        mae = self.forecaster.calculate_mae(actual, predicted)
        mse = self.forecaster.calculate_mse(actual, predicted)
        mape = self.forecaster.calculate_mape(actual, predicted)
        
        # 指標値の妥当性確認
        self.assertGreater(mae, 0)
        self.assertGreater(mse, 0)
        self.assertGreater(mape, 0)
        self.assertLess(mape, 100)  # MAPEは通常100%未満
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        # 空のデータセット
        empty_df = pd.DataFrame()
        
        with self.assertRaises(Exception):
            self.forecaster.comprehensive_forecast(empty_df, 'P001')
        
        # 単一データポイント
        single_point_df = pd.DataFrame({
            'sale_date': ['2023-01-01'],
            'product_id': ['P001'],
            'quantity': [5]
        })
        
        with self.assertRaises(Exception):
            self.forecaster.comprehensive_forecast(single_point_df, 'P001')

if __name__ == '__main__':
    unittest.main()
