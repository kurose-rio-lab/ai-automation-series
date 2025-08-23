"""
日次監視システム
AI小売システム全体のパフォーマンス監視とアラート
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yaml
import smtplib
import requests
import json
import logging
from pathlib import Path
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.image import MimeImage
import warnings
warnings.filterwarnings('ignore')

class DailyMonitoringSystem:
    def __init__(self, config_path="config/config.yaml"):
        """日次監視システムの初期化"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # ログ設定
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['monitoring']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'daily_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # アラート閾値設定
        self.alert_thresholds = {
            'forecast_accuracy_min': 0.8,  # 予測精度最低値
            'inventory_turnover_min': 2.0,  # 在庫回転率最低値
            'system_response_time_max': 5.0,  # システム応答時間最大値（秒）
            'error_rate_max': 0.05,  # エラー率最大値
            'sales_decline_max': 0.2,  # 売上減少率最大値
            'stock_shortage_max': 0.1  # 欠品率最大値
        }
        
        # 監視結果
        self.monitoring_results = {}
        self.alerts = []
    
    def load_system_data(self):
        """システムデータの読み込み"""
        data_sources = {
            'sales': 'data/processed/sales_processed_*.csv',
            'inventory': 'data/processed/inventory_processed_*.csv',
            'forecasts': 'results/forecasting/*.csv',
            'recommendations': 'results/recommendation/*.csv',
            'system_logs': 'logs/*.log'
        }
        
        loaded_data = {}
        
        for data_type, pattern in data_sources.items():
            try:
                matching_files = list(Path(".").glob(pattern))
                
                if matching_files:
                    if data_type in ['sales', 'inventory']:
                        # 最新ファイルを読み込み
                        latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                        loaded_data[data_type] = pd.read_csv(latest_file)
                        self.logger.info(f"{data_type}データ読み込み: {latest_file}")
                    
                    elif data_type == 'forecasts':
                        # 予測結果ファイル
                        forecast_files = [f for f in matching_files if 'forecast' in f.name]
                        if forecast_files:
                            latest_forecast = max(forecast_files, key=lambda x: x.stat().st_mtime)
                            loaded_data[data_type] = pd.read_csv(latest_forecast)
                    
                    elif data_type == 'recommendations':
                        # 推薦結果ファイル
                        rec_files = [f for f in matching_files if 'recommendation' in f.name]
                        if rec_files:
                            latest_rec = max(rec_files, key=lambda x: x.stat().st_mtime)
                            loaded_data[data_type] = pd.read_csv(latest_rec)
                
                else:
                    self.logger.warning(f"{data_type}のデータファイルが見つかりません")
            
            except Exception as e:
                self.logger.error(f"{data_type}データ読み込みエラー: {e}")
        
        return loaded_data
    
    def monitor_forecast_accuracy(self, sales_data, forecast_data):
        """予測精度の監視"""
        if sales_data is None or forecast_data is None:
            return None
        
        try:
            # 最新7日間の実績vs予測比較
            latest_date = pd.to_datetime(sales_data['date']).max()
            week_ago = latest_date - timedelta(days=7)
            
            recent_sales = sales_data[pd.to_datetime(sales_data['date']) > week_ago]
            
            if 'forecast' in forecast_data.columns:
                # 商品別精度計算
                accuracy_by_product = {}
                
                for product_id in recent_sales['product_id'].unique():
                    actual = recent_sales[recent_sales['product_id'] == product_id]['quantity'].sum()
                    
                    if product_id in forecast_data['product_id'].values:
                        forecast = forecast_data[forecast_data['product_id'] == product_id]['forecast'].iloc[0]
                        
                        if actual > 0:
                            accuracy = 1 - abs(actual - forecast) / actual
                            accuracy_by_product[product_id] = max(0, accuracy)
                
                if accuracy_by_product:
                    avg_accuracy = np.mean(list(accuracy_by_product.values()))
                    
                    result = {
                        'metric': 'forecast_accuracy',
                        'value': avg_accuracy,
                        'threshold': self.alert_thresholds['forecast_accuracy_min'],
                        'status': 'OK' if avg_accuracy >= self.alert_thresholds['forecast_accuracy_min'] else 'ALERT',
                        'details': {
                            'product_count': len(accuracy_by_product),
                            'best_accuracy': max(accuracy_by_product.values()),
                            'worst_accuracy': min(accuracy_by_product.values()),
                            'products_below_threshold': len([a for a in accuracy_by_product.values() if a < self.alert_thresholds['forecast_accuracy_min']])
                        }
                    }
                    
                    self.monitoring_results['forecast_accuracy'] = result
                    
                    if result['status'] == 'ALERT':
                        self.alerts.append({
                            'type': 'forecast_accuracy',
                            'message': f"予測精度低下: {avg_accuracy:.2%} (閾値: {self.alert_thresholds['forecast_accuracy_min']:.2%})",
                            'severity': 'HIGH',
                            'timestamp': datetime.now()
                        })
                    
                    return result
        
        except Exception as e:
            self.logger.error(f"予測精度監視エラー: {e}")
        
        return None
    
    def monitor_inventory_performance(self, inventory_data, sales_data):
        """在庫パフォーマンスの監視"""
        if inventory_data is None or sales_data is None:
            return None
        
        try:
            # 在庫回転率計算
            latest_date = pd.to_datetime(sales_data['date']).max()
            month_ago = latest_date - timedelta(days=30)
            
            monthly_sales = sales_data[pd.to_datetime(sales_data['date']) > month_ago]
            monthly_cogs = monthly_sales.groupby('product_id')['sales_amount'].sum() * 0.7  # 仮の原価率70%
            
            inventory_turnover = {}
            stockout_products = []
            
            for _, row in inventory_data.iterrows():
                product_id = row['product_id']
                current_stock = row.get('current_stock', 0)
                
                if product_id in monthly_cogs.index:
                    cogs = monthly_cogs[product_id]
                    avg_inventory_value = current_stock * row.get('unit_cost', 1000)
                    
                    if avg_inventory_value > 0:
                        turnover = (cogs * 12) / avg_inventory_value  # 年間回転率
                        inventory_turnover[product_id] = turnover
