"""
自動発注システム
在庫レベルを監視し、自動的に発注を実行するシステム
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
import requests
import logging
from typing import Dict, List, Optional
import time

class AutoOrderingSystem:
    def __init__(self, config_path="config/config.yaml"):
        """自動発注システムの初期化"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # ログ設定
        logging.basicConfig(
            level=getattr(logging, self.config['monitoring']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/auto_ordering.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 発注パラメータ
        self.review_period = self.config['model_settings']['inventory_optimization']['review_period']
        self.email_notifications = self.config['alerts']['email_notifications']
        
        # 発注履歴
        self.order_history = []
        
        # 在庫データファイルパス
        self.inventory_file = "data/current_inventory.csv"
        self.safety_stock_file = "results/safety_stock_recommendations.csv"
        self.order_log_file = "logs/order_history.csv"
    
    def load_current_inventory(self) -> Optional[pd.DataFrame]:
        """現在の在庫データを読み込み"""
        try:
            if Path(self.inventory_file).exists():
                df = pd.read_csv(self.inventory_file)
                self.logger.info(f"在庫データ読み込み完了: {len(df)}商品")
                return df
            else:
                self.logger.warning(f"在庫ファイルが見つかりません: {self.inventory_file}")
                return None
        except Exception as e:
            self.logger.error(f"在庫データ読み込みエラー: {e}")
            return None
    
    def load_safety_stock_parameters(self) -> Optional[pd.DataFrame]:
        """安全在庫パラメータを読み込み"""
        try:
            if Path(self.safety_stock_file).exists():
                df = pd.read_csv(self.safety_stock_file)
                self.logger.info(f"安全在庫パラメータ読み込み完了: {len(df)}商品")
                return df
            else:
                self.logger.warning(f"安全在庫ファイルが見つかりません: {self.safety_stock_file}")
                return None
        except Exception as e:
            self.logger.error(f"安全在庫パラメータ読み込みエラー: {e}")
            return None
    
    def check_reorder_points(self, inventory_df: pd.DataFrame, 
                           safety_stock_df: pd.DataFrame) -> pd.DataFrame:
        """リオーダーポイントのチェック"""
        self.logger.info("リオーダーポイントをチェック中...")
        
        # データをマージ
        merged_df = pd.merge(inventory_df, safety_stock_df, on='product_id', how='inner')
        
        # 発注が必要な商品を特定
        merged_df['order_needed'] = merged_df['current_stock'] <= merged_df['reorder_point']
        merged_df['days_until_stockout'] = np.where(
            merged_df['avg_demand'] > 0,
            merged_df['current_stock'] / merged_df['avg_demand'],
            999
        )
        
        # 発注量の計算
        merged_df['recommended_order_qty'] = np.where(
            merged_df['order_needed'],
            merged_df['economic_order_quantity'],
            0
        )
        
        # 緊急度の設定
        merged_df['urgency'] = 'normal'
        merged_df.loc[merged_df['days_until_stockout'] <= 3, 'urgency'] = 'high'
        merged_df.loc[merged_df['days_until_stockout'] <= 1, 'urgency'] = 'critical'
        
        # 発注が必要な商品のみ抽出
        orders_needed = merged_df[merged_df['order_needed']].copy()
        
        self.logger.info(f"発注が必要な商品数: {len(orders_needed)}商品")
        
        return orders_needed
    
    def calculate_order_priorities(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """発注優先度の計算"""
        if len(orders_df) == 0:
            return orders_df
        
        # 優先度スコアの計算
        # 要素: 緊急度、売上貢献度、在庫回転率
        urgency_weights = {'critical': 10, 'high': 5, 'normal': 1}
        orders_df['urgency_score'] = orders_df['urgency'].map(urgency_weights)
        
        # 売上貢献度（ABCクラス）
        abc_weights = {'A': 5, 'B': 3, 'C': 1}
        # ABCクラスがない場合は平均需要から推定
        if 'abc_class' not in orders_df.columns:
            orders_df['abc_class'] = pd.cut(
                orders_df['avg_demand'], 
                bins=3, 
                labels=['C', 'B', 'A']
            )
        
        orders_df['abc_score'] = orders_df['abc_class'].map(abc_weights)
        
        # 総合優先度スコア
        orders_df['priority_score'] = (
            orders_df['urgency_score'] * 0.5 + 
            orders_df['abc_score'] * 0.3 + 
            (10 - orders_df['days_until_stockout']).clip(0, 10) * 0.2
        )
        
        # 優先度順でソート
        orders_df = orders_df.sort_values('priority_score', ascending=False)
        
        return orders_df
    
    def generate_purchase_orders(self, orders_df: pd.DataFrame) -> List[Dict]:
        """購買発注書の生成"""
        if len(orders_df) == 0:
            return []
        
        purchase_orders = []
        
        # サプライヤー別にグループ化
        for supplier_id, supplier_orders in orders_df.groupby('supplier_id'):
            
            order_id = f"PO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{supplier_id}"
            
            order_items = []
            total_amount = 0
            
            for _, item in supplier_orders.iterrows():
                item_total = item['recommended_order_qty'] * item.get('unit_cost', 1000)
                
                order_items.append({
                    'product_id': item['product_id'],
                    'product_name': item.get('product_name', item['product_id']),
                    'quantity': int(item['recommended_order_qty']),
                    'unit_cost': item.get('unit_cost', 1000),
                    'total_cost': item_total,
                    'urgency': item['urgency'],
                    'current_stock': int(item['current_stock']),
                    'reorder_point': int(item['reorder_point'])
                })
                
                total_amount += item_total
            
            purchase_order = {
                'order_id': order_id,
                'supplier_id': supplier_id,
                'order_date': datetime.now().isoformat(),
                'expected_delivery': (datetime.now() + timedelta(days=7)).isoformat(),
                'items': order_items,
                'total_amount': total_amount,
                'status': 'pending',
                'urgency_level': supplier_orders['urgency'].mode().iloc[0] if len(supplier_orders) > 0 else 'normal'
            }
            
            purchase_orders.append(purchase_order)
            self.order_history.append(purchase_order)
        
        self.logger.info(f"発注書生成完了: {len(purchase_orders)}件")
        
        return purchase_orders
    
    def save_purchase_orders(self, purchase_orders: List[Dict]) -> List[str]:
        """発注書の保存"""
        if not purchase_orders:
            return []
        
        output_dir = Path("orders")
        output_dir.mkdir(exist_ok=True)
        
        saved_files = []
        
        for order in purchase_orders:
            # JSON形式で保存
            order_file = output_dir / f"{order['order_id']}.json"
            with open(order_file, 'w', encoding='utf-8') as f:
                json.dump(order, f, ensure_ascii=False, indent=2)
            
            # CSV形式でも保存（Excel で開きやすく）
            items_df = pd.DataFrame(order['items'])
            csv_file = output_dir / f"{order['order_id']}_items.csv"
            items_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            saved_files.extend([str(order_file), str(csv_file)])
            
            self.logger.info(f"発注書保存: {order['order_id']}")
        
        return saved_files
    
    def send_order_notifications(self, purchase_orders: List[Dict]):
        """発注通知の送信"""
        if not purchase_orders or not self.email_notifications:
            return
        
        try:
            # 緊急発注の確認
            critical_orders = [o for o in purchase_orders if o['urgency_level'] == 'critical']
            high_orders = [o for o in purchase_orders if o['urgency_level'] == 'high']
            
            # メール本文作成
            subject = f"自動発注通知 - {len(purchase_orders)}件の発注が実行されました"
            
            body = f"""
自動発注システムから通知です。

■ 発注サマリー
- 総発注件数: {len(purchase_orders)}件
- 緊急発注: {len(critical_orders)}件
- 高優先度発注: {len(high_orders)}件
- 発注日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

■ 発注詳細
"""
            
            for order in purchase_orders:
                body += f"""
【発注ID】: {order['order_id']}
【サプライヤー】: {order['supplier_id']}
【商品数】: {len(order['items'])}点
【総額】: {order['total_amount']:,}円
【緊急度】: {order['urgency_level']}
【予定納期】: {order['expected_delivery'][:10]}
"""
            
            body += f"""

■ 次回チェック予定
{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')}

※ このメールは自動送信されています。
※ 発注詳細は orders/ フォルダをご確認ください。
"""
            
            # Slack通知（設定されている場合）
            if self.config['alerts'].get('slack_webhook'):
                self.send_slack_notification(purchase_orders)
            
            self.logger.info("発注通知送信完了")
            
        except Exception as e:
            self.logger.error(f"通知送信エラー: {e}")
    
    def send_slack_notification(self, purchase_orders: List[Dict]):
        """Slack通知の送信"""
        try:
            webhook_url = self.config['alerts']['slack_webhook']
            
            # Slack メッセージ作成
            text = f"🛒 自動発注システム通知\n"
            text += f"発注件数: {len(purchase_orders)}件\n"
            
            for order in purchase_orders:
                urgency_emoji = {'critical': '🚨', 'high': '⚠️', 'normal': '📋'}
                emoji = urgency_emoji.get(order['urgency_level'], '📋')
                text += f"{emoji} {order['order_id']}: {order['supplier_id']} ({len(order['items'])}点)\n"
            
            payload = {
                'text': text,
                'username': '在庫管理Bot',
                'icon_emoji': ':robot_face:'
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            self.logger.info("Slack通知送信完了")
            
        except Exception as e:
            self.logger.error(f"Slack通知送信エラー: {e}")
    
    def update_inventory_status(self, orders_df: pd.DataFrame):
        """在庫ステータスの更新"""
        try:
            # 発注済みフラグの追加
            inventory_df = self.load_current_inventory()
            if inventory_df is not None:
                # 発注された商品にフラグを設定
                ordered_products = orders_df['product_id'].tolist()
                inventory_df['order_pending'] = inventory_df['product_id'].isin(ordered_products)
                inventory_df['last_order_date'] = datetime.now().strftime('%Y-%m-%d')
                
                # 更新保存
                inventory_df.to_csv(self.inventory_file, index=False, encoding='utf-8-sig')
                self.logger.info("在庫ステータス更新完了")
                
        except Exception as e:
            self.logger.error(f"在庫ステータス更新エラー: {e}")
    
    def save_order_log(self, purchase_orders: List[Dict]):
        """発注履歴の保存"""
        if not purchase_orders:
            return
        
        try:
            # 発注ログデータの準備
            log_data = []
            for order in purchase_orders:
                for item in order['items']:
                    log_data.append({
                        'order_date': order['order_date'][:10],
                        'order_id': order['order_id'],
                        'supplier_id': order['supplier_id'],
                        'product_id': item['product_id'],
                        'quantity': item['quantity'],
                        'unit_cost': item['unit_cost'],
                        'total_cost': item['total_cost'],
                        'urgency': item['urgency'],
                        'status': order['status']
                    })
            
            log_df = pd.DataFrame(log_data)
            
            # 既存ログに追記
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            if Path(self.order_log_file).exists():
                existing_log = pd.read_csv(self.order_log_file)
                combined_log = pd.concat([existing_log, log_df], ignore_index=True)
            else:
                combined_log = log_df
            
            combined_log.to_csv(self.order_log_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"発注履歴保存完了: {len(log_data)}件")
            
        except Exception as e:
            self.logger.error(f"発注履歴保存エラー: {e}")
    
    def run_ordering_cycle(self):
        """発注サイクルの実行"""
        self.logger.info("=== 自動発注サイクル開始 ===")
        
        try:
            # 1. データ読み込み
            inventory_df = self.load_current_inventory()
            safety_stock_df = self.load_safety_stock_parameters()
            
            if inventory_df is None or safety_stock_df is None:
                self.logger.error("必要なデータファイルが見つかりません")
                return
            
            # 2. リオーダーポイントチェック
            orders_needed = self.check_reorder_points(inventory_df, safety_stock_df)
            
            if len(orders_needed) == 0:
                self.logger.info("発注が必要な商品はありません")
                return
            
            # 3. 発注優先度計算
            orders_prioritized = self.calculate_order_priorities(orders_needed)
            
            # 4. 発注書生成
            purchase_orders = self.generate_purchase_orders(orders_prioritized)
            
            # 5. 発注書保存
            saved_files = self.save_purchase_orders(purchase_orders)
            
            # 6. 通知送信
            self.send_order_notifications(purchase_orders)
            
            # 7. 在庫ステータス更新
            self.update_inventory_status(orders_needed)
            
            # 8. 発注履歴保存
            self.save_order_log(purchase_orders)
            
            self.logger.info(f"=== 自動発注サイクル完了: {len(purchase_orders)}件の発注実行 ===")
            
            return purchase_orders
            
        except Exception as e:
            self.logger.error(f"発注サイクル実行エラー: {e}")
            raise
    
    def generate_ordering_report(self):
        """発注レポートの生成"""
        try:
            if not Path(self.order_log_file).exists():
                print("発注履歴がありません")
                return
            
            log_df = pd.read_csv(self.order_log_file)
            
            print("\n" + "="*50)
            print("自動発注システム レポート")
            print("="*50)
            
            # 期間別発注統計
            log_df['order_date'] = pd.to_datetime(log_df['order_date'])
            recent_orders = log_df[log_df['order_date'] >= datetime.now() - timedelta(days=30)]
            
            print(f"\n■ 直近30日間の発注実績")
            print(f"発注回数: {len(recent_orders)}回")
            print(f"発注総額: {recent_orders['total_cost'].sum():,.0f}円")
            print(f"平均発注額: {recent_orders['total_cost'].mean():,.0f}円")
            
            # サプライヤー別統計
            supplier_stats = recent_orders.groupby('supplier_id').agg({
                'order_id': 'nunique',
                'total_cost': 'sum'
            }).round(0)
            
            print(f"\n■ サプライヤー別発注実績")
            print(supplier_stats)
            
            # 緊急度別統計
            urgency_stats = recent_orders['urgency'].value_counts()
            print(f"\n■ 緊急度別発注実績")
            print(urgency_stats)
            
        except Exception as e:
            self.logger.error(f"レポート生成エラー: {e}")

def create_sample_inventory_data():
    """サンプル在庫データの生成"""
    np.random.seed(42)
    
    products = [f"PROD_{i:04d}" for i in range(1, 101)]
    
    inventory_data = []
    for product_id in products:
        # 現在の在庫レベル（リオーダーポイント付近になるよう調整）
        reorder_point = np.random.uniform(20, 100)
        current_stock = np.random.uniform(0, reorder_point * 1.5)  # 一部は発注必要レベル
        
        inventory_data.append({
            'product_id': product_id,
            'product_name': f'商品{product_id}',
            'current_stock': int(current_stock),
            'supplier_id': f"SUP_{np.random.randint(1, 21):02d}",
            'unit_cost': np.random.uniform(500, 5000),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'order_pending': False
        })
    
    df = pd.DataFrame(inventory_data)
    
    # 保存
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    df.to_csv("data/current_inventory.csv", index=False, encoding='utf-8-sig')
    print(f"サンプル在庫データを生成: data/current_inventory.csv")
    
    return df

def main():
    """メイン実行関数"""
    print("自動発注システムを開始します...")
    
    # サンプルデータ生成
    if not Path("data/current_inventory.csv").exists():
        print("サンプル在庫データを生成中...")
        create_sample_inventory_data()
    
    # ログディレクトリ作成
    Path("logs").mkdir(exist_ok=True)
    
    # 自動発注システムの初期化
    ordering_system = AutoOrderingSystem()
    
    # 発注サイクル実行
    try:
        purchase_orders = ordering_system.run_ordering_cycle()
        
        if purchase_orders:
            print(f"\n✅ {len(purchase_orders)}件の発注を実行しました")
            print("📁 orders/ フォルダに発注書が保存されています")
        else:
            print("\n📋 現在発注が必要な商品はありません")
        
        # レポート表示
        ordering_system.generate_ordering_report()
        
    except Exception as e:
        print(f"❌ 発注システム実行エラー: {e}")
        return
    
    print("\n🎯 自動発注システムが正常に完了しました")

def run_continuous_monitoring():
    """継続的な監視モード"""
    ordering_system = AutoOrderingSystem()
    
    print("継続監視モードを開始します（Ctrl+C で停止）")
    
    try:
        while True:
            print(f"\n{datetime.now()}: 在庫チェック実行中...")
            
            try:
                ordering_system.run_ordering_cycle()
            except Exception as e:
                print(f"エラー発生: {e}")
            
            # 1日間隔で監視
            print("次回チェックまで24時間待機中...")
            time.sleep(24 * 60 * 60)  # 24時間待機
            
    except KeyboardInterrupt:
        print("\n監視モードを停止します")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        run_continuous_monitoring()
    else:
        main()
