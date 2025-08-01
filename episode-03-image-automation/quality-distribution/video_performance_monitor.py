# video_performance_monitor.py
# リアルタイム監視・分析・レポート生成による動画制作システムの完全可視化

import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import os
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    video_id: str
    platform: str
    generation_time: float
    processing_time: float
    quality_score: float
    file_size: int
    success: bool
    error_message: Optional[str]
    cost: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class QualityMetrics:
    video_id: str
    technical_score: float
    visual_score: float
    audio_score: float
    content_score: float
    overall_score: float
    recommendations: List[str]
    created_at: datetime

@dataclass
class SystemHealth:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_processes: int
    queue_length: int
    error_rate: float
    avg_response_time: float

class DatabaseManager:
    """データベース管理クラス"""
    
    def __init__(self, db_path: str = "video_performance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # パフォーマンステーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                generation_time REAL,
                processing_time REAL,
                quality_score REAL,
                file_size INTEGER,
                success BOOLEAN,
                error_message TEXT,
                cost REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 品質メトリクステーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                technical_score REAL,
                visual_score REAL,
                audio_score REAL,
                content_score REAL,
                overall_score REAL,
                recommendations TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # システムヘルステーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                active_processes INTEGER,
                queue_length INTEGER,
                error_rate REAL,
                avg_response_time REAL
            )
        ''')
        
        # コストトラッキングテーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cost_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                platform TEXT,
                api_provider TEXT,
                operation_type TEXT,
                cost REAL,
                usage_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # アラートテーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved_at DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """データベース接続取得"""
        return sqlite3.connect(self.db_path)

class MetricsCollector:
    """メトリクス収集クラス"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def log_video_performance(self, metrics: PerformanceMetrics):
        """動画パフォーマンス記録"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO video_performance 
            (video_id, platform, generation_time, processing_time, quality_score,
             file_size, success, error_message, cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.video_id, metrics.platform, metrics.generation_time,
            metrics.processing_time, metrics.quality_score, metrics.file_size,
            metrics.success, metrics.error_message, metrics.cost
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Performance metrics logged for video {metrics.video_id}")
    
    def log_quality_metrics(self, metrics: QualityMetrics):
        """品質メトリクス記録"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_metrics 
            (video_id, technical_score, visual_score, audio_score, content_score,
             overall_score, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.video_id, metrics.technical_score, metrics.visual_score,
            metrics.audio_score, metrics.content_score, metrics.overall_score,
            json.dumps(metrics.recommendations)
        ))
        
        conn.commit()
        conn.close()
    
    def log_system_health(self, health: SystemHealth):
        """システムヘルス記録"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_health 
            (cpu_usage, memory_usage, disk_usage, active_processes,
             queue_length, error_rate, avg_response_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            health.cpu_usage, health.memory_usage, health.disk_usage,
            health.active_processes, health.queue_length, 
            health.error_rate, health.avg_response_time
        ))
        
        conn.commit()
        conn.close()
    
    def log_cost_data(self, date: str, platform: str, api_provider: str,
                     operation_type: str, cost: float, usage_count: int):
        """コストデータ記録"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO cost_tracking 
            (date, platform, api_provider, operation_type, cost, usage_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date, platform, api_provider, operation_type, cost, usage_count))
        
        conn.commit()
        conn.close()

class AlertManager:
    """アラート管理クラス"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # アラート閾値設定
        self.thresholds = {
            'error_rate': 0.05,  # 5%
            'avg_response_time': 300,  # 5分
            'quality_score': 0.7,
            'cost_daily_limit': 100.0,  # $100
            'processing_time': 600,  # 10分
            'queue_length': 50
        }
    
    def check_and_create_alerts(self):
        """アラートチェックと作成"""
        # エラー率チェック
        self._check_error_rate()
        
        # 応答時間チェック
        self._check_response_time()
        
        # 品質スコアチェック
        self._check_quality_scores()
        
        # コストチェック
        self._check_daily_costs()
        
        # キュー長チェック
        self._check_queue_length()
    
    def _check_error_rate(self):
        """エラー率チェック"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        # 過去1時間のエラー率
        cursor.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as errors
            FROM video_performance 
            WHERE created_at >= datetime('now', '-1 hour')
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] > 0:
            error_rate = result[1] / result[0]
            if error_rate > self.thresholds['error_rate']:
                self._create_alert(
                    'error_rate_high',
                    'warning',
                    f'High error rate detected: {error_rate:.1%}',
                    {'error_rate': error_rate, 'threshold': self.thresholds['error_rate']}
                )
    
    def _check_response_time(self):
        """応答時間チェック"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT AVG(processing_time) as avg_time
            FROM video_performance 
            WHERE created_at >= datetime('now', '-1 hour')
            AND success = 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            avg_time = result[0]
            if avg_time > self.thresholds['avg_response_time']:
                self._create_alert(
                    'slow_response_time',
                    'warning',
                    f'Slow average response time: {avg_time:.1f}s',
                    {'avg_time': avg_time, 'threshold': self.thresholds['avg_response_time']}
                )
    
    def _check_quality_scores(self):
        """品質スコアチェック"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT AVG(overall_score) as avg_quality
            FROM quality_metrics 
            WHERE created_at >= datetime('now', '-1 hour')
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            avg_quality = result[0]
            if avg_quality < self.thresholds['quality_score']:
                self._create_alert(
                    'low_quality_score',
                    'warning',
                    f'Low average quality score: {avg_quality:.2f}',
                    {'avg_quality': avg_quality, 'threshold': self.thresholds['quality_score']}
                )
    
    def _check_daily_costs(self):
        """日次コストチェック"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        cursor.execute('''
            SELECT SUM(cost) as total_cost
            FROM cost_tracking 
            WHERE date = ?
        ''', (today,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            total_cost = result[0]
            if total_cost > self.thresholds['cost_daily_limit']:
                self._create_alert(
                    'daily_cost_exceeded',
                    'critical',
                    f'Daily cost limit exceeded: ${total_cost:.2f}',
                    {'total_cost': total_cost, 'limit': self.thresholds['cost_daily_limit']}
                )
    
    def _check_queue_length(self):
        """キュー長チェック"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT queue_length
            FROM system_health 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            queue_length = result[0]
            if queue_length > self.thresholds['queue_length']:
                self._create_alert(
                    'high_queue_length',
                    'warning',
                    f'High queue length: {queue_length} items',
                    {'queue_length': queue_length, 'threshold': self.thresholds['queue_length']}
                )
    
    def _create_alert(self, alert_type: str, severity: str, message: str, details: Dict[str, Any]):
        """アラート作成"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        # 重複チェック（同じタイプのアラートが過去1時間以内に未解決で存在するか）
        cursor.execute('''
            SELECT COUNT(*) FROM alerts 
            WHERE alert_type = ? AND resolved = 0 
            AND created_at >= datetime('now', '-1 hour')
        ''', (alert_type,))
        
        if cursor.fetchone()[0] == 0:  # 重複なし
            cursor.execute('''
                INSERT INTO alerts (alert_type, severity, message, details)
                VALUES (?, ?, ?, ?)
            ''', (alert_type, severity, message, json.dumps(details)))
            
            conn.commit()
            self.logger.warning(f"Alert created: {alert_type} - {message}")
        
        conn.close()
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """アクティブアラート取得"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT alert_type, severity, message, details, created_at
            FROM alerts 
            WHERE resolved = 0
            ORDER BY created_at DESC
        ''')
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                'alert_type': row[0],
                'severity': row[1],
                'message': row[2],
                'details': json.loads(row[3]) if row[3] else {},
                'created_at': row[4]
            })
        
        conn.close()
        return alerts
    
    def resolve_alert(self, alert_type: str):
        """アラート解決"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE alerts 
            SET resolved = 1, resolved_at = datetime('now')
            WHERE alert_type = ? AND resolved = 0
        ''', (alert_type,))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Alert resolved: {alert_type}")

class ReportGenerator:
    """レポート生成クラス"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.output_dir = "reports"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_daily_report(self, date: str = None) -> Dict[str, Any]:
        """日次レポート生成"""
        if date is None:
            date = datetime.now().date().isoformat()
        
        conn = self.db_manager.get_connection()
        
        # 基本統計
        performance_stats = pd.read_sql_query('''
            SELECT 
                platform,
                COUNT(*) as total_videos,
                AVG(generation_time) as avg_generation_time,
                AVG(processing_time) as avg_processing_time,
                AVG(quality_score) as avg_quality_score,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_videos,
                SUM(cost) as total_cost
            FROM video_performance 
            WHERE DATE(created_at) = ?
            GROUP BY platform
        ''', conn, params=[date])
        
        # エラー分析
        error_analysis = pd.read_sql_query('''
            SELECT error_message, COUNT(*) as count
            FROM video_performance 
            WHERE DATE(created_at) = ? AND success = 0
            GROUP BY error_message
            ORDER BY count DESC
        ''', conn, params=[date])
        
        # 品質トレンド
        quality_trends = pd.read_sql_query('''
            SELECT 
                strftime('%H', created_at) as hour,
                AVG(overall_score) as avg_quality
            FROM quality_metrics 
            WHERE DATE(created_at) = ?
            GROUP BY strftime('%H', created_at)
            ORDER BY hour
        ''', conn, params=[date])
        
        # コスト分析
        cost_breakdown = pd.read_sql_query('''
            SELECT 
                api_provider,
                operation_type,
                SUM(cost) as total_cost,
                SUM(usage_count) as total_usage
            FROM cost_tracking 
            WHERE date = ?
            GROUP BY api_provider, operation_type
        ''', conn, params=[date])
        
        conn.close()
        
        # 成功率計算
        if not performance_stats.empty:
            performance_stats['success_rate'] = (
                performance_stats['successful_videos'] / performance_stats['total_videos'] * 100
            )
        
        report = {
            'date': date,
            'summary': {
                'total_videos': performance_stats['total_videos'].sum() if not performance_stats.empty else 0,
                'overall_success_rate': (
                    performance_stats['successful_videos'].sum() / performance_stats['total_videos'].sum() * 100
                    if not performance_stats.empty and performance_stats['total_videos'].sum() > 0 else 0
                ),
                'total_cost': performance_stats['total_cost'].sum() if not performance_stats.empty else 0,
                'avg_quality_score': performance_stats['avg_quality_score'].mean() if not performance_stats.empty else 0
            },
            'by_platform': performance_stats.to_dict('records') if not performance_stats.empty else [],
            'error_analysis': error_analysis.to_dict('records') if not error_analysis.empty else [],
            'quality_trends': quality_trends.to_dict('records') if not quality_trends.empty else [],
            'cost_breakdown': cost_breakdown.to_dict('records') if not cost_breakdown.empty else [],
            'recommendations': self._generate_daily_recommendations(performance_stats, error_analysis)
        }
        
        return report
    
    def generate_weekly_report(self) -> Dict[str, Any]:
        """週次レポート生成"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        
        conn = self.db_manager.get_connection()
        
        # 週次トレンド
        weekly_trends = pd.read_sql_query('''
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as total_videos,
                AVG(quality_score) as avg_quality,
                SUM(cost) as daily_cost,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
            FROM video_performance 
            WHERE DATE(created_at) BETWEEN ? AND ?
            GROUP BY DATE(created_at)
            ORDER BY date
        ''', conn, params=[start_date.isoformat(), end_date.isoformat()])
        
        # プラットフォーム比較
        platform_comparison = pd.read_sql_query('''
            SELECT 
                platform,
                COUNT(*) as total_videos,
                AVG(processing_time) as avg_processing_time,
                AVG(quality_score) as avg_quality,
                SUM(cost) as total_cost,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
            FROM video_performance 
            WHERE DATE(created_at) BETWEEN ? AND ?
            GROUP BY platform
            ORDER BY total_videos DESC
        ''', conn, params=[start_date.isoformat(), end_date.isoformat()])
        
        conn.close()
        
        return {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'weekly_trends': weekly_trends.to_dict('records') if not weekly_trends.empty else [],
            'platform_comparison': platform_comparison.to_dict('records') if not platform_comparison.empty else [],
            'insights': self._generate_weekly_insights(weekly_trends, platform_comparison)
        }
    
    def _generate_daily_recommendations(self, performance_stats: pd.DataFrame, 
                                      error_analysis: pd.DataFrame) -> List[str]:
        """日次推奨事項生成"""
        recommendations = []
        
        if not performance_stats.empty:
            # 成功率が低いプラットフォーム
            low_success_platforms = performance_stats[performance_stats['success_rate'] < 95]
            if not low_success_platforms.empty:
                platforms = ', '.join(low_success_platforms['platform'].tolist())
                recommendations.append(f"成功率が低いプラットフォーム ({platforms}) の調査が必要です")
            
            # 処理時間が長いプラットフォーム
            slow_platforms = performance_stats[performance_stats['avg_processing_time'] > 300]
            if not slow_platforms.empty:
                platforms = ', '.join(slow_platforms['platform'].tolist())
                recommendations.append(f"処理時間が長いプラットフォーム ({platforms}) の最適化を検討してください")
            
            # 品質スコアが低いプラットフォーム
            low_quality_platforms = performance_stats[performance_stats['avg_quality_score'] < 0.8]
            if not low_quality_platforms.empty:
                platforms = ', '.join(low_quality_platforms['platform'].tolist())
                recommendations.append(f"品質向上が必要なプラットフォーム: {platforms}")
        
        if not error_analysis.empty and len(error_analysis) > 0:
            most_common_error = error_analysis.iloc[0]['error_message']
            recommendations.append(f"最も多いエラー「{most_common_error}」の対策を優先してください")
        
        if not recommendations:
            recommendations.append("システムは正常に動作しています")
        
        return recommendations
    
    def _generate_weekly_insights(self, weekly_trends: pd.DataFrame, 
                                platform_comparison: pd.DataFrame) -> List[str]:
        """週次インサイト生成"""
        insights = []
        
        if not weekly_trends.empty:
            # トレンド分析
            if len(weekly_trends) >= 2:
                recent_quality = weekly_trends.tail(3)['avg_quality'].mean()
                earlier_quality = weekly_trends.head(4)['avg_quality'].mean()
                
                if recent_quality > earlier_quality * 1.05:
                    insights.append("品質スコアが向上傾向にあります")
                elif recent_quality < earlier_quality * 0.95:
                    insights.append("品質スコアが低下傾向にあります。調査が必要です")
        
        if not platform_comparison.empty:
            # 最も効率的なプラットフォーム
            best_platform = platform_comparison.loc[platform_comparison['success_rate'].idxmax()]
            insights.append(f"最も安定しているプラットフォーム: {best_platform['platform']} (成功率: {best_platform['success_rate']:.1f}%)")
            
            # コスト効率分析
            platform_comparison['cost_per_video'] = platform_comparison['total_cost'] / platform_comparison['total_videos']
            most_cost_effective = platform_comparison.loc[platform_comparison['cost_per_video'].idxmin()]
            insights.append(f"最もコスト効率が良いプラットフォーム: {most_cost_effective['platform']}")
        
        return insights
    
    def create_visual_dashboard(self, report_data: Dict[str, Any], output_path: str = None):
        """ビジュアルダッシュボード作成"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"dashboard_{datetime.now().strftime('%Y%m%d')}.html")
        
        # Plotlyを使用してインタラクティブダッシュボード作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['プラットフォーム別成功率', '品質スコア時系列', 'コスト分析', 'エラー分析'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # プラットフォーム別成功率
        if report_data['by_platform']:
            platforms = [p['platform'] for p in report_data['by_platform']]
            success_rates = [p['success_rate'] for p in report_data['by_platform']]
            
            fig.add_trace(
                go.Bar(x=platforms, y=success_rates, name="成功率"),
                row=1, col=1
            )
        
        # 品質トレンド
        if report_data['quality_trends']:
            hours = [int(q['hour']) for q in report_data['quality_trends']]
            quality_scores = [q['avg_quality'] for q in report_data['quality_trends']]
            
            fig.add_trace(
                go.Scatter(x=hours, y=quality_scores, mode='lines+markers', name="品質スコア"),
                row=1, col=2
            )
        
        # コスト分析
        if report_data['cost_breakdown']:
            providers = [c['api_provider'] for c in report_data['cost_breakdown']]
            costs = [c['total_cost'] for c in report_data['cost_breakdown']]
            
            fig.add_trace(
                go.Pie(labels=providers, values=costs, name="コスト分析"),
                row=2, col=1
            )
        
        # エラー分析
        if report_data['error_analysis']:
            errors = [e['error_message'][:30] + '...' if len(e['error_message']) > 30 else e['error_message'] 
                     for e in report_data['error_analysis'][:10]]  # 上位10個
            counts = [e['count'] for e in report_data['error_analysis'][:10]]
            
            fig.add_trace(
                go.Bar(x=counts, y=errors, orientation='h', name="エラー数"),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text=f"動画制作システム ダッシュボード - {report_data['date']}",
            showlegend=False
        )
        
        fig.write_html(output_path)
        return output_path

class VideoPerformanceMonitor:
    """統合動画パフォーマンス監視システム"""
    
    def __init__(self, db_path: str = "video_performance.db"):
        self.db_manager = DatabaseManager(db_path)
        self.metrics_collector = MetricsCollector(self.db_manager)
        self.alert_manager = AlertManager(self.db_manager)
        self.report_generator = ReportGenerator(self.db_manager)
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def log_video_generation(self, video_data: Dict[str, Any]):
        """動画生成ログ記録"""
        metrics = PerformanceMetrics(
            video_id=video_data['video_id'],
            platform=video_data.get('platform', 'unknown'),
            generation_time=video_data.get('generation_time', 0),
            processing_time=video_data.get('processing_time', 0),
            quality_score=video_data.get('quality_score', 0),
            file_size=video_data.get('file_size', 0),
            success=video_data.get('success', False),
            error_message=video_data.get('error_message'),
            cost=video_data.get('cost', 0),
            created_at=datetime.now()
        )
        
        self.metrics_collector.log_video_performance(metrics)
    
    def log_quality_analysis(self, video_id: str, quality_data: Dict[str, Any]):
        """品質分析ログ記録"""
        metrics = QualityMetrics(
            video_id=video_id,
            technical_score=quality_data.get('technical_score', 0),
            visual_score=quality_data.get('visual_score', 0),
            audio_score=quality_data.get('audio_score', 0),
            content_score=quality_data.get('content_score', 0),
            overall_score=quality_data.get('overall_score', 0),
            recommendations=quality_data.get('recommendations', []),
            created_at=datetime.now()
        )
        
        self.metrics_collector.log_quality_metrics(metrics)
    
    def log_system_status(self, cpu_usage: float, memory_usage: float, 
                         disk_usage: float, active_processes: int,
                         queue_length: int, error_rate: float, avg_response_time: float):
        """システム状態ログ記録"""
        health = SystemHealth(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            active_processes=active_processes,
            queue_length=queue_length,
            error_rate=error_rate,
            avg_response_time=avg_response_time
        )
        
        self.metrics_collector.log_system_health(health)
    
    def generate_daily_report(self, date: str = None) -> Dict[str, Any]:
        """日次レポート生成"""
        report = self.report_generator.generate_daily_report(date)
        
        # アラートチェック
        self.alert_manager.check_and_create_alerts()
        
        # アクティブアラートを追加
        report['active_alerts'] = self.alert_manager.get_active_alerts()
        
        return report
    
    def generate_weekly_report(self) -> Dict[str, Any]:
        """週次レポート生成"""
        return self.report_generator.generate_weekly_report()
    
    def create_dashboard(self, date: str = None) -> str:
        """ダッシュボード作成"""
        report = self.generate_daily_report(date)
        dashboard_path = self.report_generator.create_visual_dashboard(report)
        
        self.logger.info(f"Dashboard created: {dashboard_path}")
        return dashboard_path
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """リアルタイムメトリクス取得"""
        conn = self.db_manager.get_connection()
        
        # 過去1時間の統計
        current_stats = pd.read_sql_query('''
            SELECT 
                COUNT(*) as total_videos,
                AVG(processing_time) as avg_processing_time,
                AVG(quality_score) as avg_quality,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                SUM(cost) as total_cost
            FROM video_performance 
            WHERE created_at >= datetime('now', '-1 hour')
        ''', conn)
        
        # 最新のシステムヘルス
        system_health = pd.read_sql_query('''
            SELECT cpu_usage, memory_usage, disk_usage, queue_length, error_rate
            FROM system_health 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''', conn)
        
        # アクティブアラート数
        active_alerts_count = pd.read_sql_query('''
            SELECT COUNT(*) as alert_count
            FROM alerts 
            WHERE resolved = 0
        ''', conn)
        
        conn.close()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'current_stats': current_stats.to_dict('records')[0] if not current_stats.empty else {},
            'system_health': system_health.to_dict('records')[0] if not system_health.empty else {},
            'active_alerts_count': active_alerts_count.iloc[0]['alert_count'] if not active_alerts_count.empty else 0
        }
        
        return metrics
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """パフォーマンストレンド取得"""
        conn = self.db_manager.get_connection()
        
        start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
        
        trends = pd.read_sql_query('''
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as total_videos,
                AVG(processing_time) as avg_processing_time,
                AVG(quality_score) as avg_quality,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                SUM(cost) as daily_cost
            FROM video_performance 
            WHERE DATE(created_at) >= ?
            GROUP BY DATE(created_at)
            ORDER BY date
        ''', conn, params=[start_date])
        
        conn.close()
        
        return {
            'period_days': days,
            'trends': trends.to_dict('records') if not trends.empty else []
        }

# 使用例
def main():
    """動画パフォーマンス監視システムの使用例"""
    
    # 監視システム初期化
    monitor = VideoPerformanceMonitor()
    
    # サンプルデータ記録
    sample_video_data = {
        'video_id': 'video_001',
        'platform': 'youtube',
        'generation_time': 120.5,
        'processing_time': 45.2,
        'quality_score': 0.87,
        'file_size': 15728640,  # 15MB
        'success': True,
        'cost': 2.50
    }
    
    monitor.log_video_generation(sample_video_data)
    
    # 品質分析データ記録
    quality_data = {
        'technical_score': 0.85,
        'visual_score': 0.90,
        'audio_score': 0.82,
        'content_score': 0.88,
        'overall_score': 0.86,
        'recommendations': ['解像度を向上させてください', '音声品質の改善が必要です']
    }
    
    monitor.log_quality_analysis('video_001', quality_data)
    
    # システム状態記録
    monitor.log_system_status(
        cpu_usage=45.2,
        memory_usage=68.5,
        disk_usage=78.9,
        active_processes=12,
        queue_length=3,
        error_rate=0.02,
        avg_response_time=67.8
    )
    
    # 日次レポート生成
    daily_report = monitor.generate_daily_report()
    print("=== Daily Report ===")
    print(f"Total videos: {daily_report['summary']['total_videos']}")
    print(f"Success rate: {daily_report['summary']['overall_success_rate']:.1f}%")
    print(f"Total cost: ${daily_report['summary']['total_cost']:.2f}")
    print(f"Average quality: {daily_report['summary']['avg_quality_score']:.2f}")
    
    if daily_report['active_alerts']:
        print(f"\n⚠️  Active alerts: {len(daily_report['active_alerts'])}")
        for alert in daily_report['active_alerts']:
            print(f"  - {alert['severity'].upper()}: {alert['message']}")
    
    # ダッシュボード作成
    dashboard_path = monitor.create_dashboard()
    print(f"\nDashboard created: {dashboard_path}")
    
    # リアルタイムメトリクス取得
    real_time = monitor.get_real_time_metrics()
    print(f"\n=== Real-time Metrics ===")
    print(f"Timestamp: {real_time['timestamp']}")
    if real_time['current_stats']:
        print(f"Videos processed (last hour): {real_time['current_stats'].get('total_videos', 0)}")
        print(f"Average processing time: {real_time['current_stats'].get('avg_processing_time', 0):.1f}s")
    
    # パフォーマンストレンド
    trends = monitor.get_performance_trends(7)
    print(f"\n=== 7-Day Performance Trends ===")
    if trends['trends']:
        for day in trends['trends'][-3:]:  # 最近3日間
            print(f"{day['date']}: {day['total_videos']} videos, {day['success_rate']:.1f}% success rate")

if __name__ == "__main__":
    main()
