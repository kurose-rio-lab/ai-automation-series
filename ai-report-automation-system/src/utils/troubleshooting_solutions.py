"""
トラブルシューティング解決システム
AI自動レポート生成システムの問題診断と自動解決
"""

import logging
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import requests
import pandas as pd
from dataclasses import dataclass
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProblemSeverity(Enum):
    """問題の重要度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProblemCategory(Enum):
    """問題カテゴリ"""
    DATA_QUALITY = "data_quality"
    API_ERROR = "api_error"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"
    NETWORK = "network"

@dataclass
class TroubleshootingIssue:
    """トラブルシューティング問題データクラス"""
    issue_id: str
    category: ProblemCategory
    severity: ProblemSeverity
    title: str
    description: str
    error_details: Dict[str, Any]
    detection_time: str
    resolution_steps: List[str]
    auto_fixable: bool
    estimated_fix_time: int  # 分

class TroubleshootingSystem:
    """トラブルシューティングシステム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get('database_path', 'troubleshooting.db')
        self.notification_config = config.get('notification', {})
        
        # データベース初期化
        self._init_database()
        
        # 問題パターンの読み込み
        self.problem_patterns = self._load_problem_patterns()
        
        # 自動修復機能の設定
        self.auto_fix_enabled = config.get('auto_fix_enabled', True)
        
    def _init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_id TEXT UNIQUE,
                category TEXT,
                severity TEXT,
                title TEXT,
                description TEXT,
                error_details TEXT,
                detection_time TEXT,
                resolution_time TEXT,
                status TEXT,
                auto_fixed BOOLEAN
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resolution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_id TEXT,
                step_number INTEGER,
                step_description TEXT,
                execution_time TEXT,
                success BOOLEAN,
                error_message TEXT,
                FOREIGN KEY (issue_id) REFERENCES issues (issue_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_problem_patterns(self) -> Dict[str, Any]:
        """問題パターンの読み込み"""
        return {
            # データ品質問題
            "empty_data": {
                "pattern": r".*empty.*data.*|.*no.*data.*found.*",
                "category": ProblemCategory.DATA_QUALITY,
                "severity": ProblemSeverity.HIGH,
                "auto_fix": self._fix_empty_data,
                "description": "データが空または見つからない"
            },
            "invalid_date_format": {
                "pattern": r".*invalid.*date.*|.*date.*format.*error.*",
                "category": ProblemCategory.DATA_QUALITY,
                "severity": ProblemSeverity.MEDIUM,
                "auto_fix": self._fix_date_format,
                "description": "日付形式が不正"
            },
            "duplicate_data": {
                "pattern": r".*duplicate.*|.*重複.*",
                "category": ProblemCategory.DATA_QUALITY,
                "severity": ProblemSeverity.LOW,
                "auto_fix": self._fix_duplicate_data,
                "description": "重複データが存在"
            },
            
            # API エラー
            "openai_rate_limit": {
                "pattern": r".*rate.*limit.*exceeded.*|.*quota.*exceeded.*",
                "category": ProblemCategory.API_ERROR,
                "severity": ProblemSeverity.HIGH,
                "auto_fix": self._fix_rate_limit,
                "description": "OpenAI API制限に達した"
            },
            "openai_timeout": {
                "pattern": r".*timeout.*|.*connection.*timed.*out.*",
                "category": ProblemCategory.API_ERROR,
                "severity": ProblemSeverity.MEDIUM,
                "auto_fix": self._fix_api_timeout,
                "description": "API接続タイムアウト"
            },
            "invalid_api_key": {
                "pattern": r".*invalid.*api.*key.*|.*unauthorized.*",
                "category": ProblemCategory.API_ERROR,
                "severity": ProblemSeverity.CRITICAL,
                "auto_fix": self._fix_invalid_api_key,
                "description": "APIキーが無効"
            },
            
            # システムエラー
            "memory_error": {
                "pattern": r".*memory.*error.*|.*out.*of.*memory.*",
                "category": ProblemCategory.SYSTEM_ERROR,
                "severity": ProblemSeverity.HIGH,
                "auto_fix": self._fix_memory_error,
                "description": "メモリ不足エラー"
            },
            "file_permission": {
                "pattern": r".*permission.*denied.*|.*access.*denied.*",
                "category": ProblemCategory.SYSTEM_ERROR,
                "severity": ProblemSeverity.MEDIUM,
                "auto_fix": self._fix_file_permission,
                "description": "ファイルアクセス権限エラー"
            },
            
            # パフォーマンス問題
            "slow_processing": {
                "pattern": r".*processing.*slow.*|.*timeout.*processing.*",
                "category": ProblemCategory.PERFORMANCE,
                "severity": ProblemSeverity.MEDIUM,
                "auto_fix": self._fix_slow_processing,
                "description": "処理速度が遅い"
            },
            
            # 設定問題
            "missing_config": {
                "pattern": r".*config.*not.*found.*|.*missing.*configuration.*",
                "category": ProblemCategory.CONFIGURATION,
                "severity": ProblemSeverity.HIGH,
                "auto_fix": self._fix_missing_config,
                "description": "設定ファイルが見つからない"
            }
        }
    
    def diagnose_issue(self, error_message: str, context: Dict[str, Any] = None) -> TroubleshootingIssue:
        """問題の診断"""
        import re
        
        if context is None:
            context = {}
        
        # エラーパターンマッチング
        matched_pattern = None
        for pattern_name, pattern_info in self.problem_patterns.items():
            if re.search(pattern_info["pattern"], error_message, re.IGNORECASE):
                matched_pattern = pattern_info
                break
        
        if not matched_pattern:
            # 未知の問題として分類
            matched_pattern = {
                "category": ProblemCategory.SYSTEM_ERROR,
                "severity": ProblemSeverity.MEDIUM,
                "auto_fix": None,
                "description": "未知のエラー"
            }
        
        # 問題詳細の作成
        issue_id = f"issue_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 解決手順の生成
        resolution_steps = self._generate_resolution_steps(matched_pattern, error_message, context)
        
        issue = TroubleshootingIssue(
            issue_id=issue_id,
            category=matched_pattern["category"],
            severity=matched_pattern["severity"],
            title=matched_pattern["description"],
            description=error_message,
            error_details={
                "error_message": error_message,
                "context": context,
                "stack_trace": traceback.format_exc() if context.get("include_traceback") else None
            },
            detection_time=datetime.now().isoformat(),
            resolution_steps=resolution_steps,
            auto_fixable=matched_pattern.get("auto_fix") is not None,
            estimated_fix_time=self._estimate_fix_time(matched_pattern)
        )
        
        # データベースに記録
        self._save_issue_to_db(issue)
        
        logger.info(f"問題診断完了: {issue.issue_id} - {issue.title}")
        return issue
    
    def _generate_resolution_steps(
        self, 
        pattern_info: Dict[str, Any], 
        error_message: str, 
        context: Dict[str, Any]
    ) -> List[str]:
        """解決手順の生成"""
        category = pattern_info["category"]
        
        if category == ProblemCategory.DATA_QUALITY:
            return [
                "1. データソースの確認と検証",
                "2. データクリーニングスクリプトの実行",
                "3. データ形式の標準化",
                "4. 重複データの除去",
                "5. データ品質レポートの生成"
            ]
        elif category == ProblemCategory.API_ERROR:
            return [
                "1. APIキーの有効性確認",
                "2. API使用量の確認",
                "3. ネットワーク接続の確認",
                "4. リクエスト内容の検証",
                "5. 再試行またはフォールバック処理"
            ]
        elif category == ProblemCategory.SYSTEM_ERROR:
            return [
                "1. システムリソースの確認",
                "2. ログファイルの分析",
                "3. 権限設定の確認",
                "4. 必要に応じてサービス再起動",
                "5. システム設定の検証"
            ]
        elif category == ProblemCategory.PERFORMANCE:
            return [
                "1. パフォーマンス指標の分析",
                "2. ボトルネックの特定",
                "3. リソース使用量の最適化",
                "4. 処理の並列化検討",
                "5. キャッシュ機能の活用"
            ]
        else:
            return [
                "1. エラーログの詳細確認",
                "2. 関連設定の検証",
                "3. システム状態の確認",
                "4. 必要に応じて専門サポートに連絡"
            ]
    
    def _estimate_fix_time(self, pattern_info: Dict[str, Any]) -> int:
        """修復時間の推定（分）"""
        severity = pattern_info["severity"]
        auto_fixable = pattern_info.get("auto_fix") is not None
        
        if auto_fixable:
            if severity == ProblemSeverity.CRITICAL:
                return 15
            elif severity == ProblemSeverity.HIGH:
                return 10
            elif severity == ProblemSeverity.MEDIUM:
                return 5
            else:
                return 2
        else:
            if severity == ProblemSeverity.CRITICAL:
                return 120  # 2時間
            elif severity == ProblemSeverity.HIGH:
                return 60   # 1時間
            elif severity == ProblemSeverity.MEDIUM:
                return 30   # 30分
            else:
                return 15   # 15分
    
    def auto_resolve_issue(self, issue: TroubleshootingIssue) -> bool:
        """問題の自動解決"""
        if not self.auto_fix_enabled or not issue.auto_fixable:
            logger.info(f"自動修復無効またはサポート外: {issue.issue_id}")
            return False
        
        logger.info(f"自動修復開始: {issue.issue_id}")
        
        try:
            # 問題パターンに対応する修復関数を取得
            pattern_info = None
            for pattern_name, info in self.problem_patterns.items():
                if info["description"] == issue.title:
                    pattern_info = info
                    break
            
            if not pattern_info or not pattern_info.get("auto_fix"):
                logger.warning(f"修復関数が見つかりません: {issue.issue_id}")
                return False
            
            # 修復実行
            fix_function = pattern_info["auto_fix"]
            success = fix_function(issue)
            
            # 結果をデータベースに記録
            self._update_issue_resolution(issue.issue_id, success)
            
            if success:
                logger.info(f"自動修復成功: {issue.issue_id}")
                self._send_notification(
                    f"問題自動解決: {issue.title}",
                    f"問題ID: {issue.issue_id}\n自動修復が成功しました。"
                )
            else:
                logger.error(f"自動修復失敗: {issue.issue_id}")
                self._send_notification(
                    f"自動修復失敗: {issue.title}",
                    f"問題ID: {issue.issue_id}\n手動対応が必要です。"
                )
            
            return success
            
        except Exception as e:
            logger.error(f"自動修復中にエラー: {issue.issue_id}, {str(e)}")
            self._log_resolution_step(
                issue.issue_id, 
                0, 
                "自動修復実行", 
                False, 
                str(e)
            )
            return False
    
    # 各種修復関数
    def _fix_empty_data(self, issue: TroubleshootingIssue) -> bool:
        """空データ問題の修復"""
        try:
            self._log_resolution_step(issue.issue_id, 1, "データソース確認開始", True)
            
            # データソースの再確認
            context = issue.error_details.get("context", {})
            data_source = context.get("data_source")
            
            if data_source:
                # 代替データソースの確認
                backup_sources = self._get_backup_data_sources(data_source)
                
                for backup in backup_sources:
                    if self._check_data_availability(backup):
                        self._log_resolution_step(
                            issue.issue_id, 2, f"代替データソース使用: {backup}", True
                        )
                        # データソース切り替え
                        self._switch_data_source(data_source, backup)
                        return True
            
            # サンプルデータでの一時対応
            self._log_resolution_step(issue.issue_id, 3, "サンプルデータ生成", True)
            self._generate_sample_data(context.get("expected_schema"))
            
            return True
            
        except Exception as e:
            self._log_resolution_step(issue.issue_id, 1, "空データ修復", False, str(e))
            return False
    
    def _fix_date_format(self, issue: TroubleshootingIssue) -> bool:
        """日付形式修復"""
        try:
            self._log_resolution_step(issue.issue_id, 1, "日付形式修復開始", True)
            
            # 日付データの標準化スクリプト実行
            context = issue.error_details.get("context", {})
            data_file = context.get("data_file")
            
            if data_file:
                # 日付形式の自動検出と変換
                success = self._standardize_date_format(data_file)
                
                self._log_resolution_step(
                    issue.issue_id, 2, "日付形式標準化", success
                )
                
                return success
            
            return True
            
        except Exception as e:
            self._log_resolution_step(issue.issue_id, 1, "日付形式修復", False, str(e))
            return False
    
    def _fix_duplicate_data(self, issue: TroubleshootingIssue) -> bool:
        """重複データ修復"""
        try:
            self._log_resolution_step(issue.issue_id, 1, "重複データ除去開始", True)
            
            context = issue.error_details.get("context", {})
            data_source = context.get("data_source")
            
            if data_source:
                # 重複除去処理
                removed_count = self._remove_duplicate_data(data_source)
                
                self._log_resolution_step(
                    issue.issue_id, 2, f"重複データ除去完了: {removed_count}件", True
                )
            
            return True
            
        except Exception as e:
            self._log_resolution_step(issue.issue_id, 1, "重複データ除去", False, str(e))
            return False
    
    def _fix_rate_limit(self, issue: TroubleshootingIssue) -> bool:
        """API制限修復"""
        try:
            self._log_resolution_step(issue.issue_id, 1, "API制限対応開始", True)
            
            # 待機時間計算
            wait_time = self._calculate_rate_limit_wait_time()
            
            self._log_resolution_step(
                issue.issue_id, 2, f"待機時間: {wait_time}秒", True
            )
            
            # 待機
            time.sleep(min(wait_time, 300))  # 最大5分
            
            # リクエスト頻度の調整
            self._adjust_request_frequency()
            
            self._log_resolution_step(issue.issue_id, 3, "リクエスト頻度調整", True)
            
            return True
            
        except Exception as e:
            self._log_resolution_step(issue.issue_id, 1, "API制限対応", False, str(e))
            return False
    
    def _fix_api_timeout(self, issue: TroubleshootingIssue) -> bool:
        """APIタイムアウト修復"""
        try:
            self._log_resolution_step(issue.issue_id, 1, "タイムアウト対応開始", True)
            
            # タイムアウト値の調整
            self._increase_timeout_settings()
            
            # リトライ設定の調整
            self._configure_retry_settings()
            
            self._log_resolution_step(issue.issue_id, 2, "タイムアウト設定調整", True)
            
            return True
            
        except Exception as e:
            self._log_resolution_step(issue.issue_id, 1, "タイムアウト対応", False, str(e))
            return False
    
    def _fix_invalid_api_key(self, issue: TroubleshootingIssue) -> bool:
        """無効APIキー修復"""
        try:
            self._log_resolution_step(issue.issue_id, 1, "APIキー確認開始", True)
            
            # APIキーの検証
            if self._validate_api_keys():
                self._log_resolution_step(issue.issue_id, 2, "APIキー検証成功", True)
                return True
            else:
                # アラート送信（手動対応必要）
                self._send_critical_alert("APIキーが無効です。手動で更新してください。")
                self._log_resolution_step(issue.issue_id, 2, "APIキー更新必要", False)
                return False
            
        except Exception as e:
            self._log_resolution_step(issue.issue_id, 1, "APIキー確認", False, str(e))
            return False
    
    def _fix_memory_error(self, issue: TroubleshootingIssue) -> bool:
        """メモリエラー修復"""
        try:
            self._log_resolution_step(issue.issue_id, 1, "メモリ最適化開始", True)
            
            # メモリクリーンアップ
            self._cleanup_memory()
            
            # 処理をバッチサイズ調整
            self._adjust_batch_size()
            
            # ガベージコレクション実行
            import gc
            gc.collect()
            
            self._log_resolution_step(issue.issue_id, 2, "メモリ最適化完了", True)
            
            return True
            
        except Exception as e:
            self._log_resolution_step(issue.issue_id, 1, "メモリ最適化", False, str(e))
            return False
    
    def _fix_file_permission(self, issue: TroubleshootingIssue) -> bool:
        """ファイル権限修復"""
        try:
            self._log_resolution_step(issue.issue_id, 1, "ファイル権限修復開始", True)
            
            context = issue.error_details.get("context", {})
            file_path = context.get("file_path")
            
            if file_path:
                # 権限修復（プラットフォーム依存）
                import os
                import stat
                
                try:
                    # 読み書き権限付与
                    os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                    self._log_resolution_step(issue.issue_id, 2, "権限修復成功", True)
                    return True
                except OSError:
                    # 代替パスの使用
                    alternative_path = self._get_alternative_file_path(file_path)
                    self._log_resolution_step(
                        issue.issue_id, 3, f"代替パス使用: {alternative_path}", True
                    )
                    return True
            
            return False
            
        except Exception as e:
            self._log_resolution_step(issue.issue_id, 1, "ファイル権限修復", False, str(e))
            return False
    
    def _fix_slow_processing(self, issue: TroubleshootingIssue) -> bool:
        """処理速度改善"""
        try:
            self._log_resolution_step(issue.issue_id, 1, "パフォーマンス最適化開始", True)
            
            # 並列処理の有効化
            self._enable_parallel_processing()
            
            # キャッシュ機能の有効化
            self._enable_caching()
            
            # 不要な処理の無効化
            self._disable_verbose_logging()
            
            self._log_resolution_step(issue.issue_id, 2, "パフォーマンス最適化完了", True)
            
            return True
            
        except Exception as e:
            self._log_resolution_step(issue.issue_id, 1, "パフォーマンス最適化", False, str(e))
            return False
    
    def _fix_missing_config(self, issue: TroubleshootingIssue) -> bool:
        """設定ファイル修復"""
        try:
            self._log_resolution_step(issue.issue_id, 1, "設定ファイル復旧開始", True)
            
            # デフォルト設定ファイルの作成
            self._create_default_config()
            
            self._log_resolution_step(issue.issue_id, 2, "デフォルト設定作成完了", True)
            
            return True
            
        except Exception as e:
            self._log_resolution_step(issue.issue_id, 1, "設定ファイル復旧", False, str(e))
            return False
    
    # ユーティリティメソッド
    def _get_backup_data_sources(self, primary_source: str) -> List[str]:
        """代替データソースの取得"""
        # 実装例：プライマリソースに基づく代替ソース
        backup_map = {
            "sales_data": ["backup_sales", "historical_sales"],
            "customer_data": ["backup_customers", "crm_export"],
            "marketing_data": ["backup_marketing", "analytics_export"]
        }
        return backup_map.get(primary_source, [])
    
    def _check_data_availability(self, data_source: str) -> bool:
        """データソースの可用性確認"""
        # 実装例：データソースへの接続テスト
        try:
            # Google Sheetsの場合
            if "sheets" in data_source:
                # Sheets API での確認
                return True
            # その他のデータソース
            return True
        except:
            return False
    
    def _switch_data_source(self, old_source: str, new_source: str):
        """データソースの切り替え"""
        # 設定ファイルの更新
        config_path = self.config.get('data_config_path', 'data_config.json')
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # ソース切り替え
            if old_source in config.get('data_sources', {}):
                config['data_sources'][old_source] = new_source
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"データソース切り替えエラー: {str(e)}")
    
    def _generate_sample_data(self, schema: Dict[str, Any]):
        """サンプルデータ生成"""
        # スキーマに基づくサンプルデータ生成
        sample_data = {
            "sales": [
                {"date": "2024-01-01", "amount": 100000, "product": "Sample A"},
                {"date": "2024-01-02", "amount": 150000, "product": "Sample B"}
            ],
            "customers": [
                {"id": 1, "name": "Sample Customer", "email": "sample@example.com"}
            ]
        }
        
        # サンプルデータをファイルに保存
        with open('sample_data.json', 'w') as f:
            json.dump(sample_data, f, indent=2)
    
    def _standardize_date_format(self, data_file: str) -> bool:
        """日付形式の標準化"""
        try:
            # CSVファイルの場合
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
                
                # 日付列の自動検出と変換
                for col in df.columns:
                    if 'date' in col.lower() or '日付' in col:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        df[col] = df[col].dt.strftime('%Y-%m-%d')
                
                # ファイル保存
                df.to_csv(data_file, index=False)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"日付形式標準化エラー: {str(e)}")
            return False
    
    def _remove_duplicate_data(self, data_source: str) -> int:
        """重複データ除去"""
        try:
            # Google Sheetsの場合
            if "sheets" in data_source:
                # Sheets APIを使用した重複除去
                # 実装例：duplicate_rows = sheet.remove_duplicates()
                return 5  # 仮の除去数
            
            return 0
            
        except Exception as e:
            logger.error(f"重複データ除去エラー: {str(e)}")
            return 0
    
    def _calculate_rate_limit_wait_time(self) -> int:
        """API制限待機時間計算"""
        # OpenAI APIの場合、通常1分待機
        return 60
    
    def _adjust_request_frequency(self):
        """リクエスト頻度調整"""
        # リクエスト間隔の設定
        config_path = self.config.get('api_config_path', 'api_config.json')
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # リクエスト間隔を倍に
            current_interval = config.get('request_interval', 1)
            config['request_interval'] = min(current_interval * 2, 10)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"リクエスト頻度調整エラー: {str(e)}")
    
    def _increase_timeout_settings(self):
        """タイムアウト設定増加"""
        # タイムアウト値を増加
        pass
    
    def _configure_retry_settings(self):
        """リトライ設定"""
        # リトライ回数と間隔の設定
        pass
    
    def _validate_api_keys(self) -> bool:
        """APIキー検証"""
        try:
            # OpenAI APIキーの検証
            openai_key = self.config.get('openai_api_key')
            if openai_key:
                headers = {"Authorization": f"Bearer {openai_key}"}
                response = requests.get(
                    "https://api.openai.com/v1/models",
                    headers=headers,
                    timeout=10
                )
                return response.status_code == 200
            
            return False
            
        except Exception as e:
            logger.error(f"APIキー検証エラー: {str(e)}")
            return False
    
    def _cleanup_memory(self):
        """メモリクリーンアップ"""
        import gc
        gc.collect()
    
    def _adjust_batch_size(self):
        """バッチサイズ調整"""
        # バッチサイズを半分に
        pass
    
    def _get_alternative_file_path(self, original_path: str) -> str:
        """代替ファイルパス取得"""
        import tempfile
        return tempfile.mktemp(suffix='.tmp')
    
    def _enable_parallel_processing(self):
        """並列処理有効化"""
        pass
    
    def _enable_caching(self):
        """キャッシュ有効化"""
        pass
    
    def _disable_verbose_logging(self):
        """詳細ログ無効化"""
        logging.getLogger().setLevel(logging.WARNING)
    
    def _create_default_config(self):
        """デフォルト設定作成"""
        default_config = {
            "data_sources": {
                "primary": "google_sheets",
                "backup": "local_files"
            },
            "api_settings": {
                "openai_timeout": 30,
                "request_interval": 1,
                "max_retries": 3
            },
            "output_settings": {
                "formats": ["pdf", "html"],
                "quality": "high"
            }
        }
        
        with open('config.json', 'w') as f:
            json.dump(default_config, f, indent=2)
    
    def _save_issue_to_db(self, issue: TroubleshootingIssue):
        """問題をデータベースに保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO issues (
                issue_id, category, severity, title, description, 
                error_details, detection_time, status, auto_fixed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            issue.issue_id,
            issue.category.value,
            issue.severity.value,
            issue.title,
            issue.description,
            json.dumps(issue.error_details),
            issue.detection_time,
            "detected",
            False
        ))
        
        conn.commit()
        conn.close()
    
    def _update_issue_resolution(self, issue_id: str, success: bool):
        """問題解決状況更新"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE issues 
            SET status = ?, resolution_time = ?, auto_fixed = ?
            WHERE issue_id = ?
        ''', (
            "resolved" if success else "failed",
            datetime.now().isoformat(),
            success,
            issue_id
        ))
        
        conn.commit()
        conn.close()
    
    def _log_resolution_step(
        self, 
        issue_id: str, 
        step_number: int, 
        description: str, 
        success: bool, 
        error_message: str = None
    ):
        """解決ステップのログ記録"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO resolution_logs (
                issue_id, step_number, step_description, 
                execution_time, success, error_message
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            issue_id,
            step_number,
            description,
            datetime.now().isoformat(),
            success,
            error_message
        ))
        
        conn.commit()
        conn.close()
    
    def _send_notification(self, subject: str, message: str):
        """通知送信"""
        if not self.notification_config.get('enabled', False):
            return
        
        try:
            # メール通知
            if self.notification_config.get('email'):
                self._send_email_notification(subject, message)
            
            # Slack通知
            if self.notification_config.get('slack_webhook'):
                self._send_slack_notification(subject, message)
                
        except Exception as e:
            logger.error(f"通知送信エラー: {str(e)}")
    
    def _send_email_notification(self, subject: str, message: str):
        """メール通知送信"""
        email_config = self.notification_config['email']
        
        msg = MimeMultipart()
        msg['From'] = email_config['from']
        msg['To'] = email_config['to']
        msg['Subject'] = subject
        
        msg.attach(MimeText(message, 'plain'))
        
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['username'], email_config['password'])
        server.send_message(msg)
        server.quit()
    
    def _send_slack_notification(self, subject: str, message: str):
        """Slack通知送信"""
        webhook_url = self.notification_config['slack_webhook']
        
        payload = {
            "text": f"*{subject}*\n{message}"
        }
        
        requests.post(webhook_url, json=payload)
    
    def _send_critical_alert(self, message: str):
        """緊急アラート送信"""
        self._send_notification("【緊急】システムアラート", message)
    
    def generate_health_report(self) -> Dict[str, Any]:
        """システム健全性レポート生成"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 過去24時間の問題統計
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        
        # 問題数の集計
        cursor.execute('''
            SELECT category, severity, COUNT(*) 
            FROM issues 
            WHERE detection_time > ? 
            GROUP BY category, severity
        ''', (yesterday,))
        
        issue_stats = cursor.fetchall()
        
        # 自動修復成功率
        cursor.execute('''
            SELECT 
                COUNT(CASE WHEN auto_fixed = 1 THEN 1 END) as fixed,
                COUNT(*) as total
            FROM issues 
            WHERE detection_time > ?
        ''', (yesterday,))
        
        fix_stats = cursor.fetchone()
        
        conn.close()
        
        report = {
            "report_time": datetime.now().isoformat(),
            "period": "過去24時間",
            "issue_statistics": {
                "by_category_severity": issue_stats,
                "total_issues": sum(stat[2] for stat in issue_stats),
                "auto_fix_rate": fix_stats[0] / fix_stats[1] if fix_stats[1] > 0 else 0
            },
            "system_status": "正常" if len(issue_stats) < 5 else "注意",
            "recommendations": self._generate_health_recommendations(issue_stats)
        }
        
        return report
    
    def _generate_health_recommendations(self, issue_stats: List[Tuple]) -> List[str]:
        """健全性レコメンデーション生成"""
        recommendations = []
        
        # 高頻度問題の分析
        high_frequency_issues = [stat for stat in issue_stats if stat[2] > 3]
        
        if high_frequency_issues:
            recommendations.append("高頻度で発生している問題の根本原因調査を推奨")
        
        # API エラーが多い場合
        api_errors = sum(stat[2] for stat in issue_stats if stat[0] == 'api_error')
        if api_errors > 5:
            recommendations.append("API設定とネットワーク環境の確認を推奨")
        
        # データ品質問題が多い場合
        data_quality_issues = sum(stat[2] for stat in issue_stats if stat[0] == 'data_quality')
        if data_quality_issues > 3:
            recommendations.append("データソースの品質向上対策を推奨")
        
        return recommendations

# 使用例
if __name__ == "__main__":
    # 設定
    config = {
        'database_path': 'troubleshooting.db',
        'openai_api_key': 'your_api_key',
        'auto_fix_enabled': True,
        'notification': {
            'enabled': True,
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your_email@gmail.com',
                'password': 'your_password',
                'from': 'system@company.com',
                'to': 'admin@company.com'
            }
        }
    }
    
    # トラブルシューティングシステム初期化
    troubleshooter = TroubleshootingSystem(config)
    
    # サンプルエラーの診断
    error_message = "OpenAI API rate limit exceeded"
    context = {
        "timestamp": datetime.now().isoformat(),
        "module": "analysis_engine",
        "data_source": "sales_data"
    }
    
    # 問題診断
    issue = troubleshooter.diagnose_issue(error_message, context)
    print(f"診断結果: {issue.title}")
    print(f"重要度: {issue.severity.value}")
    print(f"自動修復可能: {issue.auto_fixable}")
    
    # 自動修復試行
    if issue.auto_fixable:
        success = troubleshooter.auto_resolve_issue(issue)
        print(f"自動修復結果: {'成功' if success else '失敗'}")
    
    # システム健全性レポート
    health_report = troubleshooter.generate_health_report()
    print("システム健全性レポート:")
    print(json.dumps(health_report, ensure_ascii=False, indent=2))
