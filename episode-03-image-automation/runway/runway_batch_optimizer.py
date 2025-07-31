import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import heapq
from concurrent.futures import ThreadPoolExecutor
import sqlite3

class Priority(Enum):
    """優先度レベル定義"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

@dataclass
class VideoRequest:
    """動画生成リクエストデータクラス"""
    id: str
    prompt: str
    image_path: Optional[str] = None
    duration: int = 10
    style: str = "default"
    priority: Priority = Priority.MEDIUM
    max_retries: int = 3
    created_at: datetime = None
    estimated_cost: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class CostTracker:
    """コスト追跡・予測システム"""
    
    def __init__(self):
        self.base_costs = {
            'text_to_video': 0.05,  # per second
            'image_to_video': 0.08, # per second
            'video_to_video': 0.12  # per second
        }
        self.daily_usage = {}
        self.monthly_budget = 1000.0  # デフォルト月予算
        
    def estimate_cost(self, request: VideoRequest) -> float:
        """コスト見積もり"""
        base_type = 'image_to_video' if request.image_path else 'text_to_video'
        base_cost = self.base_costs[base_type] * request.duration
        
        # 優先度による料金調整
        priority_multiplier = {
            Priority.LOW: 0.8,
            Priority.MEDIUM: 1.0,
            Priority.HIGH: 1.2,
            Priority.URGENT: 1.5
        }
        
        return base_cost * priority_multiplier[request.priority]
    
    def check_budget_availability(self, estimated_cost: float) -> bool:
        """予算チェック"""
        today = datetime.now().date()
        current_month = today.strftime('%Y-%m')
        
        monthly_usage = sum(
            day_usage for date, day_usage in self.daily_usage.items()
            if date.startswith(current_month)
        )
        
        return (monthly_usage + estimated_cost) <= self.monthly_budget
    
    def log_usage(self, cost: float):
        """使用量記録"""
        today = datetime.now().date().isoformat()
        if today not in self.daily_usage:
            self.daily_usage[today] = 0.0
        self.daily_usage[today] += cost

class PriorityQueueManager:
    """優先度キュー管理システム"""
    
    def __init__(self):
        self.queue = []
        self.counter = 0  # 同一優先度での順序保証
        
    def add_request(self, request: VideoRequest):
        """リクエスト追加"""
        # 優先度を負の値にしてheapqで最大ヒープのように動作させる
        priority_value = -request.priority.value
        heapq.heappush(self.queue, (priority_value, self.counter, request))
        self.counter += 1
        
    def get_next_request(self) -> Optional[VideoRequest]:
        """次のリクエスト取得"""
        if self.queue:
            _, _, request = heapq.heappop(self.queue)
            return request
        return None
    
    def size(self) -> int:
        """キューサイズ"""
        return len(self.queue)
    
    def peek_priority(self) -> Optional[Priority]:
        """次のリクエストの優先度確認"""
        if self.queue:
            priority_value, _, _ = self.queue[0]
            return Priority(-priority_value)
        return None

class RunwayBatchOptimizer:
    """RunwayML APIバッチ処理最適化システム"""
    
    def __init__(self, api_key: str, max_concurrent: int = 3):
        """初期化"""
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.base_url = "https://api.runwayml.com/v1"
        
        # コンポーネント初期化
        self.cost_tracker = CostTracker()
        self.queue_manager = PriorityQueueManager()
        
        # 実行状況管理
        self.active_jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        
        # レート制限管理
        self.last_request_time = {}
        self.min_interval = 1.0  # 秒間隔
        
        # データベース初期化
        self.init_database()
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def init_database(self):
        """SQLiteデータベース初期化"""
        self.conn = sqlite3.connect('runway_optimizer.db')
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_history (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                duration INTEGER,
                priority TEXT,
                status TEXT,
                cost REAL,
                processing_time REAL,
                created_at DATETIME,
                completed_at DATETIME,
                error_message TEXT
            )
        ''')
        
        self.conn.commit()
        
    async def add_request(self, request: VideoRequest) -> bool:
        """リクエスト追加"""
        # コスト見積もり
        estimated_cost = self.cost_tracker.estimate_cost(request)
        request.estimated_cost = estimated_cost
        
        # 予算チェック
        if not self.cost_tracker.check_budget_availability(estimated_cost):
            self.logger.warning(f"Budget exceeded for request {request.id}")
            return False
        
        # キューに追加
        self.queue_manager.add_request(request)
        
        # データベース記録
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO job_history 
            (id, prompt, duration, priority, status, cost, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            request.id, request.prompt, request.duration,
            request.priority.name, 'queued', estimated_cost,
            request.created_at
        ))
        self.conn.commit()
        
        self.logger.info(f"Request {request.id} added to queue with priority {request.priority.name}")
        return True
        
    async def process_queue(self):
        """キュー処理メインループ"""
        self.logger.info("Starting batch processing...")
        
        while True:
            # アクティブジョブ数チェック
            if len(self.active_jobs) >= self.max_concurrent:
                await asyncio.sleep(1)
                continue
                
            # 次のリクエスト取得
            request = self.queue_manager.get_next_request()
            if not request:
                await asyncio.sleep(5)
                continue
            
            # 並行処理でジョブ実行
            task = asyncio.create_task(self.process_single_request(request))
            self.active_jobs[request.id] = task
            
            # 完了したジョブのクリーンアップ
            await self.cleanup_completed_jobs()
            
    async def process_single_request(self, request: VideoRequest) -> Dict[str, Any]:
        """単一リクエスト処理"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing request {request.id}")
            
            # レート制限チェック
            await self.enforce_rate_limit()
            
            # RunwayML API呼び出し
            result = await self.call_runway_api(request)
            
            if result['success']:
                # 成功処理
                processing_time = time.time() - start_time
                self.cost_tracker.log_usage(request.estimated_cost)
                
                # データベース更新
                cursor = self.conn.cursor()
                cursor.execute('''
                    UPDATE job_history 
                    SET status = ?, processing_time = ?, completed_at = ?
                    WHERE id = ?
                ''', ('completed', processing_time, datetime.now(), request.id))
                self.conn.commit()
                
                self.completed_jobs[request.id] = result
                self.logger.info(f"Request {request.id} completed successfully")
                
            else:
                # エラー処理
                await self.handle_request_failure(request, result.get('error', 'Unknown error'))
                
        except Exception as e:
            await self.handle_request_failure(request, str(e))
            
        finally:
            # アクティブジョブから削除
            if request.id in self.active_jobs:
                del self.active_jobs[request.id]
                
        return result
        
    async def call_runway_api(self, request: VideoRequest) -> Dict[str, Any]:
        """RunwayML API呼び出し"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # リクエストペイロード構築
        payload = {
            'prompt': request.prompt,
            'duration': request.duration,
            'style': request.style
        }
        
        if request.image_path:
            payload['image'] = request.image_path
            
        async with aiohttp.ClientSession() as session:
            try:
                # ジョブ開始
                async with session.post(
                    f"{self.base_url}/generate",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        job_data = await response.json()
                        job_id = job_data['id']
                        
                        # ジョブ完了待機
                        return await self.wait_for_completion(session, headers, job_id)
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'error': f'API Error: {response.status} - {error_text}'
                        }
                        
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Request failed: {str(e)}'
                }
                
    async def wait_for_completion(self, session: aiohttp.ClientSession, 
                                headers: Dict, job_id: str) -> Dict[str, Any]:
        """ジョブ完了待機"""
        max_wait_time = 600  # 10分
        check_interval = 10  # 10秒間隔
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait_time:
            async with session.get(
                f"{self.base_url}/jobs/{job_id}",
                headers=headers
            ) as response:
                
                if response.status == 200:
                    job_status = await response.json()
                    
                    if job_status['status'] == 'completed':
                        return {
                            'success': True,
                            'job_id': job_id,
                            'video_url': job_status['output']['video_url'],
                            'processing_time': time.time() - start_time
                        }
                    elif job_status['status'] == 'failed':
                        return {
                            'success': False,
                            'error': job_status.get('error', 'Job failed')
                        }
                        
            await asyncio.sleep(check_interval)
            
        return {
            'success': False,
            'error': 'Timeout waiting for job completion'
        }
        
    async def handle_request_failure(self, request: VideoRequest, error: str):
        """リクエスト失敗処理"""
        # リトライ回数チェック
        if hasattr(request, 'retry_count'):
            request.retry_count += 1
        else:
            request.retry_count = 1
            
        if request.retry_count <= request.max_retries:
            # リトライ
            self.logger.warning(f"Retrying request {request.id} (attempt {request.retry_count})")
            await asyncio.sleep(2 ** request.retry_count)  # 指数バックオフ
            self.queue_manager.add_request(request)
        else:
            # 最終失敗
            self.logger.error(f"Request {request.id} failed permanently: {error}")
            self.failed_jobs[request.id] = {'error': error, 'request': request}
            
            # データベース更新
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE job_history 
                SET status = ?, error_message = ?, completed_at = ?
                WHERE id = ?
            ''', ('failed', error, datetime.now(), request.id))
            self.conn.commit()
            
    async def enforce_rate_limit(self):
        """レート制限制御"""
        current_time = time.time()
        if hasattr(self, 'last_api_call'):
            time_diff = current_time - self.last_api_call
            if time_diff < self.min_interval:
                await asyncio.sleep(self.min_interval - time_diff)
        
        self.last_api_call = current_time
        
    async def cleanup_completed_jobs(self):
        """完了ジョブクリーンアップ"""
        completed_tasks = []
        for job_id, task in self.active_jobs.items():
            if task.done():
                completed_tasks.append(job_id)
                
        for job_id in completed_tasks:
            del self.active_jobs[job_id]
            
    def get_status_report(self) -> Dict[str, Any]:
        """状況レポート生成"""
        return {
            'queue_size': self.queue_manager.size(),
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'daily_cost': self.cost_tracker.daily_usage.get(
                datetime.now().date().isoformat(), 0.0
            ),
            'next_priority': self.queue_manager.peek_priority().name if self.queue_manager.peek_priority() else None
        }
        
    async def shutdown(self):
        """システム終了"""
        self.logger.info("Shutting down batch optimizer...")
        
        # アクティブジョブの完了待機
        if self.active_jobs:
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)
            
        # データベース接続クローズ
        self.conn.close()
        
        self.logger.info("Shutdown complete")

# 使用例
async def main():
    """使用例デモンストレーション"""
    optimizer = RunwayBatchOptimizer(api_key="your_runway_api_key", max_concurrent=3)
    
    # サンプルリクエスト追加
    requests = [
        VideoRequest(
            id="video_001",
            prompt="A beautiful sunset over mountains",
            duration=10,
            priority=Priority.HIGH
        ),
        VideoRequest(
            id="video_002", 
            prompt="Modern city skyline at night",
            duration=15,
            priority=Priority.MEDIUM
        ),
        VideoRequest(
            id="video_003",
            prompt="Peaceful forest scene with flowing river",
            duration=12,
            priority=Priority.LOW
        )
    ]
    
    # リクエスト追加
    for request in requests:
        await optimizer.add_request(request)
    
    # バッチ処理開始（別のタスクで実行）
    process_task = asyncio.create_task(optimizer.process_queue())
    
    # 状況監視
    for _ in range(10):
        await asyncio.sleep(30)
        status = optimizer.get_status_report()
        print(f"Status: {status}")
        
        if status['queue_size'] == 0 and status['active_jobs'] == 0:
            break
    
    # システム終了
    process_task.cancel()
    await optimizer.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
