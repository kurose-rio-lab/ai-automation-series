# 大規模バッチ処理システム
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time
import json
from datetime import datetime
import logging
from pathlib import Path
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class BatchRequest:
    """バッチリクエストデータクラス"""
    id: str
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 7.5
    api_provider: str = "stable_diffusion"
    priority: int = 1
    brand: Optional[str] = None
    category: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class BatchImageProcessor:
    """バッチ画像処理システム"""
    
    def __init__(self, max_concurrent=3, max_queue_size=1000):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.results = {}
        self.processing_status = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
        # 統計情報
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'average_processing_time': 0.0,
            'start_time': datetime.now()
        }
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 処理中フラグ
        self.is_processing = False
        self.shutdown_event = threading.Event()
    
    def add_request(self, request: BatchRequest):
        """リクエストをキューに追加"""
        
        try:
            # 優先度付きキューに追加（優先度が高いほど先に処理）
            priority = -request.priority  # 負の値で優先度を逆転
            self.task_queue.put((priority, request), timeout=5)
            
            self.logger.info(f"リクエスト追加: {request.id} (優先度: {request.priority})")
            
            return True
            
        except queue.Full:
            self.logger.error(f"キューが満杯です: {request.id}")
            return False
    
    def add_batch_requests(self, requests: List[BatchRequest]):
        """複数リクエストの一括追加"""
        
        added_count = 0
        
        for request in requests:
            if self.add_request(request):
                added_count += 1
        
        self.logger.info(f"バッチリクエスト追加完了: {added_count}/{len(requests)}")
        
        return added_count
    
    async def process_single_request(self, request: BatchRequest):
        """単一リクエストの処理"""
        
        start_time = time.time()
        
        try:
            # ステータス更新
            self.processing_status[request.id] = {
                'status': 'processing',
                'start_time': start_time,
                'api_provider': request.api_provider
            }
            
            # API選択とリクエスト処理
            if request.api_provider == "midjourney":
                result = await self.process_midjourney_request(request)
            elif request.api_provider == "stable_diffusion":
                result = await self.process_stable_diffusion_request(request)
            elif request.api_provider == "dalle3":
                result = await self.process_dalle3_request(request)
            else:
                raise ValueError(f"未対応のAPIプロバイダー: {request.api_provider}")
            
            # 処理時間計算
            processing_time = time.time() - start_time
            
            # 結果格納
            self.results[request.id] = {
                'success': result['success'],
                'request': request,
                'result': result,
                'processing_time': processing_time,
                'completed_at': datetime.now()
            }
            
            # 統計更新
            self.update_stats(True, processing_time)
            
            # ステータス更新
            self.processing_status[request.id]['status'] = 'completed'
            
            self.logger.info(f"処理完了: {request.id} ({processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            # エラー処理
            processing_time = time.time() - start_time
            
            error_result = {
                'success': False,
                'error': str(e),
                'request': request,
                'processing_time': processing_time,
                'completed_at': datetime.now()
            }
            
            self.results[request.id] = error_result
            self.update_stats(False, processing_time)
            
            # ステータス更新
            self.processing_status[request.id]['status'] = 'failed'
            
            self.logger.error(f"処理エラー: {request.id} - {str(e)}")
            
            return error_result
    
    async def process_midjourney_request(self, request: BatchRequest):
        """Midjourney API処理"""
        
        # Midjourney API呼び出し（実装例）
        async with aiohttp.ClientSession() as session:
            payload = {
                'prompt': request.prompt,
                'aspect_ratio': f"{request.width}:{request.height}",
                'quality': 2 if request.steps > 25 else 1
            }
            
            # API エンドポイント（実際のURLに置き換え）
            url = "https://api.midjourney.com/v1/imagine"
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'image_url': result.get('image_url'),
                        'api_provider': 'midjourney'
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Midjourney API error: {response.status}"
                    }
    
    async def process_stable_diffusion_request(self, request: BatchRequest):
        """Stable Diffusion処理"""
        
        # Stable Diffusion生成（実装例）
        try:
            # 実際のStable Diffusion呼び出し
            # ここでは模擬的な処理
            await asyncio.sleep(request.steps * 0.1)  # 処理時間シミュレーション
            
            return {
                'success': True,
                'image_path': f"generated_images/{request.id}.png",
                'api_provider': 'stable_diffusion',
                'generation_info': {
                    'prompt': request.prompt,
                    'steps': request.steps,
                    'guidance_scale': request.guidance_scale
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Stable Diffusion error: {str(e)}"
            }
    
    async def process_dalle3_request(self, request: BatchRequest):
        """DALL-E 3 API処理"""
        
        async with aiohttp.ClientSession() as session:
            payload = {
                'prompt': request.prompt,
                'n': 1,
                'size': f"{request.width}x{request.height}",
                'quality': 'hd' if request.steps > 25 else 'standard'
            }
            
            headers = {
                'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                'Content-Type': 'application/json'
            }
            
            url = "https://api.openai.com/v1/images/generations"
            
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'image_url': result['data'][0]['url'],
                        'api_provider': 'dalle3'
                    }
                else:
                    return {
                        'success': False,
                        'error': f"DALL-E 3 API error: {response.status}"
                    }
    
    async def process_batch(self):
        """バッチ処理メインループ"""
        
        self.is_processing = True
        active_tasks = set()
        
        self.logger.info("バッチ処理開始")
        
        try:
            while not self.shutdown_event.is_set():
                # 新しいタスクを追加
                while len(active_tasks) < self.max_concurrent:
                    try:
                        # キューから次のリクエストを取得
                        priority, request = self.task_queue.get(timeout=1)
                        
                        # 非同期タスクとして処理開始
                        task = asyncio.create_task(self.process_single_request(request))
                        active_tasks.add(task)
                        
                        self.logger.info(f"タスク開始: {request.id} (アクティブ: {len(active_tasks)})")
                        
                    except queue.Empty:
                        # キューが空の場合は少し待機
                        if not active_tasks:
                            await asyncio.sleep(0.1)
                        break
                
                # 完了したタスクの処理
                if active_tasks:
                    done, active_tasks = await asyncio.wait(
                        active_tasks,
                        timeout=0.1,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for task in done:
                        try:
                            result = await task
                            self.logger.info(f"タスク完了: {result.get('request', {}).get('id', 'unknown')}")
                        except Exception as e:
                            self.logger.error(f"タスク実行エラー: {e}")
        
        except Exception as e:
            self.logger.error(f"バッチ処理エラー: {e}")
        
        finally:
            # 残りのタスクを完了まで待機
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            
            self.is_processing = False
            self.logger.info("バッチ処理終了")
    
    def update_stats(self, success: bool, processing_time: float):
        """統計情報更新"""
        
        self.stats['total_processed'] += 1
        
        if success:
            self.stats['successful'] += 1
        else:
            self.stats['failed'] += 1
        
        # 平均処理時間更新
        total_time = self.stats['average_processing_time'] * (self.stats['total_processed'] - 1)
        self.stats['average_processing_time'] = (total_time + processing_time) / self.stats['total_processed']
    
    def get_queue_status(self):
        """キューステータス取得"""
        
        return {
            'queue_size': self.task_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'is_processing': self.is_processing,
            'active_processors': len([s for s in self.processing_status.values() if s['status'] == 'processing'])
        }
    
    def get_processing_stats(self):
        """処理統計取得"""
        
        current_time = datetime.now()
        elapsed_time = (current_time - self.stats['start_time']).total_seconds()
        
        return {
            **self.stats,
            'elapsed_time': elapsed_time,
            'processing_rate': self.stats['total_processed'] / elapsed_time if elapsed_time > 0 else 0,
            'success_rate': self.stats['successful'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0
        }
    
    def get_results(self, request_ids: List[str] = None):
        """結果取得"""
        
        if request_ids is None:
            return self.results
        
        return {req_id: self.results.get(req_id) for req_id in request_ids}
    
    def save_results(self, output_file: str = "batch_results.json"):
        """結果保存"""
        
        # 結果をシリアライズ可能な形式に変換
        serializable_results = {}
        
        for req_id, result in self.results.items():
            serializable_results[req_id] = {
                'success': result['success'],
                'processing_time': result['processing_time'],
                'completed_at': result['completed_at'].isoformat(),
                'request': {
                    'id': result['request'].id,
                    'prompt': result['request'].prompt,
                    'api_provider': result['request'].api_provider,
                    'priority': result['request'].priority
                }
            }
            
            if result['success']:
                serializable_results[req_id]['result'] = result['result']
            else:
                serializable_results[req_id]['error'] = result['result'].get('error')
        
        # ファイル保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"結果保存完了: {output_file}")
    
    def shutdown(self):
        """システム終了"""
        
        self.logger.info("システム終了開始")
        self.shutdown_event.set()
        
        # 処理完了まで待機
        while self.is_processing:
            time.sleep(0.1)
        
        self.executor.shutdown(wait=True)
        self.logger.info("システム終了完了")

# 使用例
async def main():
    # バッチプロセッサー初期化
    processor = BatchImageProcessor(max_concurrent=3)
    
    # テストリクエスト作成
    test_requests = [
        BatchRequest(
            id="test_001",
            prompt="A beautiful landscape",
            api_provider="stable_diffusion",
            priority=1
        ),
        BatchRequest(
            id="test_002",
            prompt="A professional portrait",
            api_provider="midjourney",
            priority=2
        ),
        BatchRequest(
            id="test_003",
            prompt="An abstract artwork",
            api_provider="dalle3",
            priority=1
        )
    ]
    
    # リクエスト追加
    processor.add_batch_requests(test_requests)
    
    # バッチ処理開始
    batch_task = asyncio.create_task(processor.process_batch())
    
    # 進捗監視
    while processor.get_queue_status()['queue_size'] > 0 or processor.is_processing:
        stats = processor.get_processing_stats()
        queue_status = processor.get_queue_status()
        
        print(f"キューサイズ: {queue_status['queue_size']}")
        print(f"処理済み: {stats['total_processed']} (成功: {stats['successful']}, 失敗: {stats['failed']})")
        print(f"平均処理時間: {stats['average_processing_time']:.2f}s")
        print(f"処理レート: {stats['processing_rate']:.2f} 件/秒")
        print("---")
        
        await asyncio.sleep(5)
    
    # 終了
    processor.shutdown()
    await batch_task
    
    # 結果保存
    processor.save_results()
    
    print("バッチ処理完了!")

if __name__ == "__main__":
    asyncio.run(main())
