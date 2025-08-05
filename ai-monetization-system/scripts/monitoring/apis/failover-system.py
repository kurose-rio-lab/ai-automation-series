"""
AI API 自動フェイルオーバーシステム
複数のAIプロバイダー間での負荷分散と冗長性確保
"""
import asyncio
import aiohttp
import time
import random
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProviderStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

@dataclass
class AIProvider:
    name: str
    api_key: str
    endpoint: str
    priority: int  # 1が最高優先度
    max_requests_per_minute: int
    current_requests: int = 0
    last_request_time: float = 0
    status: ProviderStatus = ProviderStatus.ACTIVE
    failure_count: int = 0
    last_failure_time: float = 0

class AIProviderManager:
    def __init__(self):
        self.providers = self._initialize_providers()
        self.circuit_breaker_threshold = 5  # 5回失敗で回路遮断
        self.circuit_breaker_timeout = 300  # 5分間の待機時間
        
    def _initialize_providers(self) -> List[AIProvider]:
        """AIプロバイダーの初期化"""
        return [
            AIProvider(
                name="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
                endpoint="https://api.openai.com/v1/chat/completions",
                priority=1,
                max_requests_per_minute=60
            ),
            AIProvider(
                name="anthropic",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                endpoint="https://api.anthropic.com/v1/messages",
                priority=2,
                max_requests_per_minute=50
            ),
            AIProvider(
                name="google",
                api_key=os.getenv("GOOGLE_AI_API_KEY"),
                endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                priority=3,
                max_requests_per_minute=60
            ),
            AIProvider(
                name="azure",
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                endpoint=f"https://{os.getenv('AZURE_RESOURCE_NAME')}.openai.azure.com/openai/deployments/gpt-4/chat/completions",
                priority=4,
                max_requests_per_minute=120
            )
        ]
    
    def get_available_providers(self) -> List[AIProvider]:
        """利用可能なプロバイダーのリストを取得"""
        current_time = time.time()
        available = []
        
        for provider in self.providers:
            # 回路遮断器チェック
            if self._is_circuit_breaker_open(provider, current_time):
                continue
                
            # レート制限チェック
            if self._is_rate_limited(provider, current_time):
                continue
                
            # ステータスチェック
            if provider.status == ProviderStatus.ACTIVE:
                available.append(provider)
                
        # 優先度順にソート
        return sorted(available, key=lambda p: p.priority)
    
    def _is_circuit_breaker_open(self, provider: AIProvider, current_time: float) -> bool:
        """回路遮断器の状態確認"""
        if provider.failure_count >= self.circuit_breaker_threshold:
            time_since_last_failure = current_time - provider.last_failure_time
            if time_since_last_failure < self.circuit_breaker_timeout:
                return True
            else:
                # タイムアウト後にリセット
                provider.failure_count = 0
                provider.status = ProviderStatus.ACTIVE
                logger.info(f"回路遮断器リセット: {provider.name}")
        return False
    
    def _is_rate_limited(self, provider: AIProvider, current_time: float) -> bool:
        """レート制限チェック"""
        time_window = 60  # 1分間
        if current_time - provider.last_request_time < time_window:
            if provider.current_requests >= provider.max_requests_per_minute:
                return True
        else:
            # 時間窓口リセット
            provider.current_requests = 0
            provider.last_request_time = current_time
        return False
    
    async def get_ai_response(self, prompt: str, max_tokens: int = 4000) -> Optional[str]:
        """AIレスポンスの取得（フェイルオーバー付き）"""
        available_providers = self.get_available_providers()
        
        if not available_providers:
            logger.error("利用可能なAIプロバイダーがありません")
            return None
        
        for provider in available_providers:
            try:
                response = await self._call_provider(provider, prompt, max_tokens)
                if response:
                    # 成功時の処理
                    provider.current_requests += 1
                    provider.failure_count = 0
                    logger.info(f"成功: {provider.name}")
                    return response
                    
            except Exception as e:
                # 失敗時の処理
                provider.failure_count += 1
                provider.last_failure_time = time.time()
                logger.error(f"プロバイダー失敗 {provider.name}: {str(e)}")
                
                # 次のプロバイダーを試行
                continue
        
        logger.error("全てのAIプロバイダーが失敗しました")
        return None
    
    async def _call_provider(self, provider: AIProvider, prompt: str, max_tokens: int) -> Optional[str]:
        """個別プロバイダーへのAPI呼び出し"""
        headers = self._get_headers(provider)
        payload = self._get_payload(provider, prompt, max_tokens)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                provider.endpoint,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._extract_response(provider.name, data)
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
    
    def _get_headers(self, provider: AIProvider) -> Dict[str, str]:
        """プロバイダー別ヘッダー生成"""
        headers = {"Content-Type": "application/json"}
        
        if provider.name == "openai":
            headers["Authorization"] = f"Bearer {provider.api_key}"
        elif provider.name == "anthropic":
            headers["x-api-key"] = provider.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif provider.name == "google":
            headers["Authorization"] = f"Bearer {provider.api_key}"
        elif provider.name == "azure":
            headers["api-key"] = provider.api_key
            
        return headers
    
    def _get_payload(self, provider: AIProvider, prompt: str, max_tokens: int) -> Dict:
        """プロバイダー別ペイロード生成"""
        if provider.name in ["openai", "azure"]:
            return {
                "model": "gpt-4-turbo-preview",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
        elif provider.name == "anthropic":
            return {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
        elif provider.name == "google":
            return {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.7
                }
            }
    
    def _extract_response(self, provider_name: str, data: Dict) -> str:
        """プロバイダー別レスポンス抽出"""
        if provider_name in ["openai", "azure"]:
            return data["choices"][0]["message"]["content"]
        elif provider_name == "anthropic":
            return data["content"][0]["text"]
        elif provider_name == "google":
            return data["candidates"][0]["content"]["parts"][0]["text"]
        
        return ""
    
    def get_health_status(self) -> Dict[str, any]:
        """システム健全性ステータス"""
        status = {
            "timestamp": time.time(),
            "providers": []
        }
        
        for provider in self.providers:
            provider_status = {
                "name": provider.name,
                "status": provider.status.value,
                "failure_count": provider.failure_count,
                "current_requests": provider.current_requests,
                "circuit_breaker_open": self._is_circuit_breaker_open(provider, time.time())
            }
            status["providers"].append(provider_status)
        
        return status

# 使用例
async def main():
    manager = AIProviderManager()
    
    # AI応答の取得
    response = await manager.get_ai_response(
        "ビジネス分析レポートを作成してください。",
        max_tokens=2000
    )
    
    if response:
        print(f"AI応答: {response}")
    else:
        print("AI応答の取得に失敗しました")
    
    # ヘルスステータス確認
    health = manager.get_health_status()
    print(f"システム状態: {health}")

if __name__ == "__main__":
    asyncio.run(main())
