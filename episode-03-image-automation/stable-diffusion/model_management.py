import torch
import gc
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from typing import Dict, Optional, Any

class ModelManager:
    def __init__(self, device="cuda"):
        self.device = device
        self.loaded_models = {}
        self.model_configs = self.get_model_configurations()
        self.current_model = None
        self.max_loaded_models = 2  # メモリ制限による同時読み込み数
        
    def get_model_configurations(self) -> Dict[str, Dict]:
        """モデル設定辞書を返す"""
        return {
            'realistic': {
                'model_id': 'runwayml/stable-diffusion-v1-5',
                'description': 'リアリスティック画像生成',
                'best_for': ['商品撮影', 'ポートレート', '風景'],
                'default_params': {
                    'num_inference_steps': 25,
                    'guidance_scale': 7.5,
                    'stylize_strength': 0.4
                }
            },
            'anime': {
                'model_id': 'Linaqruf/anything-v3.0',
                'description': 'アニメ・イラスト生成',
                'best_for': ['キャラクター', 'イラスト', 'アニメ調'],
                'default_params': {
                    'num_inference_steps': 28,
                    'guidance_scale': 9.0,
                    'stylize_strength': 0.6
                }
            },
            'artistic': {
                'model_id': 'stabilityai/stable-diffusion-2-1-base',
                'description': 'アーティスティック画像生成',
                'best_for': ['抽象アート', 'クリエイティブ', 'コンセプトアート'],
                'default_params': {
                    'num_inference_steps': 30,
                    'guidance_scale': 8.0,
                    'stylize_strength': 0.7
                }
            },
            'photorealistic': {
                'model_id': 'SG161222/Realistic_Vision_V2.0',
                'description': 'フォトリアリスティック生成',
                'best_for': ['写真品質', '人物撮影', '商業写真'],
                'default_params': {
                    'num_inference_steps': 22,
                    'guidance_scale': 7.0,
                    'stylize_strength': 0.3
                }
            },
            'architecture': {
                'model_id': 'prompthero/openjourney-v4',
                'description': '建築・空間デザイン',
                'best_for': ['建築', 'インテリア', '空間デザイン'],
                'default_params': {
                    'num_inference_steps': 25,
                    'guidance_scale': 8.5,
                    'stylize_strength': 0.5
                }
            }
        }
        
    def load_model(self, model_type: str, force_reload: bool = False) -> bool:
        """指定されたモデルを読み込む"""
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # 既に読み込み済みの場合
        if model_type in self.loaded_models and not force_reload:
            self.current_model = model_type
            print(f"Model '{model_type}' already loaded")
            return True
            
        # メモリ管理
        self.manage_memory()
        
        try:
            print(f"Loading model: {model_type}")
            start_time = time.time()
            
            model_config = self.model_configs[model_type]
            model_id = model_config['model_id']
            
            # パイプライン作成
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            # スケジューラー設定
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )
            
            # デバイス転送と最適化
            pipe = pipe.to(self.device)
            pipe.enable_memory_efficient_attention()
            
            if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                pipe.enable_xformers_memory_efficient_attention()
                
            # 読み込み完了
            self.loaded_models[model_type] = {
                'pipeline': pipe,
                'config': model_config,
                'loaded_at': time.time(),
                'usage_count': 0
            }
            
            self.current_model = model_type
            load_time = time.time() - start_time
            
            print(f"✓ Model '{model_type}' loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load model '{model_type}': {e}")
            return False
            
    def unload_model(self, model_type: str):
        """指定されたモデルをメモリから解放"""
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]['pipeline']
            del self.loaded_models[model_type]
            print(f"Model '{model_type}' unloaded")
            
        torch.cuda.empty_cache()
        gc.collect()
        
    def manage_memory(self):
        """メモリ管理 - 古いモデルを自動削除"""
        if len(self.loaded_models) >= self.max_loaded_models:
            # 最も古いモデルを削除
            oldest_model = min(
                self.loaded_models.keys(),
                key=lambda x: self.loaded_models[x]['loaded_at']
            )
            
            print(f"Memory limit reached, unloading: {oldest_model}")
            self.unload_model(oldest_model)
            
    def generate_image(self, prompt: str, model_type: str = None, **kwargs) -> Any:
        """指定されたモデルで画像生成"""
        # デフォルトモデル使用
        if model_type is None:
            model_type = self.current_model or 'realistic'
            
        # モデル読み込み確認
        if model_type not in self.loaded_models:
            if not self.load_model(model_type):
                raise RuntimeError(f"Failed to load model: {model_type}")
                
        # モデル設定取得
        model_info = self.loaded_models[model_type]
        pipeline = model_info['pipeline']
        default_params = model_info['config']['default_params']
        
        # パラメータ統合
        params = {**default_params, **kwargs}
        
        # 使用回数カウント
        model_info['usage_count'] += 1
        
        # 生成実行
        with torch.autocast(self.device):
            result = pipeline(prompt, **params)
            
        return result.images[0]
        
    def get_optimal_model(self, prompt: str, category: str = None) -> str:
        """プロンプトとカテゴリに基づいて最適なモデルを推奨"""
        prompt_lower = prompt.lower()
        
        # キーワードベースのモデル選択
        if any(word in prompt_lower for word in ['anime', 'cartoon', 'illustration', 'character']):
            return 'anime'
        elif any(word in prompt_lower for word in ['photo', 'realistic', 'portrait', 'photography']):
            return 'photorealistic'
        elif any(word in prompt_lower for word in ['building', 'architecture', 'interior', 'room']):
            return 'architecture'
        elif any(word in prompt_lower for word in ['art', 'abstract', 'creative', 'concept']):
            return 'artistic'
        else:
            return 'realistic'
            
    def get_model_status(self) -> Dict:
        """現在のモデル状態を返す"""
        status = {
            'loaded_models': list(self.loaded_models.keys()),
            'current_model': self.current_model,
            'memory_usage': self.get_memory_usage(),
            'model_stats': {}
        }
        
        for model_name, info in self.loaded_models.items():
            status['model_stats'][model_name] = {
                'usage_count': info['usage_count'],
                'loaded_duration': time.time() - info['loaded_at'],
                'description': info['config']['description']
            }
            
        return status
        
    def get_memory_usage(self) -> Dict:
        """GPU/CPUメモリ使用状況"""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info['gpu'] = {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
            }
        else:
            memory_info['gpu'] = 'Not available'
            
        return memory_info
        
    def cleanup_all(self):
        """全モデルをクリーンアップ"""
        for model_type in list(self.loaded_models.keys()):
            self.unload_model(model_type)
        self.current_model = None
        print("All models cleaned up")

# 使用例とテスト
if __name__ == "__main__":
    manager = ModelManager()
    
    # モデル状態確認
    print("Initial status:", manager.get_model_status())
    
    # 複数モデルでテスト生成
    test_prompts = [
        ("A professional businesswoman in office", "realistic"),
        ("Anime character with blue hair", "anime"),
        ("Abstract colorful art piece", "artistic")
    ]
    
    for prompt, model_type in test_prompts:
        print(f"\nGenerating with {model_type}: {prompt}")
        image = manager.generate_image(prompt, model_type)
        # 画像保存処理等...
        
    # 最終状態確認
    print("\nFinal status:", manager.get_model_status())
    
    # クリーンアップ
    manager.cleanup_all()
