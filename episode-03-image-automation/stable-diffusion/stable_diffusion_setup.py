# GPU環境確認と最適化
import torch
import tensorflow as tf
import gc
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import xformers

class StableDiffusionEnvironment:
    def __init__(self):
        self.device = self.setup_device()
        self.memory_optimization()
        
    def setup_device(self):
        """GPU環境の確認と設定"""
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            device = torch.device("cpu")
            print("Using CPU (GPU not available)")
            
        return device
    
    def memory_optimization(self):
        """メモリ使用量の最適化"""
        # CUDA メモリクリア
        torch.cuda.empty_cache()
        gc.collect()
        
        # PyTorchメモリ使用量設定
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
            
        # TensorFlow GPU メモリ制限
        if tf.config.list_physical_devices('GPU'):
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
    def install_dependencies(self):
        """必要なライブラリのインストール"""
        packages = [
            "diffusers[torch]>=0.20.0",
            "transformers>=4.25.0",
            "accelerate>=0.20.0",
            "xformers>=0.0.20",
            "safetensors>=0.3.0",
            "opencv-python>=4.7.0",
            "pillow>=9.5.0",
            "requests>=2.28.0",
            "google-cloud-storage>=2.8.0"
        ]
        
        for package in packages:
            os.system(f"pip install {package}")
            
    def test_installation(self):
        """インストール確認テスト"""
        try:
            from diffusers import StableDiffusionPipeline
            print("✓ Diffusers installed successfully")
            
            import xformers
            print("✓ xFormers installed successfully")
            
            if torch.cuda.is_available():
                print("✓ CUDA available and working")
            else:
                print("⚠ CUDA not available, using CPU")
                
            return True
        except ImportError as e:
            print(f"✗ Installation test failed: {e}")
            return False
            
    def optimize_for_colab(self):
        """Google Colab特有の最適化"""
        # Colab環境変数設定
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TORCH_HOME'] = '/content/torch_cache'
        os.environ['HF_HOME'] = '/content/hf_cache'
        
        # キャッシュディレクトリ作成
        os.makedirs('/content/torch_cache', exist_ok=True)
        os.makedirs('/content/hf_cache', exist_ok=True)
        os.makedirs('/content/outputs', exist_ok=True)
        
        # Google Drive マウント（オプション）
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✓ Google Drive mounted successfully")
        except:
            print("⚠ Google Drive mount failed or not in Colab")

# 基本的なパイプライン設定
class BasicStableDiffusionPipeline:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.env = StableDiffusionEnvironment()
        self.model_id = model_id
        self.pipe = None
        
    def load_model(self):
        """モデル読み込み"""
        print(f"Loading model: {self.model_id}")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        # スケジューラー最適化
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # デバイス転送
        self.pipe = self.pipe.to(self.env.device)
        
        # メモリ最適化
        self.pipe.enable_memory_efficient_attention()
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            self.pipe.enable_xformers_memory_efficient_attention()
            
        print("✓ Model loaded and optimized")
        
    def generate_image(self, prompt, **kwargs):
        """画像生成"""
        if self.pipe is None:
            self.load_model()
            
        default_params = {
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'height': 512,
            'width': 512,
            'num_images_per_prompt': 1
        }
        
        # パラメータマージ
        params = {**default_params, **kwargs}
        
        # 生成実行
        with torch.autocast(self.env.device.type):
            result = self.pipe(prompt, **params)
            
        return result.images[0]
        
    def cleanup(self):
        """メモリクリーンアップ"""
        if self.pipe:
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()

# 使用例
if __name__ == "__main__":
    # 環境セットアップ
    sd_pipeline = BasicStableDiffusionPipeline()
    
    # テスト生成
    test_prompt = "A professional business woman in a modern office, corporate photography style, clean background"
    
    image = sd_pipeline.generate_image(
        prompt=test_prompt,
        num_inference_steps=25,
        guidance_scale=8.0
    )
    
    # 結果保存
    image.save('/content/outputs/test_generation.png')
    print("Test image generated and saved!")
    
    # クリーンアップ
    sd_pipeline.cleanup()
