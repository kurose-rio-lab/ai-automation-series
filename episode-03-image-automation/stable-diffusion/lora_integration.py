# LoRAモデル統合システム
import torch
from diffusers import StableDiffusionPipeline
import safetensors
import os
import json
from pathlib import Path
import requests
import zipfile

class LoRAManager:
    """LoRAモデル管理システム"""
    
    def __init__(self, lora_directory="./lora_models", device="cuda"):
        self.lora_directory = Path(lora_directory)
        self.device = device
        self.loaded_loras = {}
        self.active_loras = {}
        
        # LoRAディレクトリ作成
        self.lora_directory.mkdir(exist_ok=True)
        
        # 企業ブランド別LoRA設定
        self.brand_loras = {
            'tech_company': {
                'file': 'tech_brand_v2.safetensors',
                'description': 'テクノロジー企業向けブランド',
                'weight': 0.8,
                'style': 'modern, clean, corporate'
            },
            'fashion': {
                'file': 'fashion_style_v1.safetensors',
                'description': 'ファッション業界向けスタイル',
                'weight': 0.9,
                'style': 'elegant, stylish, trendy'
            },
            'food_service': {
                'file': 'food_photography_v3.safetensors',
                'description': '食品・レストラン向け',
                'weight': 0.7,
                'style': 'appetizing, professional food photography'
            },
            'healthcare': {
                'file': 'medical_clean_v1.safetensors',
                'description': '医療・ヘルスケア向け',
                'weight': 0.8,
                'style': 'clean, professional, trustworthy'
            },
            'education': {
                'file': 'education_friendly_v2.safetensors',
                'description': '教育・学習向け',
                'weight': 0.7,
                'style': 'friendly, educational, approachable'
            }
        }
    
    def download_lora(self, url, filename):
        """LoRAモデルをダウンロード"""
        
        file_path = self.lora_directory / filename
        
        if file_path.exists():
            print(f"LoRA既存: {filename}")
            return file_path
        
        print(f"LoRAダウンロード開始: {filename}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"LoRAダウンロード完了: {filename}")
            return file_path
            
        except Exception as e:
            print(f"LoRAダウンロードエラー: {e}")
            return None
    
    def load_lora(self, lora_path):
        """LoRAモデル読み込み"""
        
        if lora_path in self.loaded_loras:
            return self.loaded_loras[lora_path]
        
        try:
            # safetensorsファイル読み込み
            if lora_path.endswith('.safetensors'):
                lora_state_dict = safetensors.torch.load_file(lora_path, device=self.device)
            else:
                lora_state_dict = torch.load(lora_path, map_location=self.device)
            
            self.loaded_loras[lora_path] = lora_state_dict
            print(f"LoRA読み込み完了: {lora_path}")
            
            return lora_state_dict
            
        except Exception as e:
            print(f"LoRA読み込みエラー: {e}")
            return None
    
    def apply_lora_to_pipeline(self, pipe, lora_config, weight=1.0):
        """パイプラインにLoRAを適用"""
        
        lora_path = self.lora_directory / lora_config['file']
        
        if not lora_path.exists():
            print(f"LoRAファイルが見つかりません: {lora_path}")
            return pipe
        
        try:
            # LoRA適用
            pipe.load_lora_weights(str(lora_path))
            pipe.fuse_lora(weight * lora_config['weight'])
            
            print(f"LoRA適用完了: {lora_config['file']} (weight: {weight * lora_config['weight']})")
            
            return pipe
            
        except Exception as e:
            print(f"LoRA適用エラー: {e}")
            return pipe
    
    def get_brand_lora_config(self, brand_name):
        """ブランド別LoRA設定取得"""
        
        if brand_name in self.brand_loras:
            return self.brand_loras[brand_name]
        
        print(f"Warning: ブランド '{brand_name}' のLoRA設定が見つかりません")
        return None
    
    def enhance_prompt_for_brand(self, prompt, brand_name):
        """ブランド特化プロンプト強化"""
        
        brand_config = self.get_brand_lora_config(brand_name)
        if brand_config is None:
            return prompt
        
        # ブランドスタイルをプロンプトに追加
        enhanced_prompt = f"{prompt}, {brand_config['style']}"
        
        # 業界特有のキーワード追加
        if brand_name == 'tech_company':
            enhanced_prompt += ", professional, high-tech, innovation"
        elif brand_name == 'fashion':
            enhanced_prompt += ", haute couture, fashion photography, elegant"
        elif brand_name == 'food_service':
            enhanced_prompt += ", delicious, appetizing, professional food photography"
        elif brand_name == 'healthcare':
            enhanced_prompt += ", clean, medical, professional, trustworthy"
        elif brand_name == 'education':
            enhanced_prompt += ", educational, learning, friendly, approachable"
        
        return enhanced_prompt
    
    def create_brand_specific_pipeline(self, base_pipeline, brand_name, weight=1.0):
        """ブランド特化パイプライン作成"""
        
        brand_config = self.get_brand_lora_config(brand_name)
        if brand_config is None:
            return base_pipeline
        
        # LoRA適用
        enhanced_pipeline = self.apply_lora_to_pipeline(base_pipeline, brand_config, weight)
        
        return enhanced_pipeline
    
    def generate_brand_image(self, base_pipeline, prompt, brand_name, **kwargs):
        """ブランド特化画像生成"""
        
        # プロンプト強化
        enhanced_prompt = self.enhance_prompt_for_brand(prompt, brand_name)
        
        # パイプライン強化
        enhanced_pipeline = self.create_brand_specific_pipeline(base_pipeline, brand_name)
        
        # 生成設定
        generation_config = {
            'num_inference_steps': 25,
            'guidance_scale': 8.0,
            'width': 512,
            'height': 512,
            'negative_prompt': "low quality, blurry, distorted, unprofessional"
        }
        
        # 設定マージ
        generation_config.update(kwargs)
        
        try:
            # 画像生成
            with torch.autocast("cuda"):
                result = enhanced_pipeline(
                    prompt=enhanced_prompt,
                    **generation_config
                )
            
            return {
                'success': True,
                'image': result.images[0],
                'enhanced_prompt': enhanced_prompt,
                'brand_name': brand_name,
                'config': generation_config
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prompt': enhanced_prompt,
                'brand_name': brand_name
            }
    
    def batch_generate_brand_images(self, base_pipeline, prompts, brand_name, **kwargs):
        """ブランド特化バッチ生成"""
        
        results = []
        
        # パイプライン一度だけ強化
        enhanced_pipeline = self.create_brand_specific_pipeline(base_pipeline, brand_name)
        
        for i, prompt in enumerate(prompts):
            print(f"ブランド画像生成 ({i+1}/{len(prompts)}): {prompt[:50]}...")
            
            # プロンプト強化
            enhanced_prompt = self.enhance_prompt_for_brand(prompt, brand_name)
            
            # 生成設定
            generation_config = {
                'num_inference_steps': 25,
                'guidance_scale': 8.0,
                'width': 512,
                'height': 512,
                'negative_prompt': "low quality, blurry, distorted, unprofessional"
            }
            generation_config.update(kwargs)
            
            try:
                with torch.autocast("cuda"):
                    result = enhanced_pipeline(
                        prompt=enhanced_prompt,
                        **generation_config
                    )
                
                results.append({
                    'success': True,
                    'image': result.images[0],
                    'enhanced_prompt': enhanced_prompt,
                    'brand_name': brand_name,
                    'original_prompt': prompt
                })
                
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'prompt': enhanced_prompt,
                    'brand_name': brand_name,
                    'original_prompt': prompt
                })
            
            # メモリ管理
            if i % 3 == 0:
                torch.cuda.empty_cache()
        
        return results
    
    def save_brand_config(self, filename="brand_config.json"):
        """ブランド設定保存"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.brand_loras, f, indent=2, ensure_ascii=False)
        
        print(f"ブランド設定保存: {filename}")
    
    def load_brand_config(self, filename="brand_config.json"):
        """ブランド設定読み込み"""
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.brand_loras = json.load(f)
            
            print(f"ブランド設定読み込み: {filename}")
            
        except FileNotFoundError:
            print(f"設定ファイルが見つかりません: {filename}")
        except Exception as e:
            print(f"設定読み込みエラー: {e}")

# 使用例
if __name__ == "__main__":
    from diffusers import StableDiffusionPipeline
    
    # LoRAマネージャー初期化
    lora_manager = LoRAManager()
    
    # 基本パイプライン読み込み
    base_pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    # テスト生成
    test_cases = [
        ("A professional office environment", "tech_company"),
        ("A beautiful fashion model", "fashion"),
        ("A delicious burger and fries", "food_service"),
        ("A clean medical facility", "healthcare"),
        ("A friendly classroom scene", "education")
    ]
    
    for prompt, brand in test_cases:
        print(f"\n=== {brand} ブランド生成テスト ===")
        print(f"プロンプト: {prompt}")
        
        result = lora_manager.generate_brand_image(
            base_pipeline, 
            prompt, 
            brand,
            num_inference_steps=20
        )
        
        if result['success']:
            print(f"生成成功!")
            print(f"強化プロンプト: {result['enhanced_prompt']}")
            # result['image'].show()
        else:
            print(f"生成失敗: {result['error']}")
