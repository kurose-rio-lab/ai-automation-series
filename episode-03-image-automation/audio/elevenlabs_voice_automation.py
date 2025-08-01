# elevenlabs_voice_automation.py
import requests
import json
import time
import os
from typing import Dict, List, Optional
import logging

class ElevenLabsVoiceAutomation:
    """ElevenLabs音声生成の完全自動化システム"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        self.setup_logging()
    
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('elevenlabs_automation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_available_voices(self) -> List[Dict]:
        """利用可能な音声リストを取得"""
        try:
            response = requests.get(
                f"{self.base_url}/voices",
                headers={"xi-api-key": self.api_key}
            )
            response.raise_for_status()
            
            voices = response.json()["voices"]
            self.logger.info(f"取得した音声数: {len(voices)}")
            
            return voices
            
        except Exception as e:
            self.logger.error(f"音声リスト取得エラー: {str(e)}")
            return []
    
    def select_optimal_voice(self, content_type: str, gender: str = "female") -> str:
        """コンテンツタイプに最適な音声を自動選択"""
        voice_mapping = {
            "educational": {
                "female": "Rachel",  # 教育的で聞き取りやすい
                "male": "Josh"
            },
            "professional": {
                "female": "Bella",   # プロフェッショナル
                "male": "Antoni"
            },
            "casual": {
                "female": "Elli",    # カジュアル
                "male": "Sam"
            }
        }
        
        voices = self.get_available_voices()
        target_name = voice_mapping.get(content_type, {}).get(gender, "Rachel")
        
        for voice in voices:
            if voice["name"] == target_name:
                self.logger.info(f"選択された音声: {target_name} (ID: {voice['voice_id']})")
                return voice["voice_id"]
        
        # デフォルトとして最初の音声を返す
        if voices:
            return voices[0]["voice_id"]
        
        raise Exception("利用可能な音声が見つかりません")
    
    def optimize_text_for_speech(self, text: str) -> str:
        """音声生成用にテキストを最適化"""
        # SSML記法の追加
        optimizations = {
            "、": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "～": "...",
            # 数字の読み上げ改善
            "AI": "エーアイ",
            "API": "エーピーアイ",
            "URL": "ユーアールエル",
            "HTML": "エイチティーエムエル",
            "CSS": "シーエスエス",
            "JavaScript": "ジャバスクリプト"
        }
        
        optimized_text = text
        for old, new in optimizations.items():
            optimized_text = optimized_text.replace(old, new)
        
        # 長い文章を適切な長さに分割
        sentences = optimized_text.split('.')
        optimized_sentences = []
        
        for sentence in sentences:
            if len(sentence) > 300:
                # 長すぎる文章をカンマで分割
                parts = sentence.split(',')
                optimized_sentences.extend(parts)
            else:
                optimized_sentences.append(sentence)
        
        return '. '.join(optimized_sentences)
    
    def generate_speech(
        self, 
        text: str, 
        voice_id: str,
        output_path: str,
        voice_settings: Optional[Dict] = None
    ) -> bool:
        """音声生成実行"""
        try:
            # デフォルト音声設定
            if voice_settings is None:
                voice_settings = {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            
            # テキスト最適化
            optimized_text = self.optimize_text_for_speech(text)
            
            data = {
                "text": optimized_text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": voice_settings
            }
            
            self.logger.info(f"音声生成開始: {len(text)}文字")
            
            response = requests.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                json=data,
                headers=self.headers
            )
            response.raise_for_status()
            
            # 音声ファイル保存
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"音声生成完了: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"音声生成エラー: {str(e)}")
            return False
    
    def batch_generate_speech(
        self, 
        script_segments: List[Dict],
        voice_config: Dict
    ) -> List[str]:
        """台本セグメントの一括音声生成"""
        generated_files = []
        
        for i, segment in enumerate(script_segments):
            output_path = f"audio/segment_{i+1:03d}.mp3"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            success = self.generate_speech(
                text=segment["text"],
                voice_id=voice_config["voice_id"],
                output_path=output_path,
                voice_settings=voice_config.get("settings")
            )
            
            if success:
                generated_files.append(output_path)
                
                # 感情や速度の調整
                if segment.get("emotion"):
                    self.apply_emotion_adjustment(output_path, segment["emotion"])
            
            # API制限対応
            time.sleep(1)
        
        return generated_files
    
    def apply_emotion_adjustment(self, audio_path: str, emotion: str):
        """感情に応じた音声調整"""
        emotion_settings = {
            "excited": {"stability": 0.3, "similarity_boost": 0.8},
            "calm": {"stability": 0.7, "similarity_boost": 0.6},
            "professional": {"stability": 0.6, "similarity_boost": 0.7},
            "friendly": {"stability": 0.4, "similarity_boost": 0.75}
        }
        
        if emotion in emotion_settings:
            self.logger.info(f"感情調整適用: {emotion}")
            # 実際の実装では音声後処理ライブラリを使用
    
    def create_voice_profile(self, sample_audio_path: str, name: str) -> str:
        """カスタム音声プロファイル作成"""
        try:
            with open(sample_audio_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                data = {
                    'name': name,
                    'description': f'Custom voice for {name}'
                }
                
                response = requests.post(
                    f"{self.base_url}/voices/add",
                    files=files,
                    data=data,
                    headers={"xi-api-key": self.api_key}
                )
                response.raise_for_status()
                
                voice_id = response.json()["voice_id"]
                self.logger.info(f"カスタム音声作成完了: {voice_id}")
                
                return voice_id
                
        except Exception as e:
            self.logger.error(f"カスタム音声作成エラー: {str(e)}")
            return None
    
    def monitor_generation_usage(self) -> Dict:
        """使用量監視"""
        try:
            response = requests.get(
                f"{self.base_url}/user",
                headers={"xi-api-key": self.api_key}
            )
            response.raise_for_status()
            
            user_data = response.json()
            usage_info = {
                "character_count": user_data.get("character_count", 0),
                "character_limit": user_data.get("character_limit", 0),
                "remaining_characters": user_data.get("character_limit", 0) - user_data.get("character_count", 0)
            }
            
            self.logger.info(f"使用量: {usage_info['character_count']}/{usage_info['character_limit']}")
            
            return usage_info
            
        except Exception as e:
            self.logger.error(f"使用量取得エラー: {str(e)}")
            return {}

# 使用例
if __name__ == "__main__":
    # ElevenLabs自動化システムの実行例
    automation = ElevenLabsVoiceAutomation("your_api_key_here")
    
    # 台本セグメント
    script = [
        {
            "text": "皆さん、こんにちは。黒瀬理央です。今日はAI動画生成の完全自動化について解説します。",
            "emotion": "friendly"
        },
        {
            "text": "このシステムを使えば、動画制作の効率が劇的に向上します。",
            "emotion": "excited"
        }
    ]
    
    # 音声設定
    voice_config = {
        "voice_id": automation.select_optimal_voice("educational", "female"),
        "settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    # 一括音声生成
    audio_files = automation.batch_generate_speech(script, voice_config)
    print(f"生成された音声ファイル: {audio_files}")
