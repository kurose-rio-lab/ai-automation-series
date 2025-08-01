# bgm_auto_selector.py
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
from typing import Dict, List, Tuple
import logging
from moviepy.editor import VideoFileClip, CompositeAudioClip, AudioFileClip

class BGMAutoSelector:
    """BGM自動選択・同期システム"""
    
    def __init__(self, bgm_library_path: str = "bgm_library"):
        self.bgm_library_path = bgm_library_path
        self.setup_logging()
        self.load_bgm_database()
    
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_bgm_database(self):
        """BGMデータベース読み込み"""
        db_path = os.path.join(self.bgm_library_path, "bgm_database.json")
        
        if os.path.exists(db_path):
            with open(db_path, 'r', encoding='utf-8') as f:
                self.bgm_db = json.load(f)
        else:
            self.bgm_db = {}
            self.create_bgm_database()
    
    def create_bgm_database(self):
        """BGMデータベース作成"""
        self.logger.info("BGMデータベース作成開始")
        
        # デフォルトBGMカテゴリ
        default_categories = {
            "upbeat": {
                "description": "明るく活発な音楽",
                "tempo_range": [120, 180],
                "energy_level": "high",
                "mood": "positive"
            },
            "calm": {
                "description": "穏やかで落ち着いた音楽",
                "tempo_range": [60, 100],
                "energy_level": "low",
                "mood": "neutral"
            },
            "professional": {
                "description": "プロフェッショナルなビジネス向け音楽",
                "tempo_range": [90, 130],
                "energy_level": "medium",
                "mood": "focused"
            },
            "dramatic": {
                "description": "ドラマチックで印象的な音楽",
                "tempo_range": [70, 140],
                "energy_level": "high",
                "mood": "intense"
            },
            "ambient": {
                "description": "環境音楽・バックグラウンド向け",
                "tempo_range": [40, 80],
                "energy_level": "very_low",
                "mood": "atmospheric"
            }
        }
        
        self.bgm_db = {
            "categories": default_categories,
            "tracks": {}
        }
        
        # BGMライブラリをスキャン
        if os.path.exists(self.bgm_library_path):
            self.scan_bgm_library()
        
        self.save_bgm_database()
    
    def scan_bgm_library(self):
        """BGMライブラリスキャン"""
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac']
        
        for root, dirs, files in os.walk(self.bgm_library_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    file_path = os.path.join(root, file)
                    self.analyze_bgm_track(file_path)
    
    def analyze_bgm_track(self, file_path: str) -> Dict:
        """BGMトラック解析"""
        try:
            # 音声読み込み
            y, sr = librosa.load(file_path, duration=30)  # 最初の30秒を解析
            
            # 特徴量抽出
            features = self.extract_audio_features(y, sr)
            
            # カテゴリ分類
            category = self.classify_bgm_category(features)
            
            track_info = {
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "duration": librosa.get_duration(y=y, sr=sr),
                "category": category,
                "features": features,
                "analysis_date": "2024-01-01"  # 実際の実装では現在日時
            }
            
            # データベースに追加
            track_id = os.path.splitext(os.path.basename(file_path))[0]
            self.bgm_db["tracks"][track_id] = track_info
            
            self.logger.info(f"BGM解析完了: {file_path} -> {category}")
            return track_info
            
        except Exception as e:
            self.logger.error(f"BGM解析エラー {file_path}: {str(e)}")
            return {}
    
    def extract_audio_features(self, y: np.ndarray, sr: int) -> Dict:
        """音声特徴量抽出"""
        features = {}
        
        # テンポ分析
        tempo, _ = librosa.beat.tempo(y=y, sr=sr)
        features["tempo"] = float(tempo)
        
        # スペクトラル特徴
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid"] = float(np.mean(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spectral_rolloff"] = float(np.mean(spectral_rolloff))
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["mfcc"] = [float(np.mean(mfcc)) for mfcc in mfccs]
        
        # エネルギー分析
        energy = np.sum(y**2) / len(y)
        features["energy"] = float(energy)
        
        # ゼロ交差率
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zero_crossing_rate"] = float(np.mean(zcr))
        
        # クロマ特徴
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features["chroma"] = [float(np.mean(c)) for c in chroma]
        
        # 音の強さの変動
        rms = librosa.feature.rms(y=y)[0]
        features["rms_variation"] = float(np.std(rms))
        
        return features
    
    def classify_bgm_category(self, features: Dict) -> str:
        """BGMカテゴリ分類"""
        tempo = features.get("tempo", 120)
        energy = features.get("energy", 0.1)
        rms_variation = features.get("rms_variation", 0.1)
        
        # ルールベース分類
        if tempo > 140 and energy > 0.2:
            return "upbeat"
        elif tempo < 80 and energy < 0.1:
            return "ambient"
        elif 90 <= tempo <= 130 and 0.1 <= energy <= 0.3:
            return "professional"
        elif rms_variation > 0.15 and energy > 0.15:
            return "dramatic"
        else:
            return "calm"
    
    def analyze_content_mood(self, content_text: str) -> Dict:
        """コンテンツの雰囲気解析"""
        # キーワードベース雰囲気分析
        mood_keywords = {
            "energetic": ["効率", "革命", "劇的", "急速", "飛躍", "向上", "成功"],
            "professional": ["ビジネス", "システム", "分析", "戦略", "管理", "企業"],
            "educational": ["学習", "解説", "理解", "説明", "手順", "方法", "基礎"],
            "innovative": ["AI", "自動化", "最新", "技術", "革新", "次世代"],
            "calm": ["安定", "継続", "着実", "確実", "慎重", "丁寧"]
        }
        
        mood_scores = {}
        for mood, keywords in mood_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_text)
            mood_scores[mood] = score
        
        # 最も高いスコアの雰囲気を返す
        primary_mood = max(mood_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "primary_mood": primary_mood,
            "mood_scores": mood_scores,
            "content_length": len(content_text)
        }
    
    def select_optimal_bgm(
        self, 
        content_analysis: Dict,
        duration_target: float,
        exclude_tracks: List[str] = None
    ) -> Dict:
        """最適なBGM選択"""
        if exclude_tracks is None:
            exclude_tracks = []
        
        primary_mood = content_analysis.get("primary_mood", "professional")
        
        # 雰囲気からBGMカテゴリへのマッピング
        mood_to_category = {
            "energetic": "upbeat",
            "professional": "professional", 
            "educational": "calm",
            "innovative": "dramatic",
            "calm": "ambient"
        }
        
        target_category = mood_to_category.get(primary_mood, "professional")
        
        # カテゴリに該当するトラックを検索
        candidate_tracks = []
        for track_id, track_info in self.bgm_db["tracks"].items():
            if (track_info.get("category") == target_category and 
                track_id not in exclude_tracks):
                candidate_tracks.append((track_id, track_info))
        
        if not candidate_tracks:
            # フォールバック: 任意のカテゴリから選択
            candidate_tracks = [
                (track_id, track_info) 
                for track_id, track_info in self.bgm_db["tracks"].items()
                if track_id not in exclude_tracks
            ]
        
        if not candidate_tracks:
            self.logger.warning("利用可能なBGMが見つかりません")
            return {}
        
        # 長さに基づいて最適なトラックを選択
        best_track = min(
            candidate_tracks,
            key=lambda x: abs(x[1]["duration"] - duration_target)
        )
        
        selected_track = {
            "track_id": best_track[0],
            "info": best_track[1],
            "match_reason": f"カテゴリ: {target_category}, 雰囲気: {primary_mood}"
        }
        
        self.logger.info(f"選択されたBGM: {best_track[0]} ({target_category})")
        return selected_track
    
    def adjust_bgm_volume(
        self, 
        bgm_path: str,
        speech_analysis: Dict,
        output_path: str
    ) -> bool:
        """BGM音量自動調整"""
        try:
            # 音声分析からBGM音量を決定
            speech_energy = speech_analysis.get("energy", 0.1)
            speech_volume_variation = speech_analysis.get("rms_variation", 0.1)
            
            # 基本BGM音量（音声の20-30%）
            base_volume = 0.25
            
            # 音声エネルギーに応じて調整
            if speech_energy > 0.3:
                bgm_volume = base_volume * 0.7  # 音声が大きい場合はBGMを下げる
            elif speech_energy < 0.1:
                bgm_volume = base_volume * 1.3  # 音声が小さい場合はBGMを上げる
            else:
                bgm_volume = base_volume
            
            # MoviePyで音量調整
            bgm_clip = AudioFileClip(bgm_path)
            adjusted_bgm = bgm_clip.volumex(bgm_volume)
            adjusted_bgm.write_audiofile(output_path)
            
            bgm_clip.close()
            adjusted_bgm.close()
            
            self.logger.info(f"BGM音量調整完了: {bgm_volume:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"BGM音量調整エラー: {str(e)}")
            return False
    
    def create_layered_audio(
        self, 
        speech_path: str,
        bgm_path: str,
        output_path: str,
        crossfade_duration: float = 2.0
    ) -> bool:
        """レイヤード音声作成"""
        try:
            # 音声ファイル読み込み
            speech = AudioFileClip(speech_path)
            bgm = AudioFileClip(bgm_path)
            
            # BGMの長さを音声に合わせる
            if bgm.duration < speech.duration:
                # BGMをループ
                loops_needed = int(np.ceil(speech.duration / bgm.duration))
                bgm_extended = CompositeAudioClip([
                    bgm.set_start(i * bgm.duration) 
                    for i in range(loops_needed)
                ])
                bgm = bgm_extended.subclip(0, speech.duration)
            else:
                # BGMをカット
                bgm = bgm.subclip(0, speech.duration)
            
            # フェードイン/フェードアウト効果
            bgm = bgm.audio_fadein(crossfade_duration).audio_fadeout(crossfade_duration)
            
            # 音声とBGMを合成
            final_audio = CompositeAudioClip([
                speech,
                bgm.volumex(0.3)  # BGMの音量を下げる
            ])
            
            # 出力
            final_audio.write_audiofile(output_path)
            
            # リソース解放
            speech.close()
            bgm.close()
            final_audio.close()
            
            self.logger.info(f"レイヤード音声作成完了: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"レイヤード音声作成エラー: {str(e)}")
            return False
    
    def save_bgm_database(self):
        """BGMデータベース保存"""
        os.makedirs(self.bgm_library_path, exist_ok=True)
        db_path = os.path.join(self.bgm_library_path, "bgm_database.json")
        
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(self.bgm_db, f, ensure_ascii=False, indent=2)
    
    def full_bgm_process(
        self, 
        content_text: str,
        speech_audio_path: str,
        target_duration: float,
        output_path: str
    ) -> Dict:
        """完全BGM処理プロセス"""
        try:
            self.logger.info("完全BGM処理開始")
            
            # 1. コンテンツ雰囲気解析
            content_analysis = self.analyze_content_mood(content_text)
            
            # 2. 最適BGM選択
            selected_bgm = self.select_optimal_bgm(content_analysis, target_duration)
            
            if not selected_bgm:
                return {"success": False, "error": "適切なBGMが見つかりません"}
            
            # 3. 音声特徴解析
            speech_features = {}
            if os.path.exists(speech_audio_path):
                y, sr = librosa.load(speech_audio_path)
                speech_features = self.extract_audio_features(y, sr)
            
            # 4. BGM音量調整
            bgm_path = selected_bgm["info"]["file_path"]
            adjusted_bgm_path = output_path.replace(".wav", "_bgm_adjusted.wav")
            
            volume_success = self.adjust_bgm_volume(
                bgm_path, speech_features, adjusted_bgm_path
            )
            
            # 5. レイヤード音声作成
            if volume_success:
                layer_success = self.create_layered_audio(
                    speech_audio_path, adjusted_bgm_path, output_path
                )
            else:
                layer_success = False
            
            result = {
                "success": layer_success,
                "selected_bgm": selected_bgm,
                "content_analysis": content_analysis,
                "output_path": output_path if layer_success else None
            }
            
            self.logger.info("完全BGM処理完了")
            return result
            
        except Exception as e:
            self.logger.error(f"完全BGM処理エラー: {str(e)}")
            return {"success": False, "error": str(e)}

# 使用例
if __name__ == "__main__":
    bgm_selector = BGMAutoSelector()
    
    # サンプルコンテンツ
    content = """
    皆さん、こんにちは。今日はAI動画生成の革命的なシステムについて解説します。
    このシステムを使えば、動画制作の効率が劇的に向上し、
    誰でも簡単にプロ品質の動画を作成できるようになります。
    """
    
    result = bgm_selector.full_bgm_process(
        content_text=content,
        speech_audio_path="speech.wav",
        target_duration=120.0,
        output_path="final_audio_with_bgm.wav"
    )
    
    if result["success"]:
        print("BGM処理成功!")
        print(f"選択されたBGM: {result['selected_bgm']['track_id']}")
        print(f"検出された雰囲気: {result['content_analysis']['primary_mood']}")
    else:
        print(f"BGM処理失敗: {result.get('error', '不明なエラー')}")
