# audio_video_sync.py
import cv2
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, CompositeAudioClip
import matplotlib.pyplot as plt
from scipy import signal
from typing import List, Tuple, Dict
import logging

class AudioVideoSyncSystem:
    """音声と動画の完全同期システム"""
    
    def __init__(self):
        self.setup_logging()
        self.sync_tolerance = 0.05  # 50ms以内の同期誤差を許容
    
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_audio_features(self, audio_path: str) -> Dict:
        """音声特徴量解析"""
        try:
            # 音声読み込み
            y, sr = librosa.load(audio_path)
            
            # 特徴量抽出
            features = {
                "duration": librosa.get_duration(y=y, sr=sr),
                "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
                "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
                "mfcc": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1),
                "onset_frames": librosa.onset.onset_detect(y=y, sr=sr),
                "beat_frames": librosa.beat.beat_track(y=y, sr=sr)[1]
            }
            
            # 音声エネルギー分析
            hop_length = 512
            frame_length = 2048
            energy = np.array([
                sum(abs(y[i:i+frame_length]**2))
                for i in range(0, len(y), hop_length)
            ])
            
            features["energy_profile"] = energy
            features["energy_peaks"] = signal.find_peaks(energy, height=np.max(energy)*0.3)[0]
            
            self.logger.info(f"音声解析完了: {audio_path}")
            return features
            
        except Exception as e:
            self.logger.error(f"音声解析エラー: {str(e)}")
            return {}
    
    def analyze_video_features(self, video_path: str) -> Dict:
        """動画特徴量解析"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # 基本情報取得
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # フレーム解析
            frame_features = []
            motion_scores = []
            brightness_scores = []
            
            ret, prev_frame = cap.read()
            if not ret:
                raise Exception("動画読み込みエラー")
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # モーション検出
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, None, None
                )[0]
                motion_score = np.mean(np.abs(flow)) if flow is not None else 0
                motion_scores.append(motion_score)
                
                # 明度分析
                brightness = np.mean(gray)
                brightness_scores.append(brightness)
                
                # シーン変化検出
                diff = cv2.absdiff(prev_gray, gray)
                scene_change = np.mean(diff) > 30
                
                frame_features.append({
                    "motion": motion_score,
                    "brightness": brightness,
                    "scene_change": scene_change
                })
                
                prev_gray = gray
            
            cap.release()
            
            features = {
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
                "motion_profile": motion_scores,
                "brightness_profile": brightness_scores,
                "motion_peaks": signal.find_peaks(motion_scores, height=np.max(motion_scores)*0.5)[0],
                "scene_changes": [i for i, f in enumerate(frame_features) if f["scene_change"]]
            }
            
            self.logger.info(f"動画解析完了: {video_path}")
            return features
            
        except Exception as e:
            self.logger.error(f"動画解析エラー: {str(e)}")
            return {}
    
    def detect_sync_points(
        self, 
        audio_features: Dict, 
        video_features: Dict
    ) -> List[Tuple[float, float]]:
        """同期ポイント検出"""
        sync_points = []
        
        try:
            # 音声のエネルギーピークと動画のモーションピークを対応付け
            audio_peaks = audio_features.get("energy_peaks", [])
            video_peaks = video_features.get("motion_peaks", [])
            
            # フレームレート調整
            fps = video_features.get("fps", 30)
            hop_length = 512
            sr = 22050  # librosaのデフォルト
            
            # 音声ピークを時間に変換
            audio_peak_times = [peak * hop_length / sr for peak in audio_peaks]
            
            # 動画ピークを時間に変換
            video_peak_times = [peak / fps for peak in video_peaks]
            
            # 最も近いピーク同士をマッチング
            for audio_time in audio_peak_times:
                closest_video_time = min(
                    video_peak_times,
                    key=lambda x: abs(x - audio_time)
                )
                
                # 同期誤差が許容範囲内の場合のみ追加
                if abs(audio_time - closest_video_time) <= self.sync_tolerance:
                    sync_points.append((audio_time, closest_video_time))
            
            # シーン変化点も同期ポイントとして考慮
            scene_changes = video_features.get("scene_changes", [])
            scene_change_times = [change / fps for change in scene_changes]
            
            for scene_time in scene_change_times:
                # 近くに音声ピークがあるかチェック
                closest_audio = min(
                    audio_peak_times,
                    key=lambda x: abs(x - scene_time),
                    default=None
                )
                
                if closest_audio and abs(closest_audio - scene_time) <= self.sync_tolerance * 2:
                    sync_points.append((closest_audio, scene_time))
            
            # 重複除去とソート
            sync_points = list(set(sync_points))
            sync_points.sort(key=lambda x: x[0])
            
            self.logger.info(f"検出された同期ポイント数: {len(sync_points)}")
            return sync_points
            
        except Exception as e:
            self.logger.error(f"同期ポイント検出エラー: {str(e)}")
            return []
    
    def calculate_sync_offset(self, sync_points: List[Tuple[float, float]]) -> float:
        """同期オフセット計算"""
        if not sync_points:
            return 0.0
        
        # 各同期ポイントでのオフセットを計算
        offsets = [video_time - audio_time for audio_time, video_time in sync_points]
        
        # 中央値を使用（外れ値に強い）
        offset = np.median(offsets)
        
        self.logger.info(f"計算された同期オフセット: {offset:.3f}秒")
        return offset
    
    def apply_sync_correction(
        self, 
        video_path: str, 
        audio_path: str, 
        output_path: str,
        sync_offset: float
    ) -> bool:
        """同期補正適用"""
        try:
            # 動画読み込み
            video = VideoFileClip(video_path)
            
            # 音声読み込み
            from moviepy.editor import AudioFileClip
            audio = AudioFileClip(audio_path)
            
            # オフセット適用
            if sync_offset > 0:
                # 音声を遅らせる
                audio = audio.set_start(sync_offset)
            elif sync_offset < 0:
                # 音声を早める（動画を遅らせる）
                video = video.set_start(-sync_offset)
            
            # 音声と動画を合成
            final_video = video.set_audio(audio)
            
            # 出力
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # リソース解放
            video.close()
            audio.close()
            final_video.close()
            
            self.logger.info(f"同期補正完了: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"同期補正エラー: {str(e)}")
            return False
    
    def create_sync_analysis_report(
        self, 
        audio_features: Dict, 
        video_features: Dict,
        sync_points: List[Tuple[float, float]],
        sync_offset: float
    ) -> str:
        """同期解析レポート作成"""
        report = f"""
# 音声・動画同期解析レポート

## 基本情報
- 音声長: {audio_features.get('duration', 0):.2f}秒
- 動画長: {video_features.get('duration', 0):.2f}秒
- フレームレート: {video_features.get('fps', 0):.2f}fps
- 検出された同期ポイント: {len(sync_points)}個
- 計算された同期オフセット: {sync_offset:.3f}秒

## 音声特徴
- テンポ: {audio_features.get('tempo', 0):.2f} BPM
- スペクトル重心: {audio_features.get('spectral_centroid', 0):.2f} Hz
- ゼロ交差率: {audio_features.get('zero_crossing_rate', 0):.4f}

## 動画特徴
- 平均モーション: {np.mean(video_features.get('motion_profile', [0])):.2f}
- 平均明度: {np.mean(video_features.get('brightness_profile', [0])):.2f}
- シーン変化数: {len(video_features.get('scene_changes', []))}個

## 同期品質評価
"""
        
        if len(sync_points) >= 3:
            report += "- 同期品質: 良好 ✓\n"
        elif len(sync_points) >= 1:
            report += "- 同期品質: 普通 ⚠\n"
        else:
            report += "- 同期品質: 要改善 ❌\n"
        
        if abs(sync_offset) <= self.sync_tolerance:
            report += "- オフセット: 許容範囲内 ✓\n"
        elif abs(sync_offset) <= self.sync_tolerance * 2:
            report += "- オフセット: 軽微な調整が必要 ⚠\n"
        else:
            report += "- オフセット: 大幅な調整が必要 ❌\n"
        
        return report
    
    def full_sync_process(
        self, 
        video_path: str, 
        audio_path: str, 
        output_path: str
    ) -> Dict:
        """完全同期処理"""
        try:
            self.logger.info("完全同期処理開始")
            
            # 1. 特徴量解析
            audio_features = self.analyze_audio_features(audio_path)
            video_features = self.analyze_video_features(video_path)
            
            # 2. 同期ポイント検出
            sync_points = self.detect_sync_points(audio_features, video_features)
            
            # 3. オフセット計算
            sync_offset = self.calculate_sync_offset(sync_points)
            
            # 4. 同期補正適用
            success = self.apply_sync_correction(
                video_path, audio_path, output_path, sync_offset
            )
            
            # 5. 解析レポート作成
            report = self.create_sync_analysis_report(
                audio_features, video_features, sync_points, sync_offset
            )
            
            result = {
                "success": success,
                "sync_offset": sync_offset,
                "sync_points_count": len(sync_points),
                "report": report,
                "output_path": output_path if success else None
            }
            
            self.logger.info("完全同期処理完了")
            return result
            
        except Exception as e:
            self.logger.error(f"完全同期処理エラー: {str(e)}")
            return {"success": False, "error": str(e)}

# 使用例
if __name__ == "__main__":
    sync_system = AudioVideoSyncSystem()
    
    result = sync_system.full_sync_process(
        video_path="input_video.mp4",
        audio_path="generated_audio.mp3",
        output_path="synced_output.mp4"
    )
    
    if result["success"]:
        print("同期処理成功!")
        print(f"同期オフセット: {result['sync_offset']:.3f}秒")
        print(result["report"])
    else:
        print(f"同期処理失敗: {result.get('error', '不明なエラー')}")
