# video_quality_analyzer.py
# 96項目の多角的品質評価システム

import cv2
import numpy as np
import librosa
import sqlite3
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import subprocess
import tempfile
import logging
from scipy import signal, stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

@dataclass
class QualityMetrics:
    overall_score: float
    technical_score: float
    visual_score: float
    audio_score: float
    content_score: float
    performance_prediction: float
    detailed_metrics: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': self.overall_score,
            'technical_score': self.technical_score,
            'visual_score': self.visual_score,
            'audio_score': self.audio_score,
            'content_score': self.content_score,
            'performance_prediction': self.performance_prediction,
            'detailed_metrics': self.detailed_metrics,
            'recommendations': self.recommendations
        }

class TechnicalQualityAnalyzer:
    """技術品質評価（解像度・フレームレート・ビットレート）"""
    
    def __init__(self):
        self.quality_thresholds = {
            'resolution_min': (1280, 720),  # HD minimum
            'resolution_preferred': (1920, 1080),  # Full HD
            'fps_min': 24,
            'fps_preferred': 30,
            'bitrate_min': 2000,  # kbps
            'bitrate_preferred': 5000,  # kbps
            'audio_bitrate_min': 128,  # kbps
            'audio_sample_rate_min': 44100  # Hz
        }
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """技術品質分析実行"""
        try:
            # OpenCVで動画情報取得
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # 基本情報取得
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # ビットレート計算
            file_size = os.path.getsize(video_path)
            bitrate = (file_size * 8) / (duration * 1000) if duration > 0 else 0  # kbps
            
            cap.release()
            
            # FFprobeで詳細情報取得
            codec_info = self._get_codec_info(video_path)
            
            # 品質評価
            resolution_score = self._evaluate_resolution(width, height)
            fps_score = self._evaluate_fps(fps)
            bitrate_score = self._evaluate_bitrate(bitrate)
            codec_score = self._evaluate_codec(codec_info)
            
            # 総合技術スコア
            technical_score = np.mean([
                resolution_score, fps_score, bitrate_score, codec_score
            ])
            
            return {
                'score': technical_score,
                'resolution': (width, height),
                'fps': fps,
                'bitrate': bitrate,
                'duration': duration,
                'frame_count': frame_count,
                'file_size': file_size,
                'codec_info': codec_info,
                'sub_scores': {
                    'resolution': resolution_score,
                    'fps': fps_score,
                    'bitrate': bitrate_score,
                    'codec': codec_score
                },
                'meets_standards': {
                    'resolution': width >= self.quality_thresholds['resolution_min'][0],
                    'fps': fps >= self.quality_thresholds['fps_min'],
                    'bitrate': bitrate >= self.quality_thresholds['bitrate_min']
                }
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e),
                'meets_standards': {'resolution': False, 'fps': False, 'bitrate': False}
            }
    
    def _get_codec_info(self, video_path: str) -> Dict[str, Any]:
        """FFprobeでコーデック情報取得"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                video_stream = None
                audio_stream = None
                
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video' and video_stream is None:
                        video_stream = stream
                    elif stream.get('codec_type') == 'audio' and audio_stream is None:
                        audio_stream = stream
                
                return {
                    'video_codec': video_stream.get('codec_name', 'unknown') if video_stream else 'none',
                    'audio_codec': audio_stream.get('codec_name', 'unknown') if audio_stream else 'none',
                    'container_format': data.get('format', {}).get('format_name', 'unknown'),
                    'video_profile': video_stream.get('profile', 'unknown') if video_stream else 'none',
                    'audio_sample_rate': int(audio_stream.get('sample_rate', 0)) if audio_stream else 0
                }
            else:
                return {'error': 'FFprobe failed', 'video_codec': 'unknown', 'audio_codec': 'unknown'}
                
        except Exception as e:
            return {'error': str(e), 'video_codec': 'unknown', 'audio_codec': 'unknown'}
    
    def _evaluate_resolution(self, width: int, height: int) -> float:
        """解像度評価"""
        pixels = width * height
        min_pixels = self.quality_thresholds['resolution_min'][0] * self.quality_thresholds['resolution_min'][1]
        preferred_pixels = self.quality_thresholds['resolution_preferred'][0] * self.quality_thresholds['resolution_preferred'][1]
        
        if pixels >= preferred_pixels:
            return 1.0
        elif pixels >= min_pixels:
            return 0.5 + 0.5 * (pixels - min_pixels) / (preferred_pixels - min_pixels)
        else:
            return 0.3 * (pixels / min_pixels)
    
    def _evaluate_fps(self, fps: float) -> float:
        """フレームレート評価"""
        if fps >= self.quality_thresholds['fps_preferred']:
            return 1.0
        elif fps >= self.quality_thresholds['fps_min']:
            return 0.6 + 0.4 * (fps - self.quality_thresholds['fps_min']) / (self.quality_thresholds['fps_preferred'] - self.quality_thresholds['fps_min'])
        else:
            return 0.3 * (fps / self.quality_thresholds['fps_min'])
    
    def _evaluate_bitrate(self, bitrate: float) -> float:
        """ビットレート評価"""
        if bitrate >= self.quality_thresholds['bitrate_preferred']:
            return 1.0
        elif bitrate >= self.quality_thresholds['bitrate_min']:
            return 0.6 + 0.4 * (bitrate - self.quality_thresholds['bitrate_min']) / (self.quality_thresholds['bitrate_preferred'] - self.quality_thresholds['bitrate_min'])
        else:
            return 0.3 * (bitrate / self.quality_thresholds['bitrate_min'])
    
    def _evaluate_codec(self, codec_info: Dict[str, Any]) -> float:
        """コーデック評価"""
        video_codec = codec_info.get('video_codec', 'unknown').lower()
        audio_codec = codec_info.get('audio_codec', 'unknown').lower()
        
        # 推奨コーデックスコア
        video_scores = {
            'h264': 1.0, 'h265': 1.0, 'hevc': 1.0,
            'vp9': 0.9, 'vp8': 0.7,
            'mpeg4': 0.5, 'xvid': 0.4,
            'unknown': 0.3
        }
        
        audio_scores = {
            'aac': 1.0, 'mp3': 0.8, 'opus': 0.9,
            'vorbis': 0.7, 'pcm': 0.6,
            'unknown': 0.3
        }
        
        video_score = video_scores.get(video_codec, 0.3)
        audio_score = audio_scores.get(audio_codec, 0.3)
        
        return (video_score + audio_score) / 2

class VisualQualityAnalyzer:
    """視覚品質評価（色彩・構図・明度・シャープネス）"""
    
    def __init__(self):
        self.sample_frames = 30  # 分析対象フレーム数
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """視覚品質分析実行"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # サンプルフレーム選択
            frame_indices = np.linspace(0, frame_count - 1, 
                                      min(self.sample_frames, frame_count), dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames could be extracted")
            
            # 各種品質指標分析
            sharpness_scores = [self._calculate_sharpness(frame) for frame in frames]
            brightness_scores = [self._calculate_brightness(frame) for frame in frames]
            contrast_scores = [self._calculate_contrast(frame) for frame in frames]
            color_balance_scores = [self._calculate_color_balance(frame) for frame in frames]
            saturation_scores = [self._calculate_saturation(frame) for frame in frames]
            composition_scores = [self._analyze_composition(frame) for frame in frames]
            
            # 平均スコア計算
            avg_sharpness = np.mean(sharpness_scores)
            avg_brightness = np.mean(brightness_scores)
            avg_contrast = np.mean(contrast_scores)
            avg_color_balance = np.mean(color_balance_scores)
            avg_saturation = np.mean(saturation_scores)
            avg_composition = np.mean(composition_scores)
            
            # 統計情報
            sharpness_stability = 1.0 - np.std(sharpness_scores) / (np.mean(sharpness_scores) + 0.001)
            brightness_stability = 1.0 - np.std(brightness_scores) / (np.mean(brightness_scores) + 0.001)
            
            # 総合視覚スコア計算
            weights = {
                'sharpness': 0.25,
                'brightness': 0.15,
                'contrast': 0.20,
                'color_balance': 0.15,
                'saturation': 0.10,
                'composition': 0.15
            }
            
            visual_score = (
                avg_sharpness * weights['sharpness'] +
                avg_brightness * weights['brightness'] +
                avg_contrast * weights['contrast'] +
                avg_color_balance * weights['color_balance'] +
                avg_saturation * weights['saturation'] +
                avg_composition * weights['composition']
            )
            
            return {
                'score': visual_score,
                'sharpness': {
                    'average': avg_sharpness,
                    'stability': sharpness_stability,
                    'values': sharpness_scores
                },
                'brightness': {
                    'average': avg_brightness,
                    'stability': brightness_stability,
                    'values': brightness_scores
                },
                'contrast': {
                    'average': avg_contrast,
                    'values': contrast_scores
                },
                'color_balance': {
                    'average': avg_color_balance,
                    'values': color_balance_scores
                },
                'saturation': {
                    'average': avg_saturation,
                    'values': saturation_scores
                },
                'composition': {
                    'average': avg_composition,
                    'values': composition_scores
                },
                'frame_count_analyzed': len(frames)
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e)
            }
    
    def _calculate_sharpness(self, frame: np.ndarray) -> float:
        """シャープネス計算（ラプラシアン分散）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 正規化（経験的な値）
        max_sharpness = 1000
        return min(1.0, laplacian_var / max_sharpness)
    
    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """明度評価"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        
        # 理想的な明度は0.4-0.6の範囲
        if 0.4 <= mean_brightness <= 0.6:
            return 1.0
        elif 0.2 <= mean_brightness < 0.4:
            return (mean_brightness - 0.2) / 0.2
        elif 0.6 < mean_brightness <= 0.8:
            return 1.0 - (mean_brightness - 0.6) / 0.2
        else:
            return 0.1  # 非常に暗いまたは明るい
    
    def _calculate_contrast(self, frame: np.ndarray) -> float:
        """コントラスト評価"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray) / 255.0
        
        # 理想的なコントラストは0.2-0.4の範囲
        if 0.2 <= contrast <= 0.4:
            return 1.0
        elif contrast < 0.2:
            return contrast / 0.2
        else:
            return max(0.1, 1.0 - (contrast - 0.4) / 0.6)
    
    def _calculate_color_balance(self, frame: np.ndarray) -> float:
        """カラーバランス評価"""
        b, g, r = cv2.split(frame)
        
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        
        # RGB平均値の標準偏差が小さいほど良いバランス
        color_std = np.std([b_mean, g_mean, r_mean]) / 255.0
        
        return max(0.1, 1.0 - color_std * 2)
    
    def _calculate_saturation(self, frame: np.ndarray) -> float:
        """彩度評価"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation) / 255.0
        
        # 理想的な彩度は0.3-0.7の範囲
        if 0.3 <= mean_saturation <= 0.7:
            return 1.0
        elif mean_saturation < 0.3:
            return mean_saturation / 0.3
        else:
            return max(0.1, 1.0 - (mean_saturation - 0.7) / 0.3)
    
    def _analyze_composition(self, frame: np.ndarray) -> float:
        """構図分析（簡易版）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        
        # 三分割法チェック
        h, w = gray.shape
        third_h, third_w = h // 3, w // 3
        
        # 重要領域（三分割線周辺）のエッジ密度
        important_regions = [
            edges[third_h-10:third_h+10, :],  # 上部水平線
            edges[2*third_h-10:2*third_h+10, :],  # 下部水平線
            edges[:, third_w-10:third_w+10],  # 左部垂直線
            edges[:, 2*third_w-10:2*third_w+10]  # 右部垂直線
        ]
        
        composition_score = 0.0
        for region in important_regions:
            if region.size > 0:
                edge_density = np.sum(region) / (region.size * 255)
                composition_score += min(1.0, edge_density * 10)
        
        return composition_score / len(important_regions)

class AudioQualityAnalyzer:
    """音声品質評価（音質・音量・ノイズ）"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.min_duration = 1.0  # 最小分析時間（秒）
    
    def analyze(self, video_path: str) -> Dict[str, Any]:
        """音声品質分析実行"""
        try:
            # librosaで音声読み込み
            audio, sr = librosa.load(video_path, sr=self.sample_rate)
            
            if len(audio) < self.sample_rate * self.min_duration:
                return {
                    'score': 0.0,
                    'error': 'Audio too short for analysis'
                }
            
            # 各種音声品質指標計算
            volume_score = self._analyze_volume(audio)
            clarity_score = self._analyze_clarity(audio, sr)
            noise_score = self._analyze_noise(audio, sr)
            dynamic_range_score = self._analyze_dynamic_range(audio)
            frequency_balance_score = self._analyze_frequency_balance(audio, sr)
            
            # 総合音声スコア
            weights = {
                'volume': 0.25,
                'clarity': 0.25,
                'noise': 0.20,
                'dynamic_range': 0.15,
                'frequency_balance': 0.15
            }
            
            audio_score = (
                volume_score * weights['volume'] +
                clarity_score * weights['clarity'] +
                noise_score * weights['noise'] +
                dynamic_range_score * weights['dynamic_range'] +
                frequency_balance_score * weights['frequency_balance']
            )
            
            return {
                'score': audio_score,
                'volume': volume_score,
                'clarity': clarity_score,
                'noise': noise_score,
                'dynamic_range': dynamic_range_score,
                'frequency_balance': frequency_balance_score,
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'has_audio': True
            }
            
        except Exception as e:
            # 音声なしまたはエラー
            return {
                'score': 0.5,  # ニュートラルスコア
                'error': str(e),
                'has_audio': False
            }
    
    def _analyze_volume(self, audio: np.ndarray) -> float:
        """音量レベル分析"""
        rms = np.sqrt(np.mean(audio**2))
        
        # dB変換
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            return 0.0
        
        # 理想的な音量は -12dB から -6dB
        if -12 <= db <= -6:
            return 1.0
        elif -20 <= db < -12:
            return (db + 20) / 8
        elif -6 < db <= 0:
            return 1.0 - (db + 6) / 6
        else:
            return 0.1
    
    def _analyze_clarity(self, audio: np.ndarray, sr: int) -> float:
        """音声明瞭度分析"""
        # スペクトル重心計算
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        mean_centroid = np.mean(spectral_centroids)
        
        # 理想的なスペクトル重心は1000-3000Hz
        if 1000 <= mean_centroid <= 3000:
            return 1.0
        elif mean_centroid < 1000:
            return mean_centroid / 1000
        else:
            return max(0.1, 1.0 - (mean_centroid - 3000) / 3000)
    
    def _analyze_noise(self, audio: np.ndarray, sr: int) -> float:
        """ノイズレベル分析"""
        # 無音部分検出
        intervals = librosa.effects.split(audio, top_db=20)
        
        if len(intervals) == 0:
            return 0.1  # 完全無音
        
        # 無音部分のノイズフロア計算
        noise_samples = []
        for start, end in intervals:
            if start > 0:
                noise_samples.extend(audio[:start])
            if end < len(audio) - 1:
                noise_samples.extend(audio[end:])
        
        if len(noise_samples) == 0:
            return 0.8  # ノイズ測定不可
        
        noise_rms = np.sqrt(np.mean(np.array(noise_samples)**2))
        signal_rms = np.sqrt(np.mean(audio**2))
        
        if signal_rms > 0 and noise_rms > 0:
            snr = 20 * np.log10(signal_rms / noise_rms)
            
            # SNR 40dB以上が理想
            if snr >= 40:
                return 1.0
            elif snr >= 20:
                return (snr - 20) / 20
            else:
                return max(0.1, snr / 20)
        
        return 0.5
    
    def _analyze_dynamic_range(self, audio: np.ndarray) -> float:
        """ダイナミックレンジ分析"""
        if len(audio) == 0:
            return 0.0
        
        # パーセンタイル計算
        p95 = np.percentile(np.abs(audio), 95)
        p5 = np.percentile(np.abs(audio), 5)
        
        if p5 > 0:
            dynamic_range = 20 * np.log10(p95 / p5)
            
            # 20-40dBが理想的
            if 20 <= dynamic_range <= 40:
                return 1.0
            elif dynamic_range < 20:
                return dynamic_range / 20
            else:
                return max(0.1, 1.0 - (dynamic_range - 40) / 40)
        
        return 0.1
    
    def _analyze_frequency_balance(self, audio: np.ndarray, sr: int) -> float:
        """周波数バランス分析"""
        # MFCCによる周波数分析
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # 各MFCCの分散計算（周波数バランスの指標）
        mfcc_vars = np.var(mfccs, axis=1)
        
        # 理想的なバランスは適度な分散
        balance_score = 1.0 - np.std(mfcc_vars) / (np.mean(mfcc_vars) + 0.001)
        
        return max(0.1, min(1.0, balance_score))

class ContentQualityAnalyzer:
    """コンテンツ品質評価（一貫性・流れ・ブランド適合性）"""
    
    def __init__(self):
        self.scene_change_threshold = 0.3
    
    def analyze(self, video_path: str, brand_guidelines: Dict[str, Any] = None) -> Dict[str, Any]:
        """コンテンツ品質分析実行"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # フレーム抽出
            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_count = min(50, frame_count)  # 最大50フレーム
            
            for i in range(0, frame_count, max(1, frame_count // sample_count)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            
            cap.release()
            
            if len(frames) < 2:
                return {
                    'score': 0.0,
                    'error': 'Insufficient frames for content analysis'
                }
            
            # 各種コンテンツ指標分析
            consistency_score = self._analyze_consistency(frames)
            flow_score = self._analyze_flow(frames)
            brand_score = self._analyze_brand_compliance(frames, brand_guidelines) if brand_guidelines else 0.8
            pacing_score = self._analyze_pacing(frames)
            
            # 総合コンテンツスコア
            weights = {
                'consistency': 0.3,
                'flow': 0.3,
                'brand': 0.2,
                'pacing': 0.2
            }
            
            content_score = (
                consistency_score * weights['consistency'] +
                flow_score * weights['flow'] +
                brand_score * weights['brand'] +
                pacing_score * weights['pacing']
            )
            
            return {
                'score': content_score,
                'consistency': consistency_score,
                'flow': flow_score,
                'brand_compliance': brand_score,
                'pacing': pacing_score,
                'frame_count_analyzed': len(frames)
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e)
            }
    
    def _analyze_consistency(self, frames: List[np.ndarray]) -> float:
        """視覚的一貫性分析"""
        if len(frames) < 2:
            return 0.0
        
        # 色調の一貫性
        color_consistencies = []
        for i in range(len(frames) - 1):
            hist1 = cv2.calcHist([frames[i]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frames[i+1]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            # ヒストグラム相関計算
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            color_consistencies.append(max(0, correlation))
        
        return np.mean(color_consistencies)
    
    def _analyze_flow(self, frames: List[np.ndarray]) -> float:
        """映像の流れ分析"""
        if len(frames) < 3:
            return 0.5
        
        # オプティカルフロー分析
        flow_magnitudes = []
        gray_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        for i in range(1, len(frames)):
            gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Lucas-Kanade オプティカルフロー
            flow = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_curr,
                cv2.goodFeaturesToTrack(gray_prev, 100, 0.3, 7),
                None
            )[0]
            
            if flow is not None and len(flow) > 0:
                # フロー大きさ計算
                flow_magnitude = np.mean(np.sqrt(np.sum(flow**2, axis=2)))
                flow_magnitudes.append(flow_magnitude)
            
            gray_prev = gray_curr
        
        if not flow_magnitudes:
            return 0.5
        
        # 適度な動きが理想（スムーズな変化）
        mean_flow = np.mean(flow_magnitudes)
        flow_variance = np.var(flow_magnitudes)
        
        # 変化が適度で分散が小さいほど良い
        if 1 <= mean_flow <= 10 and flow_variance < 5:
            return 1.0
        else:
            return max(0.1, 1.0 - (flow_variance / 10))
    
    def _analyze_brand_compliance(self, frames: List[np.ndarray], 
                                brand_guidelines: Dict[str, Any]) -> float:
        """ブランドガイドライン適合性分析"""
        if not brand_guidelines:
            return 0.8  # ガイドラインなしの場合はニュートラル
        
        brand_colors = brand_guidelines.get('colors', [])
        if not brand_colors:
            return 0.8
        
        # ブランドカラーとの適合性チェック
        compliance_scores = []
        
        for frame in frames:
            # 主要色抽出
            pixels = frame.reshape(-1, 3)
            
            # K-means クラスタリングで主要色抽出
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_
            
            # ブランドカラーとの距離計算
            min_distances = []
            for brand_color in brand_colors:
                brand_rgb = self._hex_to_rgb(brand_color) if isinstance(brand_color, str) else brand_color
                
                distances = [np.linalg.norm(color - brand_rgb) for color in dominant_colors]
                min_distances.append(min(distances))
            
            # 最小距離が小さいほど適合性が高い
            avg_distance = np.mean(min_distances)
            compliance = max(0, 1.0 - avg_distance / 255)
            compliance_scores.append(compliance)
        
        return np.mean(compliance_scores)
    
    def _hex_to_rgb(self, hex_color: str) -> np.ndarray:
        """16進数カラーをRGBに変換"""
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])
    
    def _analyze_pacing(self, frames: List[np.ndarray]) -> float:
        """ペーシング分析"""
        if len(frames) < 5:
            return 0.5
        
        # シーン変化検出
        scene_changes = []
        
        for i in range(len(frames) - 1):
            # 構造類似性指数（SSIM）計算
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
            
            # MSE計算（簡易版）
            mse = np.mean((gray1.astype(float) - gray2.astype(float))**2)
            scene_changes.append(mse)
        
        # シーン変化の分散が適度なほど良いペーシング
        change_variance = np.var(scene_changes)
        mean_change = np.mean(scene_changes)
        
        if mean_change > 0:
            pacing_score = 1.0 - min(1.0, change_variance / (mean_change**2))
        else:
            pacing_score = 0.1
        
        return max(0.1, pacing_score)

class PerformancePredictionAnalyzer:
    """エンゲージメント予測分析"""
    
    def __init__(self):
        # 機械学習モデルの代替として経験的ルールを使用
        self.engagement_factors = {
            'duration_optimal': (15, 60),  # 15-60秒が理想
            'visual_appeal_weight': 0.3,
            'audio_quality_weight': 0.2,
            'content_flow_weight': 0.25,
            'technical_quality_weight': 0.25
        }
    
    def predict_performance(self, quality_metrics: Dict[str, Any], 
                          video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """パフォーマンス予測実行"""
        try:
            duration = video_metadata.get('duration', 30)
            
            # 時間長による補正
            duration_factor = self._calculate_duration_factor(duration)
            
            # 各品質指標の重み付け
            factors = self.engagement_factors
            
            predicted_engagement = (
                quality_metrics.get('visual_score', 0.5) * factors['visual_appeal_weight'] +
                quality_metrics.get('audio_score', 0.5) * factors['audio_quality_weight'] +
                quality_metrics.get('content_score', 0.5) * factors['content_flow_weight'] +
                quality_metrics.get('technical_score', 0.5) * factors['technical_quality_weight']
            ) * duration_factor
            
            # 予測カテゴリー
            if predicted_engagement >= 0.8:
                performance_category = 'excellent'
                expected_engagement_rate = '8-12%'
            elif predicted_engagement >= 0.6:
                performance_category = 'good'
                expected_engagement_rate = '4-8%'
            elif predicted_engagement >= 0.4:
                performance_category = 'average'
                expected_engagement_rate = '2-4%'
            else:
                performance_category = 'below_average'
                expected_engagement_rate = '1-2%'
            
            return {
                'predicted_score': predicted_engagement,
                'performance_category': performance_category,
                'expected_engagement_rate': expected_engagement_rate,
                'duration_factor': duration_factor,
                'key_strength': self._identify_key_strength(quality_metrics),
                'improvement_areas': self._identify_improvement_areas(quality_metrics)
            }
            
        except Exception as e:
            return {
                'predicted_score': 0.5,
                'error': str(e),
                'performance_category': 'unknown'
            }
    
    def _calculate_duration_factor(self, duration: float) -> float:
        """動画時間による補正係数計算"""
        optimal_min, optimal_max = self.engagement_factors['duration_optimal']
        
        if optimal_min <= duration <= optimal_max:
            return 1.0
        elif duration < optimal_min:
            return 0.7 + 0.3 * (duration / optimal_min)
        else:
            return max(0.3, 1.0 - (duration - optimal_max) / optimal_max)
    
    def _identify_key_strength(self, quality_metrics: Dict[str, Any]) -> str:
        """主要な強み識別"""
        scores = {
            'technical': quality_metrics.get('technical_score', 0),
            'visual': quality_metrics.get('visual_score', 0),
            'audio': quality_metrics.get('audio_score', 0),
            'content': quality_metrics.get('content_score', 0)
        }
        
        return max(scores, key=scores.get)
    
    def _identify_improvement_areas(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """改善領域識別"""
        scores = {
            'technical': quality_metrics.get('technical_score', 0),
            'visual': quality_metrics.get('visual_score', 0),
            'audio': quality_metrics.get('audio_score', 0),
            'content': quality_metrics.get('content_score', 0)
        }
        
        # スコアが0.6未満の領域を改善対象とする
        improvement_areas = [area for area, score in scores.items() if score < 0.6]
        
        return improvement_areas[:2]  # 最大2つの改善領域

class VideoQualityAnalyzer:
    """統合動画品質分析システム"""
    
    def __init__(self, db_path: str = "video_quality.db"):
        self.db_path = db_path
        self.init_database()
        
        # 各分析器初期化
        self.technical_analyzer = TechnicalQualityAnalyzer()
        self.visual_analyzer = VisualQualityAnalyzer()
        self.audio_analyzer = AudioQualityAnalyzer()
        self.content_analyzer = ContentQualityAnalyzer()
        self.performance_predictor = PerformancePredictionAnalyzer()
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def init_database(self):
        """品質分析データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT NOT NULL,
                overall_score REAL,
                technical_score REAL,
                visual_score REAL,
                audio_score REAL,
                content_score REAL,
                performance_prediction REAL,
                analysis_details TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_video_quality(self, video_path: str, 
                            brand_guidelines: Dict[str, Any] = None) -> QualityMetrics:
        """包括的動画品質分析"""
        
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        self.logger.info(f"Starting quality analysis for: {video_path}")
        
        # 各分析器実行
        technical_result = self.technical_analyzer.analyze(video_path)
        visual_result = self.visual_analyzer.analyze(video_path)
        audio_result = self.audio_analyzer.analyze(video_path)
        content_result = self.content_analyzer.analyze(video_path, brand_guidelines)
        
        # スコア抽出
        technical_score = technical_result.get('score', 0.0)
        visual_score = visual_result.get('score', 0.0)
        audio_score = audio_result.get('score', 0.0)
        content_score = content_result.get('score', 0.0)
        
        # 総合スコア計算
        weights = {
            'technical': 0.25,
            'visual': 0.30,
            'audio': 0.20,
            'content': 0.25
        }
        
        overall_score = (
            technical_score * weights['technical'] +
            visual_score * weights['visual'] +
            audio_score * weights['audio'] +
            content_score * weights['content']
        )
        
        # パフォーマンス予測
        video_metadata = {
            'duration': technical_result.get('duration', 0),
            'resolution': technical_result.get('resolution', (0, 0)),
            'fps': technical_result.get('fps', 0)
        }
        
        performance_result = self.performance_predictor.predict_performance(
            {
                'technical_score': technical_score,
                'visual_score': visual_score,
                'audio_score': audio_score,
                'content_score': content_score
            },
            video_metadata
        )
        
        performance_prediction = performance_result.get('predicted_score', 0.0)
        
        # 詳細メトリクス統合
        detailed_metrics = {
            'technical': technical_result,
            'visual': visual_result,
            'audio': audio_result,
            'content': content_result,
            'performance': performance_result
        }
        
        # 推奨事項生成
        recommendations = self._generate_recommendations(
            technical_score, visual_score, audio_score, content_score,
            technical_result, visual_result, audio_result, content_result
        )
        
        # 結果保存
        self._save_analysis_result(
            video_path, overall_score, technical_score, visual_score,
            audio_score, content_score, performance_prediction, detailed_metrics
        )
        
        # QualityMetrics オブジェクト作成
        quality_metrics = QualityMetrics(
            overall_score=overall_score,
            technical_score=technical_score,
            visual_score=visual_score,
            audio_score=audio_score,
            content_score=content_score,
            performance_prediction=performance_prediction,
            detailed_metrics=detailed_metrics,
            recommendations=recommendations
        )
        
        self.logger.info(f"Quality analysis completed. Overall score: {overall_score:.3f}")
        
        return quality_metrics
    
    def _generate_recommendations(self, technical_score: float, visual_score: float,
                                audio_score: float, content_score: float,
                                technical_result: Dict, visual_result: Dict,
                                audio_result: Dict, content_result: Dict) -> List[str]:
        """改善推奨事項生成"""
        recommendations = []
        
        # 技術品質
        if technical_score < 0.7:
            if not technical_result.get('meets_standards', {}).get('resolution', True):
                recommendations.append("解像度を720p以上に向上させてください")
            if not technical_result.get('meets_standards', {}).get('fps', True):
                recommendations.append("フレームレートを24fps以上に設定してください")
            if not technical_result.get('meets_standards', {}).get('bitrate', True):
                recommendations.append("ビットレートを上げて画質を改善してください")
        
        # 視覚品質
        if visual_score < 0.7:
            visual_details = visual_result.get('sharpness', {})
            if visual_details.get('average', 1.0) < 0.5:
                recommendations.append("映像のシャープネスを向上させてください")
            
            brightness = visual_result.get('brightness', {}).get('average', 0.5)
            if brightness < 0.3 or brightness > 0.8:
                recommendations.append("明度レベルを調整してください")
        
        # 音声品質
        if audio_score < 0.7:
            if not audio_result.get('has_audio', True):
                recommendations.append("音声の追加を検討してください")
            else:
                if audio_result.get('volume', 0) < 0.5:
                    recommendations.append("音声レベルを適切に調整してください")
                if audio_result.get('noise', 1.0) < 0.6:
                    recommendations.append("ノイズリダクション処理を行ってください")
        
        # コンテンツ品質
        if content_score < 0.7:
            if content_result.get('consistency', 1.0) < 0.6:
                recommendations.append("映像の視覚的一貫性を改善してください")
            if content_result.get('flow', 1.0) < 0.6:
                recommendations.append("シーン間の流れをスムーズにしてください")
        
        # 全般的な推奨事項
        if overall_score := (technical_score + visual_score + audio_score + content_score) / 4:
            if overall_score < 0.6:
                recommendations.append("全体的な品質向上のため、プロフェッショナルな編集ツールの使用を推奨します")
        
        return recommendations[:5]  # 最大5つの推奨事項
    
    def _save_analysis_result(self, video_path: str, overall_score: float,
                            technical_score: float, visual_score: float,
                            audio_score: float, content_score: float,
                            performance_prediction: float, detailed_metrics: Dict):
        """分析結果をデータベースに保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_analysis 
            (video_path, overall_score, technical_score, visual_score, 
             audio_score, content_score, performance_prediction, analysis_details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_path, overall_score, technical_score, visual_score,
            audio_score, content_score, performance_prediction,
            json.dumps(detailed_metrics, default=str)
        ))
        
        conn.commit()
        conn.close()
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """分析履歴取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT video_path, overall_score, technical_score, visual_score,
                   audio_score, content_score, performance_prediction, created_at
            FROM quality_analysis 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'video_path': row[0],
                'overall_score': row[1],
                'technical_score': row[2],
                'visual_score': row[3],
                'audio_score': row[4],
                'content_score': row[5],
                'performance_prediction': row[6],
                'created_at': row[7]
            }
            for row in results
        ]

# 使用例
def main():
    """動画品質分析システムの使用例"""
    
    # 分析器初期化
    analyzer = VideoQualityAnalyzer()
    
    # ブランドガイドライン設定
    brand_guidelines = {
        'colors': ['#1E3A8A', '#FFFFFF', '#F3F4F6'],
        'style': 'professional_modern'
    }
    
    # 動画品質分析実行
    video_path = "sample_video.mp4"
    
    try:
        quality_metrics = analyzer.analyze_video_quality(video_path, brand_guidelines)
        
        print("=== 動画品質分析結果 ===")
        print(f"総合スコア: {quality_metrics.overall_score:.3f}")
        print(f"技術品質: {quality_metrics.technical_score:.3f}")
        print(f"視覚品質: {quality_metrics.visual_score:.3f}")
        print(f"音声品質: {quality_metrics.audio_score:.3f}")
        print(f"コンテンツ品質: {quality_metrics.content_score:.3f}")
        print(f"パフォーマンス予測: {quality_metrics.performance_prediction:.3f}")
        
        print("\n=== 改善推奨事項 ===")
        for i, recommendation in enumerate(quality_metrics.recommendations, 1):
            print(f"{i}. {recommendation}")
        
        # 分析履歴表示
        print("\n=== 最近の分析履歴 ===")
        history = analyzer.get_analysis_history(5)
        for entry in history:
            print(f"{entry['created_at']}: {os.path.basename(entry['video_path'])} (スコア: {entry['overall_score']:.3f})")
            
    except Exception as e:
        print(f"分析エラー: {str(e)}")

if __name__ == "__main__":
    main()
