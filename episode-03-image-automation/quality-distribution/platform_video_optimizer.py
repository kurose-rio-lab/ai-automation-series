# platform_video_optimizer.py
# YouTube/Instagram/TikTok/Twitter/LinkedIn/Facebook対応の自動動画最適化システム

import ffmpeg
import os
import json
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import hashlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class Platform(Enum):
    YOUTUBE = "youtube"
    INSTAGRAM_FEED = "instagram_feed"
    INSTAGRAM_STORY = "instagram_story"
    INSTAGRAM_REEL = "instagram_reel"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    FACEBOOK_STORY = "facebook_story"

@dataclass
class PlatformSpec:
    resolution: Tuple[int, int]
    aspect_ratio: str
    max_duration: int  # seconds
    min_duration: int  # seconds
    max_file_size: int  # MB
    video_codec: str
    audio_codec: str
    fps: int
    bitrate_video: str  # kbps
    bitrate_audio: str  # kbps
    container_format: str
    
    def __post_init__(self):
        self.width, self.height = self.resolution

@dataclass
class OptimizationResult:
    success: bool
    output_path: str
    platform: Platform
    original_size: int
    optimized_size: int
    compression_ratio: float
    processing_time: float
    specs_applied: Dict[str, Any]
    warnings: List[str]
    error_message: Optional[str] = None

class PlatformSpecsManager:
    """プラットフォーム別仕様管理"""
    
    def __init__(self):
        self.specs = self._initialize_platform_specs()
    
    def _initialize_platform_specs(self) -> Dict[Platform, PlatformSpec]:
        """各プラットフォームの仕様定義"""
        return {
            Platform.YOUTUBE: PlatformSpec(
                resolution=(1920, 1080),
                aspect_ratio="16:9",
                max_duration=43200,  # 12 hours
                min_duration=1,
                max_file_size=256000,  # 256GB
                video_codec="h264",
                audio_codec="aac",
                fps=30,
                bitrate_video="5000k",
                bitrate_audio="128k",
                container_format="mp4"
            ),
            
            Platform.INSTAGRAM_FEED: PlatformSpec(
                resolution=(1080, 1080),
                aspect_ratio="1:1",
                max_duration=60,
                min_duration=3,
                max_file_size=100,  # 100MB
                video_codec="h264",
                audio_codec="aac",
                fps=30,
                bitrate_video="3500k",
                bitrate_audio="128k",
                container_format="mp4"
            ),
            
            Platform.INSTAGRAM_STORY: PlatformSpec(
                resolution=(1080, 1920),
                aspect_ratio="9:16",
                max_duration=15,
                min_duration=1,
                max_file_size=100,  # 100MB
                video_codec="h264",
                audio_codec="aac",
                fps=30,
                bitrate_video="3500k",
                bitrate_audio="128k",
                container_format="mp4"
            ),
            
            Platform.INSTAGRAM_REEL: PlatformSpec(
                resolution=(1080, 1920),
                aspect_ratio="9:16",
                max_duration=90,
                min_duration=3,
                max_file_size=100,  # 100MB
                video_codec="h264",
                audio_codec="aac",
                fps=30,
                bitrate_video="3500k",
                bitrate_audio="128k",
                container_format="mp4"
            ),
            
            Platform.TIKTOK: PlatformSpec(
                resolution=(1080, 1920),
                aspect_ratio="9:16",
                max_duration=180,  # 3 minutes
                min_duration=1,
                max_file_size=72,  # 72MB
                video_codec="h264",
                audio_codec="aac",
                fps=30,
                bitrate_video="4000k",
                bitrate_audio="128k",
                container_format="mp4"
            ),
            
            Platform.TWITTER: PlatformSpec(
                resolution=(1280, 720),
                aspect_ratio="16:9",
                max_duration=140,
                min_duration=1,
                max_file_size=512,  # 512MB
                video_codec="h264",
                audio_codec="aac",
                fps=30,
                bitrate_video="2500k",
                bitrate_audio="128k",
                container_format="mp4"
            ),
            
            Platform.LINKEDIN: PlatformSpec(
                resolution=(1920, 1080),
                aspect_ratio="16:9",
                max_duration=600,  # 10 minutes
                min_duration=3,
                max_file_size=5000,  # 5GB
                video_codec="h264",
                audio_codec="aac",
                fps=30,
                bitrate_video="5000k",
                bitrate_audio="128k",
                container_format="mp4"
            ),
            
            Platform.FACEBOOK: PlatformSpec(
                resolution=(1920, 1080),
                aspect_ratio="16:9",
                max_duration=14400,  # 4 hours
                min_duration=1,
                max_file_size=10000,  # 10GB
                video_codec="h264",
                audio_codec="aac",
                fps=30,
                bitrate_video="4000k",
                bitrate_audio="128k",
                container_format="mp4"
            ),
            
            Platform.FACEBOOK_STORY: PlatformSpec(
                resolution=(1080, 1920),
                aspect_ratio="9:16",
                max_duration=20,
                min_duration=1,
                max_file_size=100,  # 100MB
                video_codec="h264",
                audio_codec="aac",
                fps=30,
                bitrate_video="3500k",
                bitrate_audio="128k",
                container_format="mp4"
            )
        }
    
    def get_spec(self, platform: Platform) -> PlatformSpec:
        """プラットフォーム仕様取得"""
        return self.specs.get(platform)
    
    def get_all_specs(self) -> Dict[Platform, PlatformSpec]:
        """全プラットフォーム仕様取得"""
        return self.specs.copy()

class VideoAnalyzer:
    """動画ファイル分析"""
    
    @staticmethod
    def analyze_video(video_path: str) -> Dict[str, Any]:
        """動画ファイルの詳細情報取得"""
        try:
            probe = ffmpeg.probe(video_path)
            
            video_stream = None
            audio_stream = None
            
            for stream in probe['streams']:
                if stream['codec_type'] == 'video' and video_stream is None:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and audio_stream is None:
                    audio_stream = stream
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # 基本情報
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            duration = float(video_stream.get('duration', probe['format']['duration']))
            fps = eval(video_stream['r_frame_rate'])  # "30/1" -> 30.0
            
            # ファイルサイズ
            file_size = int(probe['format']['size'])
            file_size_mb = file_size / (1024 * 1024)
            
            # ビットレート
            bitrate_video = int(video_stream.get('bit_rate', 0)) / 1000  # kbps
            bitrate_audio = int(audio_stream.get('bit_rate', 0)) / 1000 if audio_stream else 0
            
            # アスペクト比計算
            aspect_ratio = VideoAnalyzer._calculate_aspect_ratio(width, height)
            
            return {
                'width': width,
                'height': height,
                'duration': duration,
                'fps': fps,
                'file_size': file_size,
                'file_size_mb': file_size_mb,
                'bitrate_video': bitrate_video,
                'bitrate_audio': bitrate_audio,
                'aspect_ratio': aspect_ratio,
                'video_codec': video_stream['codec_name'],
                'audio_codec': audio_stream['codec_name'] if audio_stream else None,
                'has_audio': audio_stream is not None,
                'container_format': probe['format']['format_name']
            }
            
        except Exception as e:
            raise ValueError(f"Failed to analyze video: {str(e)}")
    
    @staticmethod
    def _calculate_aspect_ratio(width: int, height: int) -> str:
        """アスペクト比計算"""
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        common_divisor = gcd(width, height)
        ratio_w = width // common_divisor
        ratio_h = height // common_divisor
        
        # 一般的なアスペクト比にマッピング
        ratio_map = {
            (16, 9): "16:9",
            (9, 16): "9:16",
            (1, 1): "1:1",
            (4, 3): "4:3",
            (3, 4): "3:4",
            (21, 9): "21:9",
            (5, 3): "5:3",
            (3, 5): "3:5"
        }
        
        return ratio_map.get((ratio_w, ratio_h), f"{ratio_w}:{ratio_h}")

class VideoProcessor:
    """動画処理エンジン"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = logging.getLogger(__name__)
    
    def __del__(self):
        """一時ディレクトリクリーンアップ"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def resize_video(self, input_path: str, output_path: str, 
                    target_resolution: Tuple[int, int], 
                    maintain_aspect: bool = True) -> str:
        """動画リサイズ"""
        target_width, target_height = target_resolution
        
        if maintain_aspect:
            # アスペクト比を維持してリサイズ + クロップ
            scale_filter = f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase"
            crop_filter = f"crop={target_width}:{target_height}"
            video_filter = f"{scale_filter},{crop_filter}"
        else:
            # 強制リサイズ
            video_filter = f"scale={target_width}:{target_height}"
        
        try:
            (
                ffmpeg
                .input(input_path)
                .video.filter(video_filter)
                .output(output_path)
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Video resize failed: {e}")
    
    def adjust_fps(self, input_path: str, output_path: str, target_fps: int) -> str:
        """フレームレート調整"""
        try:
            (
                ffmpeg
                .input(input_path)
                .video.filter('fps', fps=target_fps)
                .output(output_path)
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"FPS adjustment failed: {e}")
    
    def adjust_bitrate(self, input_path: str, output_path: str, 
                      video_bitrate: str, audio_bitrate: str) -> str:
        """ビットレート調整"""
        try:
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path,
                    video_bitrate=video_bitrate,
                    audio_bitrate=audio_bitrate
                )
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Bitrate adjustment failed: {e}")
    
    def trim_video(self, input_path: str, output_path: str, 
                  start_time: float = 0, duration: Optional[float] = None) -> str:
        """動画トリミング"""
        try:
            input_stream = ffmpeg.input(input_path, ss=start_time)
            
            if duration:
                input_stream = ffmpeg.input(input_path, ss=start_time, t=duration)
            
            (
                input_stream
                .output(output_path)
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Video trimming failed: {e}")
    
    def add_watermark(self, input_path: str, output_path: str, 
                     watermark_path: str, position: str = "bottom_right",
                     opacity: float = 0.7) -> str:
        """ウォーターマーク追加"""
        try:
            # ポジション設定
            position_map = {
                "top_left": "10:10",
                "top_right": "W-w-10:10",
                "bottom_left": "10:H-h-10",
                "bottom_right": "W-w-10:H-h-10",
                "center": "(W-w)/2:(H-h)/2"
            }
            
            overlay_position = position_map.get(position, position_map["bottom_right"])
            
            # 透明度調整
            watermark = ffmpeg.input(watermark_path).video.filter('format', 'rgba').filter('colorchannelmixer', aa=opacity)
            
            (
                ffmpeg
                .input(input_path)
                .overlay(watermark, x=overlay_position.split(':')[0], y=overlay_position.split(':')[1])
                .output(output_path)
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Watermark addition failed: {e}")
    
    def create_thumbnail(self, input_path: str, output_path: str, 
                        timestamp: float = 1.0, 
                        resolution: Tuple[int, int] = (1280, 720)) -> str:
        """サムネイル生成"""
        try:
            width, height = resolution
            (
                ffmpeg
                .input(input_path, ss=timestamp)
                .video.filter('scale', width, height)
                .output(output_path, vframes=1)
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Thumbnail creation failed: {e}")

class MultiPlatformVideoOptimizer:
    """マルチプラットフォーム動画最適化システム"""
    
    def __init__(self, output_directory: str = "optimized_videos"):
        self.output_directory = output_directory
        self.specs_manager = PlatformSpecsManager()
        self.video_processor = VideoProcessor()
        
        # ディレクトリ作成
        os.makedirs(output_directory, exist_ok=True)
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def optimize_for_platform(self, input_video_path: str, platform: Platform,
                            custom_settings: Dict[str, Any] = None,
                            add_watermark: bool = False,
                            watermark_path: str = None) -> OptimizationResult:
        """プラットフォーム別最適化実行"""
        
        start_time = datetime.now()
        
        try:
            # 入力動画分析
            video_info = VideoAnalyzer.analyze_video(input_video_path)
            original_size = video_info['file_size']
            
            # プラットフォーム仕様取得
            spec = self.specs_manager.get_spec(platform)
            if not spec:
                raise ValueError(f"Unsupported platform: {platform.value}")
            
            # カスタム設定適用
            if custom_settings:
                spec = self._apply_custom_settings(spec, custom_settings)
            
            # 出力ファイル名生成
            base_name = os.path.splitext(os.path.basename(input_video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_name}_{platform.value}_{timestamp}.{spec.container_format}"
            output_path = os.path.join(self.output_directory, output_filename)
            
            # 最適化処理実行
            warnings = []
            optimized_path = self._execute_optimization(
                input_video_path, output_path, video_info, spec, warnings
            )
            
            # ウォーターマーク追加
            if add_watermark and watermark_path and os.path.exists(watermark_path):
                watermarked_path = output_path.replace(f".{spec.container_format}", 
                                                     f"_watermarked.{spec.container_format}")
                self.video_processor.add_watermark(optimized_path, watermarked_path, watermark_path)
                optimized_path = watermarked_path
            
            # サムネイル生成
            thumbnail_path = output_path.replace(f".{spec.container_format}", "_thumbnail.jpg")
            try:
                self.video_processor.create_thumbnail(optimized_path, thumbnail_path)
            except Exception as e:
                warnings.append(f"Thumbnail generation failed: {str(e)}")
            
            # 最適化後のファイルサイズ
            optimized_size = os.path.getsize(optimized_path)
            compression_ratio = (original_size - optimized_size) / original_size * 100
            
            # 処理時間計算
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 結果作成
            result = OptimizationResult(
                success=True,
                output_path=optimized_path,
                platform=platform,
                original_size=original_size,
                optimized_size=optimized_size,
                compression_ratio=compression_ratio,
                processing_time=processing_time,
                specs_applied=self._spec_to_dict(spec),
                warnings=warnings
            )
            
            self.logger.info(f"Optimization completed for {platform.value}: {compression_ratio:.1f}% compression")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                success=False,
                output_path="",
                platform=platform,
                original_size=video_info.get('file_size', 0) if 'video_info' in locals() else 0,
                optimized_size=0,
                compression_ratio=0,
                processing_time=processing_time,
                specs_applied={},
                warnings=[],
                error_message=str(e)
            )
    
    def optimize_for_multiple_platforms(self, input_video_path: str, 
                                      platforms: List[Platform],
                                      custom_settings: Dict[Platform, Dict[str, Any]] = None,
                                      add_watermark: bool = False,
                                      watermark_path: str = None) -> Dict[Platform, OptimizationResult]:
        """複数プラットフォーム一括最適化"""
        
        results = {}
        
        for platform in platforms:
            platform_settings = None
            if custom_settings and platform in custom_settings:
                platform_settings = custom_settings[platform]
            
            result = self.optimize_for_platform(
                input_video_path, platform, platform_settings, 
                add_watermark, watermark_path
            )
            
            results[platform] = result
        
        return results
    
    def _execute_optimization(self, input_path: str, output_path: str,
                            video_info: Dict[str, Any], spec: PlatformSpec,
                            warnings: List[str]) -> str:
        """最適化処理実行"""
        
        temp_files = []
        current_path = input_path
        
        try:
            # 1. 時間制限チェック・トリミング
            if video_info['duration'] > spec.max_duration:
                trimmed_path = os.path.join(self.video_processor.temp_dir, "trimmed.mp4")
                current_path = self.video_processor.trim_video(
                    current_path, trimmed_path, 0, spec.max_duration
                )
                temp_files.append(trimmed_path)
                warnings.append(f"Video trimmed to {spec.max_duration} seconds")
            
            # 2. 解像度・アスペクト比調整
            if (video_info['width'] != spec.width or 
                video_info['height'] != spec.height):
                
                resized_path = os.path.join(self.video_processor.temp_dir, "resized.mp4")
                current_path = self.video_processor.resize_video(
                    current_path, resized_path, (spec.width, spec.height)
                )
                temp_files.append(resized_path)
            
            # 3. フレームレート調整
            if abs(video_info['fps'] - spec.fps) > 1:
                fps_path = os.path.join(self.video_processor.temp_dir, "fps_adjusted.mp4")
                current_path = self.video_processor.adjust_fps(current_path, fps_path, spec.fps)
                temp_files.append(fps_path)
            
            # 4. 最終エンコーディング（コーデック・ビットレート）
            self._final_encode(current_path, output_path, spec)
            
            # 5. ファイルサイズチェック
            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            if output_size_mb > spec.max_file_size:
                warnings.append(f"Output file size ({output_size_mb:.1f}MB) exceeds platform limit ({spec.max_file_size}MB)")
                
                # ビットレート調整してリトライ
                reduced_bitrate = int(spec.bitrate_video.replace('k', '')) * 0.7
                self._final_encode(current_path, output_path, spec, f"{reduced_bitrate}k")
            
            return output_path
            
        finally:
            # 一時ファイルクリーンアップ
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def _final_encode(self, input_path: str, output_path: str, spec: PlatformSpec,
                     custom_video_bitrate: str = None) -> None:
        """最終エンコーディング"""
        
        video_bitrate = custom_video_bitrate or spec.bitrate_video
        
        try:
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path,
                    vcodec=spec.video_codec,
                    acodec=spec.audio_codec,
                    video_bitrate=video_bitrate,
                    audio_bitrate=spec.bitrate_audio,
                    r=spec.fps,
                    format=spec.container_format
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Final encoding failed: {e}")
    
    def _apply_custom_settings(self, spec: PlatformSpec, 
                              custom_settings: Dict[str, Any]) -> PlatformSpec:
        """カスタム設定適用"""
        
        # 新しいPlatformSpecオブジェクトを作成（元のspecは変更しない）
        new_spec = PlatformSpec(
            resolution=custom_settings.get('resolution', spec.resolution),
            aspect_ratio=custom_settings.get('aspect_ratio', spec.aspect_ratio),
            max_duration=custom_settings.get('max_duration', spec.max_duration),
            min_duration=custom_settings.get('min_duration', spec.min_duration),
            max_file_size=custom_settings.get('max_file_size', spec.max_file_size),
            video_codec=custom_settings.get('video_codec', spec.video_codec),
            audio_codec=custom_settings.get('audio_codec', spec.audio_codec),
            fps=custom_settings.get('fps', spec.fps),
            bitrate_video=custom_settings.get('bitrate_video', spec.bitrate_video),
            bitrate_audio=custom_settings.get('bitrate_audio', spec.bitrate_audio),
            container_format=custom_settings.get('container_format', spec.container_format)
        )
        
        return new_spec
    
    def _spec_to_dict(self, spec: PlatformSpec) -> Dict[str, Any]:
        """PlatformSpecを辞書に変換"""
        return {
            'resolution': spec.resolution,
            'aspect_ratio': spec.aspect_ratio,
            'max_duration': spec.max_duration,
            'min_duration': spec.min_duration,
            'max_file_size': spec.max_file_size,
            'video_codec': spec.video_codec,
            'audio_codec': spec.audio_codec,
            'fps': spec.fps,
            'bitrate_video': spec.bitrate_video,
            'bitrate_audio': spec.bitrate_audio,
            'container_format': spec.container_format
        }
    
    def get_platform_requirements(self, platform: Platform) -> Dict[str, Any]:
        """プラットフォーム要件取得"""
        spec = self.specs_manager.get_spec(platform)
        if not spec:
            return {}
        
        return self._spec_to_dict(spec)
    
    def validate_video_for_platform(self, video_path: str, 
                                   platform: Platform) -> Dict[str, Any]:
        """プラットフォーム適合性検証"""
        
        try:
            video_info = VideoAnalyzer.analyze_video(video_path)
            spec = self.specs_manager.get_spec(platform)
            
            issues = []
            warnings = []
            
            # 解像度チェック
            if (video_info['width'] != spec.width or 
                video_info['height'] != spec.height):
                issues.append(f"Resolution mismatch: {video_info['width']}x{video_info['height']} vs required {spec.width}x{spec.height}")
            
            # 時間制限チェック
            if video_info['duration'] > spec.max_duration:
                issues.append(f"Duration too long: {video_info['duration']:.1f}s vs max {spec.max_duration}s")
            elif video_info['duration'] < spec.min_duration:
                issues.append(f"Duration too short: {video_info['duration']:.1f}s vs min {spec.min_duration}s")
            
            # ファイルサイズチェック
            if video_info['file_size_mb'] > spec.max_file_size:
                issues.append(f"File size too large: {video_info['file_size_mb']:.1f}MB vs max {spec.max_file_size}MB")
            
            # フレームレートチェック
            if abs(video_info['fps'] - spec.fps) > 2:
                warnings.append(f"FPS difference: {video_info['fps']} vs recommended {spec.fps}")
            
            # コーデックチェック
            if video_info['video_codec'] != spec.video_codec:
                warnings.append(f"Video codec: {video_info['video_codec']} vs recommended {spec.video_codec}")
            
            is_compliant = len(issues) == 0
            needs_optimization = len(issues) > 0 or len(warnings) > 0
            
            return {
                'is_compliant': is_compliant,
                'needs_optimization': needs_optimization,
                'issues': issues,
                'warnings': warnings,
                'video_info': video_info,
                'platform_requirements': self._spec_to_dict(spec)
            }
            
        except Exception as e:
            return {
                'is_compliant': False,
                'needs_optimization': True,
                'issues': [f"Analysis failed: {str(e)}"],
                'warnings': [],
                'video_info': {},
                'platform_requirements': {}
            }
    
    def generate_optimization_report(self, results: Dict[Platform, OptimizationResult]) -> str:
        """最適化レポート生成"""
        
        report_lines = [
            "=== Multi-Platform Video Optimization Report ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        successful_optimizations = [r for r in results.values() if r.success]
        failed_optimizations = [r for r in results.values() if not r.success]
        
        # サマリー
        report_lines.extend([
            "SUMMARY:",
            f"Total platforms: {len(results)}",
            f"Successful: {len(successful_optimizations)}",
            f"Failed: {len(failed_optimizations)}",
            ""
        ])
        
        # 成功した最適化
        if successful_optimizations:
            report_lines.append("SUCCESSFUL OPTIMIZATIONS:")
            
            for result in successful_optimizations:
                report_lines.extend([
                    f"Platform: {result.platform.value}",
                    f"  Output: {os.path.basename(result.output_path)}",
                    f"  Size reduction: {result.compression_ratio:.1f}%",
                    f"  Processing time: {result.processing_time:.1f}s",
                    f"  Warnings: {len(result.warnings)}",
                    ""
                ])
        
        # 失敗した最適化
        if failed_optimizations:
            report_lines.append("FAILED OPTIMIZATIONS:")
            
            for result in failed_optimizations:
                report_lines.extend([
                    f"Platform: {result.platform.value}",
                    f"  Error: {result.error_message}",
                    ""
                ])
        
        # 全体統計
        if successful_optimizations:
            total_original_size = sum(r.original_size for r in successful_optimizations)
            total_optimized_size = sum(r.optimized_size for r in successful_optimizations)
            overall_compression = (total_original_size - total_optimized_size) / total_original_size * 100
            avg_processing_time = sum(r.processing_time for r in successful_optimizations) / len(successful_optimizations)
            
            report_lines.extend([
                "OVERALL STATISTICS:",
                f"Total original size: {total_original_size / (1024*1024):.1f} MB",
                f"Total optimized size: {total_optimized_size / (1024*1024):.1f} MB",
                f"Overall compression: {overall_compression:.1f}%",
                f"Average processing time: {avg_processing_time:.1f}s",
                ""
            ])
        
        return "\n".join(report_lines)

# 使用例
def main():
    """マルチプラットフォーム最適化システムの使用例"""
    
    # 最適化システム初期化
    optimizer = MultiPlatformVideoOptimizer("output_videos")
    
    # 入力動画
    input_video = "sample_video.mp4"
    
    if not os.path.exists(input_video):
        print(f"Input video not found: {input_video}")
        return
    
    # 対象プラットフォーム
    target_platforms = [
        Platform.YOUTUBE,
        Platform.INSTAGRAM_FEED,
        Platform.INSTAGRAM_STORY,
        Platform.TIKTOK,
        Platform.TWITTER
    ]
    
    # カスタム設定（オプション）
    custom_settings = {
        Platform.YOUTUBE: {
            'bitrate_video': '8000k',  # 高品質設定
            'fps': 60
        },
        Platform.INSTAGRAM_STORY: {
            'max_duration': 10  # 10秒制限
        }
    }
    
    print("Starting multi-platform optimization...")
    
    # 複数プラットフォーム一括最適化
    results = optimizer.optimize_for_multiple_platforms(
        input_video, 
        target_platforms,
        custom_settings,
        add_watermark=True,
        watermark_path="watermark.png"
    )
    
    # 結果表示
    print("\n=== Optimization Results ===")
    for platform, result in results.items():
        if result.success:
            print(f"✅ {platform.value}: {result.compression_ratio:.1f}% compression")
            if result.warnings:
                for warning in result.warnings:
                    print(f"   ⚠️  {warning}")
        else:
            print(f"❌ {platform.value}: {result.error_message}")
    
    # レポート生成
    report = optimizer.generate_optimization_report(results)
    print(f"\n{report}")
    
    # 個別プラットフォーム検証例
    print("\n=== Platform Validation Example ===")
    validation = optimizer.validate_video_for_platform(input_video, Platform.TIKTOK)
    print(f"TikTok compliance: {validation['is_compliant']}")
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")

if __name__ == "__main__":
    main()
