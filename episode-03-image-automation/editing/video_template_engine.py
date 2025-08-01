# video_template_engine.py
# プロレベルの動画テンプレートシステム

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
import subprocess
import tempfile

class TemplateType(Enum):
    PRODUCT_SHOWCASE = "product_showcase"
    SOCIAL_STORY = "social_story"
    EXPLAINER_VIDEO = "explainer_video"
    BRAND_INTRO = "brand_intro"
    TESTIMONIAL = "testimonial"
    TUTORIAL = "tutorial"

class SegmentType(Enum):
    LOGO_ANIMATION = "logo_animation"
    PRODUCT_VIDEO = "product_video"
    TEXT_INFO = "text_info"
    CTA_DISPLAY = "cta_display"
    ATTENTION_HOOK = "attention_hook"
    MAIN_CONTENT = "main_content"
    BRAND_EXPOSURE = "brand_exposure"
    ACTION_PROMPT = "action_prompt"
    INTRO_SEQUENCE = "intro_sequence"
    EXPLANATION = "explanation"
    DEMONSTRATION = "demonstration"
    CONCLUSION = "conclusion"

@dataclass
class SegmentConfig:
    start_time: float
    end_time: float
    segment_type: SegmentType
    asset_path: Optional[str] = None
    text_content: Optional[str] = None
    transition_type: str = "fade"
    animation_style: str = "smooth"
    audio_level: float = 1.0
    effects: List[str] = None
    
    def __post_init__(self):
        if self.effects is None:
            self.effects = []
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

@dataclass
class VideoTemplate:
    name: str
    template_type: TemplateType
    total_duration: int
    aspect_ratio: str
    segments: List[SegmentConfig]
    default_style: Dict[str, Any]
    audio_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'template_type': self.template_type.value,
            'total_duration': self.total_duration,
            'aspect_ratio': self.aspect_ratio,
            'segments': [asdict(segment) for segment in self.segments],
            'default_style': self.default_style,
            'audio_config': self.audio_config
        }

class VideoTemplateEngine:
    """プロレベルの動画テンプレートシステム"""
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = templates_dir
        self.templates = {}
        self.logger = logging.getLogger(__name__)
        
        # ディレクトリ作成
        os.makedirs(templates_dir, exist_ok=True)
        
        # デフォルトテンプレート初期化
        self._initialize_default_templates()
        
        # エフェクトライブラリ
        self.effects_library = self._load_effects_library()
        
    def _initialize_default_templates(self):
        """デフォルトテンプレート初期化"""
        
        # 商品紹介テンプレート（30秒）
        product_showcase = VideoTemplate(
            name="Product Showcase",
            template_type=TemplateType.PRODUCT_SHOWCASE,
            total_duration=30,
            aspect_ratio="16:9",
            segments=[
                SegmentConfig(0.0, 3.0, SegmentType.LOGO_ANIMATION, 
                            transition_type="fade_in", animation_style="elegant"),
                SegmentConfig(3.0, 20.0, SegmentType.PRODUCT_VIDEO, 
                            transition_type="slide", animation_style="smooth"),
                SegmentConfig(20.0, 27.0, SegmentType.TEXT_INFO, 
                            transition_type="zoom", animation_style="dynamic"),
                SegmentConfig(27.0, 30.0, SegmentType.CTA_DISPLAY, 
                            transition_type="fade_out", animation_style="attention")
            ],
            default_style={
                'color_scheme': 'professional',
                'font_family': 'modern_sans',
                'lighting': 'soft_professional',
                'camera_movement': 'steady_with_subtle_motion'
            },
            audio_config={
                'background_music': True,
                'narration': True,
                'sound_effects': False,
                'music_genre': 'corporate_uplifting'
            }
        )
        
        # SNSストーリーテンプレート（15秒）
        social_story = VideoTemplate(
            name="Social Story",
            template_type=TemplateType.SOCIAL_STORY,
            total_duration=15,
            aspect_ratio="9:16",
            segments=[
                SegmentConfig(0.0, 2.0, SegmentType.ATTENTION_HOOK, 
                            transition_type="quick_cut", animation_style="energetic"),
                SegmentConfig(2.0, 10.0, SegmentType.MAIN_CONTENT, 
                            transition_type="dynamic_zoom", animation_style="engaging"),
                SegmentConfig(10.0, 13.0, SegmentType.BRAND_EXPOSURE, 
                            transition_type="slide_up", animation_style="smooth"),
                SegmentConfig(13.0, 15.0, SegmentType.ACTION_PROMPT, 
                            transition_type="pulse", animation_style="urgent")
            ],
            default_style={
                'color_scheme': 'vibrant',
                'font_family': 'trendy_bold',
                'lighting': 'dynamic_colorful',
                'camera_movement': 'handheld_dynamic'
            },
            audio_config={
                'background_music': True,
                'narration': False,
                'sound_effects': True,
                'music_genre': 'trending_upbeat'
            }
        )
        
        # 説明動画テンプレート（60秒）
        explainer_video = VideoTemplate(
            name="Explainer Video",
            template_type=TemplateType.EXPLAINER_VIDEO,
            total_duration=60,
            aspect_ratio="16:9",
            segments=[
                SegmentConfig(0.0, 5.0, SegmentType.INTRO_SEQUENCE, 
                            transition_type="fade_in", animation_style="professional"),
                SegmentConfig(5.0, 15.0, SegmentType.EXPLANATION, 
                            transition_type="dissolve", animation_style="clear"),
                SegmentConfig(15.0, 40.0, SegmentType.DEMONSTRATION, 
                            transition_type="wipe", animation_style="instructional"),
                SegmentConfig(40.0, 50.0, SegmentType.TEXT_INFO, 
                            transition_type="slide", animation_style="informative"),
                SegmentConfig(50.0, 60.0, SegmentType.CONCLUSION, 
                            transition_type="fade_out", animation_style="confident")
            ],
            default_style={
                'color_scheme': 'educational',
                'font_family': 'readable_sans',
                'lighting': 'clear_even',
                'camera_movement': 'steady_controlled'
            },
            audio_config={
                'background_music': True,
                'narration': True,
                'sound_effects': True,
                'music_genre': 'educational_ambient'
            }
        )
        
        # ブランド紹介テンプレート（45秒）
        brand_intro = VideoTemplate(
            name="Brand Introduction",
            template_type=TemplateType.BRAND_INTRO,
            total_duration=45,
            aspect_ratio="16:9",
            segments=[
                SegmentConfig(0.0, 8.0, SegmentType.LOGO_ANIMATION, 
                            transition_type="cinematic_reveal", animation_style="premium"),
                SegmentConfig(8.0, 25.0, SegmentType.BRAND_EXPOSURE, 
                            transition_type="elegant_slide", animation_style="sophisticated"),
                SegmentConfig(25.0, 35.0, SegmentType.TEXT_INFO, 
                            transition_type="typewriter", animation_style="storytelling"),
                SegmentConfig(35.0, 45.0, SegmentType.CTA_DISPLAY, 
                            transition_type="elegant_fade", animation_style="invitation")
            ],
            default_style={
                'color_scheme': 'premium_brand',
                'font_family': 'elegant_serif',
                'lighting': 'cinematic_mood',
                'camera_movement': 'cinematic_smooth'
            },
            audio_config={
                'background_music': True,
                'narration': True,
                'sound_effects': False,
                'music_genre': 'cinematic_inspiring'
            }
        )
        
        # テンプレート登録
        self.templates = {
            TemplateType.PRODUCT_SHOWCASE: product_showcase,
            TemplateType.SOCIAL_STORY: social_story,
            TemplateType.EXPLAINER_VIDEO: explainer_video,
            TemplateType.BRAND_INTRO: brand_intro
        }
    
    def _load_effects_library(self) -> Dict[str, Dict[str, Any]]:
        """エフェクトライブラリ読み込み"""
        return {
            'transitions': {
                'fade': {'duration': 0.5, 'easing': 'ease_in_out'},
                'fade_in': {'duration': 1.0, 'easing': 'ease_in'},
                'fade_out': {'duration': 1.0, 'easing': 'ease_out'},
                'slide': {'duration': 0.8, 'direction': 'left', 'easing': 'ease_in_out'},
                'slide_up': {'duration': 0.6, 'direction': 'up', 'easing': 'ease_out'},
                'zoom': {'duration': 0.7, 'scale_start': 0.8, 'scale_end': 1.0},
                'dynamic_zoom': {'duration': 0.4, 'scale_start': 1.2, 'scale_end': 1.0},
                'quick_cut': {'duration': 0.1, 'style': 'instant'},
                'dissolve': {'duration': 1.0, 'opacity_curve': 'smooth'},
                'wipe': {'duration': 0.8, 'direction': 'horizontal'},
                'cinematic_reveal': {'duration': 2.0, 'style': 'premium'},
                'elegant_slide': {'duration': 1.2, 'style': 'smooth'},
                'typewriter': {'duration': 2.0, 'character_by_character': True},
                'pulse': {'duration': 0.3, 'intensity': 1.2}
            },
            'animations': {
                'smooth': {'curve': 'ease_in_out', 'intensity': 'subtle'},
                'elegant': {'curve': 'ease_in_out', 'intensity': 'refined'},
                'dynamic': {'curve': 'ease_out', 'intensity': 'energetic'},
                'attention': {'curve': 'bounce', 'intensity': 'high'},
                'energetic': {'curve': 'ease_in', 'intensity': 'high'},
                'engaging': {'curve': 'ease_in_out', 'intensity': 'medium'},
                'urgent': {'curve': 'ease_in', 'intensity': 'high'},
                'professional': {'curve': 'ease_in_out', 'intensity': 'low'},
                'clear': {'curve': 'linear', 'intensity': 'minimal'},
                'instructional': {'curve': 'ease_in_out', 'intensity': 'medium'},
                'informative': {'curve': 'ease_out', 'intensity': 'low'},
                'confident': {'curve': 'ease_in_out', 'intensity': 'medium'},
                'premium': {'curve': 'ease_in_out', 'intensity': 'sophisticated'},
                'sophisticated': {'curve': 'ease_in_out', 'intensity': 'refined'},
                'storytelling': {'curve': 'ease_in_out', 'intensity': 'narrative'},
                'invitation': {'curve': 'ease_out', 'intensity': 'welcoming'}
            },
            'effects': {
                'glow': {'intensity': 0.3, 'color': 'white'},
                'shadow': {'blur': 10, 'opacity': 0.3},
                'blur': {'radius': 5, 'type': 'gaussian'},
                'sharpen': {'intensity': 0.2},
                'color_grade': {'temperature': 0, 'tint': 0, 'saturation': 1.0},
                'vignette': {'intensity': 0.2, 'size': 0.8},
                'lens_flare': {'intensity': 0.4, 'position': 'top_right'},
                'particle_overlay': {'count': 50, 'type': 'sparkle'},
                'film_grain': {'intensity': 0.1, 'size': 1.0}
            }
        }
    
    def get_template(self, template_type: TemplateType) -> Optional[VideoTemplate]:
        """テンプレート取得"""
        return self.templates.get(template_type)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """利用可能テンプレート一覧"""
        return [
            {
                'type': template_type.value,
                'name': template.name,
                'duration': template.total_duration,
                'aspect_ratio': template.aspect_ratio,
                'segments_count': len(template.segments)
            }
            for template_type, template in self.templates.items()
        ]
    
    def apply_template(self, template_type: TemplateType, assets: List[Dict[str, Any]], 
                      customizations: Dict[str, Any] = None) -> Dict[str, Any]:
        """テンプレート適用メイン関数"""
        
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Template {template_type.value} not found")
        
        # カスタマイゼーション適用
        if customizations:
            template = self._apply_customizations(template, customizations)
        
        # アセット割り当て
        composition = self._assign_assets_to_segments(template, assets)
        
        # エフェクト設定
        composition = self._apply_effects(composition)
        
        # 品質チェック
        validation_result = self._validate_composition(composition)
        
        return {
            'template_name': template.name,
            'template_type': template_type.value,
            'total_duration': template.total_duration,
            'aspect_ratio': template.aspect_ratio,
            'composition': composition,
            'audio_config': template.audio_config,
            'validation': validation_result,
            'export_ready': validation_result['is_valid']
        }
    
    def _apply_customizations(self, template: VideoTemplate, 
                            customizations: Dict[str, Any]) -> VideoTemplate:
        """カスタマイゼーション適用"""
        
        # 持続時間調整
        if 'duration' in customizations:
            new_duration = customizations['duration']
            scale_factor = new_duration / template.total_duration
            
            # セグメント時間をスケール
            for segment in template.segments:
                segment.start_time *= scale_factor
                segment.end_time *= scale_factor
            
            template.total_duration = new_duration
        
        # アスペクト比変更
        if 'aspect_ratio' in customizations:
            template.aspect_ratio = customizations['aspect_ratio']
        
        # スタイル設定上書き
        if 'style' in customizations:
            template.default_style.update(customizations['style'])
        
        # 音声設定上書き
        if 'audio' in customizations:
            template.audio_config.update(customizations['audio'])
        
        return template
    
    def _assign_assets_to_segments(self, template: VideoTemplate, 
                                  assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """アセットをセグメントに割り当て"""
        
        composition = []
        asset_index = 0
        
        for segment in template.segments:
            segment_data = {
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'duration': segment.duration,
                'type': segment.segment_type.value,
                'transition': segment.transition_type,
                'animation': segment.animation_style,
                'audio_level': segment.audio_level,
                'effects': segment.effects.copy()
            }
            
            # アセット割り当て
            if segment.segment_type in [SegmentType.PRODUCT_VIDEO, SegmentType.MAIN_CONTENT, 
                                      SegmentType.DEMONSTRATION, SegmentType.BRAND_EXPOSURE]:
                if asset_index < len(assets):
                    segment_data['asset'] = assets[asset_index]
                    asset_index += 1
                else:
                    # アセット不足の場合は最後のアセットを再利用
                    if assets:
                        segment_data['asset'] = assets[-1]
            
            # テキストコンテンツ処理
            elif segment.segment_type in [SegmentType.TEXT_INFO, SegmentType.CTA_DISPLAY]:
                segment_data['text_content'] = self._generate_text_content(
                    segment.segment_type, template.template_type
                )
            
            # ロゴアニメーション
            elif segment.segment_type == SegmentType.LOGO_ANIMATION:
                segment_data['logo_config'] = {
                    'animation_type': 'elegant_reveal',
                    'duration': segment.duration,
                    'position': 'center'
                }
            
            composition.append(segment_data)
        
        return composition
    
    def _generate_text_content(self, segment_type: SegmentType, 
                              template_type: TemplateType) -> Dict[str, Any]:
        """テキストコンテンツ生成"""
        
        text_templates = {
            TemplateType.PRODUCT_SHOWCASE: {
                SegmentType.TEXT_INFO: {
                    'title': '革新的な機能',
                    'subtitle': '品質と性能の完璧な融合',
                    'bullet_points': ['高品質素材', '革新的デザイン', '優れた耐久性']
                },
                SegmentType.CTA_DISPLAY: {
                    'main_text': '今すぐ詳細をご覧ください',
                    'button_text': '詳細を見る',
                    'urgency_text': '限定オファー実施中'
                }
            },
            TemplateType.SOCIAL_STORY: {
                SegmentType.TEXT_INFO: {
                    'hashtags': ['#新商品', '#限定', '#話題'],
                    'caption': 'あなたの生活を変える体験'
                },
                SegmentType.ACTION_PROMPT: {
                    'main_text': 'フォロー&いいね!',
                    'emoji': '👍✨'
                }
            },
            TemplateType.EXPLAINER_VIDEO: {
                SegmentType.TEXT_INFO: {
                    'title': '重要なポイント',
                    'steps': ['ステップ1: 準備', 'ステップ2: 実行', 'ステップ3: 完了']
                }
            }
        }
        
        template_texts = text_templates.get(template_type, {})
        return template_texts.get(segment_type, {'text': 'デフォルトテキスト'})
    
    def _apply_effects(self, composition: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """エフェクト適用"""
        
        for segment in composition:
            # トランジション設定
            transition_name = segment.get('transition', 'fade')
            if transition_name in self.effects_library['transitions']:
                segment['transition_config'] = self.effects_library['transitions'][transition_name]
            
            # アニメーション設定
            animation_name = segment.get('animation', 'smooth')
            if animation_name in self.effects_library['animations']:
                segment['animation_config'] = self.effects_library['animations'][animation_name]
            
            # 追加エフェクト
            for effect_name in segment.get('effects', []):
                if effect_name in self.effects_library['effects']:
                    if 'effect_configs' not in segment:
                        segment['effect_configs'] = {}
                    segment['effect_configs'][effect_name] = self.effects_library['effects'][effect_name]
        
        return composition
    
    def _validate_composition(self, composition: List[Dict[str, Any]]) -> Dict[str, Any]:
        """コンポジション検証"""
        
        issues = []
        warnings = []
        
        # 時間的整合性チェック
        total_duration = 0
        for i, segment in enumerate(composition):
            if segment['start_time'] < 0:
                issues.append(f"Segment {i}: Negative start time")
            
            if segment['end_time'] <= segment['start_time']:
                issues.append(f"Segment {i}: Invalid time range")
            
            if i > 0 and segment['start_time'] < composition[i-1]['end_time']:
                warnings.append(f"Segment {i}: Overlapping with previous segment")
            
            total_duration = max(total_duration, segment['end_time'])
        
        # アセット存在チェック
        for i, segment in enumerate(composition):
            if 'asset' in segment:
                asset = segment['asset']
                if 'file_path' in asset and not os.path.exists(asset['file_path']):
                    issues.append(f"Segment {i}: Asset file not found: {asset['file_path']}")
        
        # 品質スコア計算
        quality_score = 1.0
        if issues:
            quality_score -= len(issues) * 0.2
        if warnings:
            quality_score -= len(warnings) * 0.1
        
        quality_score = max(0.0, min(1.0, quality_score))
        
        return {
            'is_valid': len(issues) == 0,
            'quality_score': quality_score,
            'total_duration': total_duration,
            'segment_count': len(composition),
            'issues': issues,
            'warnings': warnings,
            'recommendations': self._generate_recommendations(issues, warnings)
        }
    
    def _generate_recommendations(self, issues: List[str], 
                                warnings: List[str]) -> List[str]:
        """改善提案生成"""
        recommendations = []
        
        if issues:
            recommendations.append("重要な問題を修正してください")
        
        if warnings:
            recommendations.append("警告事項を確認し、必要に応じて調整してください")
        
        if not issues and not warnings:
            recommendations.append("コンポジションは最適化されています")
        
        return recommendations
    
    def export_composition(self, composition_data: Dict[str, Any], 
                          output_path: str, format: str = "json") -> str:
        """コンポジションエクスポート"""
        
        if format.lower() == "json":
            output_file = f"{output_path}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(composition_data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "ffmpeg":
            output_file = f"{output_path}_ffmpeg.txt"
            ffmpeg_commands = self._generate_ffmpeg_commands(composition_data)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(ffmpeg_commands))
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Composition exported to: {output_file}")
        return output_file
    
    def _generate_ffmpeg_commands(self, composition_data: Dict[str, Any]) -> List[str]:
        """FFmpeg コマンド生成"""
        commands = []
        
        # ベースコマンド
        commands.append("# FFmpeg Video Composition Commands")
        commands.append("# Generated by VideoTemplateEngine")
        commands.append("")
        
        # 各セグメントの処理
        for i, segment in enumerate(composition_data['composition']):
            if 'asset' in segment and 'file_path' in segment['asset']:
                asset_path = segment['asset']['file_path']
                start_time = segment['start_time']
                duration = segment['duration']
                
                cmd = f"ffmpeg -ss {start_time} -t {duration} -i \"{asset_path}\""
                
                # エフェクト追加
                filters = []
                
                # トランジション
                if 'transition_config' in segment:
                    transition = segment['transition_config']
                    if 'fade' in segment.get('transition', ''):
                        filters.append(f"fade=in:0:{transition.get('duration', 0.5)}")
                
                # フィルター適用
                if filters:
                    cmd += f" -vf \"{','.join(filters)}\""
                
                cmd += f" segment_{i}.mp4"
                commands.append(cmd)
        
        # 最終結合
        segment_count = len(composition_data['composition'])
        if segment_count > 1:
            inputs = " ".join([f"-i segment_{i}.mp4" for i in range(segment_count)])
            filter_complex = "|".join([f"[{i}:v][{i}:a]" for i in range(segment_count)])
            
            commands.append("")
            commands.append(f"ffmpeg {inputs} -filter_complex \"{filter_complex}concat=n={segment_count}:v=1:a=1[outv][outa]\" -map \"[outv]\" -map \"[outa]\" final_output.mp4")
        
        return commands

# 使用例
def main():
    """テンプレートエンジンの使用例"""
    
    # エンジン初期化
    engine = VideoTemplateEngine()
    
    # 利用可能テンプレート確認
    templates = engine.list_templates()
    print("利用可能テンプレート:")
    for template in templates:
        print(f"- {template['name']} ({template['duration']}秒, {template['aspect_ratio']})")
    
    # アセット準備
    assets = [
        {
            'file_path': '/path/to/product_image1.jpg',
            'type': 'image',
            'quality_score': 0.9
        },
        {
            'file_path': '/path/to/product_image2.jpg',
            'type': 'image',
            'quality_score': 0.85
        },
        {
            'file_path': '/path/to/brand_logo.png',
            'type': 'logo',
            'quality_score': 1.0
        }
    ]
    
    # カスタマイゼーション設定
    customizations = {
        'duration': 25,  # 30秒から25秒に短縮
        'style': {
            'color_scheme': 'modern_blue',
            'font_family': 'roboto_bold'
        },
        'audio': {
            'music_genre': 'tech_ambient'
        }
    }
    
    # テンプレート適用
    result = engine.apply_template(
        TemplateType.PRODUCT_SHOWCASE,
        assets,
        customizations
    )
    
    print(f"\nテンプレート適用結果:")
    print(f"- テンプレート: {result['template_name']}")
    print(f"- 総時間: {result['total_duration']}秒")
    print(f"- セグメント数: {len(result['composition'])}")
    print(f"- 品質スコア: {result['validation']['quality_score']:.2f}")
    print(f"- エクスポート準備: {result['export_ready']}")
    
    # コンポジションエクスポート
    if result['export_ready']:
        json_file = engine.export_composition(result, "output/composition", "json")
        ffmpeg_file = engine.export_composition(result, "output/composition", "ffmpeg")
        
        print(f"\nエクスポート完了:")
        print(f"- JSON: {json_file}")
        print(f"- FFmpeg: {ffmpeg_file}")

if __name__ == "__main__":
    main()
