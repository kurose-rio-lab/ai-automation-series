import cv2
import numpy as np
from PIL import Image, ImageStat
import colorsys
from typing import Dict, List, Tuple, Optional

class ImageQualityChecker:
    def __init__(self, min_resolution=(1024, 1024), quality_threshold=0.7):
        self.min_resolution = min_resolution
        self.quality_threshold = quality_threshold
        
        # 品質評価項目の重み付け
        self.weights = {
            'resolution': 0.2,
            'sharpness': 0.25,
            'color_balance': 0.15,
            'contrast': 0.15,
            'brightness': 0.1,
            'saturation': 0.1,
            'noise_level': 0.05
        }
        
    def check_resolution(self, image_path: str) -> Dict:
        """解像度チェック"""
        try:
            image = Image.open(image_path)
            width, height = image.size
            total_pixels = width * height
            
            meets_minimum = (width >= self.min_resolution[0] and 
                           height >= self.min_resolution[1])
            
            score = 1.0 if meets_minimum else (
                min(width / self.min_resolution[0], height / self.min_resolution[1])
            )
            
            return {
                'score': score,
                'width': width,
                'height': height,
                'total_pixels': total_pixels,
                'meets_minimum': meets_minimum,
                'details': f"{width}x{height} ({'✓' if meets_minimum else '✗'} minimum)"
            }
        except Exception as e:
            return {'score': 0.0, 'error': str(e)}
            
    def check_sharpness(self, image_path: str) -> Dict:
        """シャープネスチェック"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Unable to load image")
                
            # ラプラシアン変数によるシャープネス測定
            laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
            
            # ソーベルフィルターによる追加測定
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_var = np.sqrt(sobelx**2 + sobely**2).var()
            
            # 正規化スコア計算
            sharpness_score = min(laplacian_var / 1000, 1.0)  # 閾値調整可能
            
            quality_level = 'excellent' if laplacian_var > 500 else \
                           'good' if laplacian_var > 200 else \
                           'fair' if laplacian_var > 100 else 'poor'
                           
            return {
                'score': sharpness_score,
                'laplacian_variance': laplacian_var,
                'sobel_variance': sobel_var,
                'quality_level': quality_level,
                'details': f"Laplacian: {laplacian_var:.1f} ({quality_level})"
            }
        except Exception as e:
            return {'score': 0.0, 'error': str(e)}
            
    def check_color_balance(self, image_path: str) -> Dict:
        """カラーバランスチェック"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Unable to load image")
                
            # BGR チャンネル分離
            b, g, r = cv2.split(image)
            
            # 各チャンネルの統計
            b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
            b_std, g_std, r_std = np.std(b), np.std(g), np.std(r)
            
            # カラーバランススコア計算
            mean_balance = 1 - (np.std([b_mean, g_mean, r_mean]) / 255)
            std_balance = 1 - (np.std([b_std, g_std, r_std]) / 128)
            
            overall_balance = (mean_balance + std_balance) / 2
            
            # 色相分布分析
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 彩度と明度の分布
            saturation_mean = np.mean(s)
            value_mean = np.mean(v)
            
            return {
                'score': overall_balance,
                'mean_balance': mean_balance,
                'std_balance': std_balance,
                'rgb_means': [r_mean, g_mean, b_mean],
                'saturation_mean': saturation_mean,
                'value_mean': value_mean,
                'details': f"Balance: {overall_balance:.3f}, Sat: {saturation_mean:.1f}"
            }
        except Exception as e:
            return {'score': 0.0, 'error': str(e)}
            
    def check_contrast(self, image_path: str) -> Dict:
        """コントラストチェック"""
        try:
            image = Image.open(image_path).convert('L')  # グレースケール変換
            stat = ImageStat.Stat(image)
            
            # 標準偏差をコントラスト指標として使用
            contrast = stat.stddev[0]
            
            # RMSコントラスト計算
            img_array = np.array(image)
            rms_contrast = np.sqrt(np.mean((img_array - np.mean(img_array)) ** 2))
            
            # 正規化スコア
            contrast_score = min(contrast / 64, 1.0)  # 64が理想的な標準偏差
            
            quality_level = 'excellent' if contrast > 60 else \
                           'good' if contrast > 45 else \
                           'fair' if contrast > 30 else 'poor'
                           
            return {
                'score': contrast_score,
                'contrast': contrast,
                'rms_contrast': rms_contrast,
                'quality_level': quality_level,
                'details': f"Contrast: {contrast:.1f} ({quality_level})"
            }
        except Exception as e:
            return {'score': 0.0, 'error': str(e)}
            
    def check_brightness(self, image_path: str) -> Dict:
        """明度チェック"""
        try:
            image = Image.open(image_path).convert('L')
            stat = ImageStat.Stat(image)
            brightness = stat.mean[0]
            
            # 理想的な明度範囲（80-180）との距離で評価
            ideal_range = (80, 180)
            if ideal_range[0] <= brightness <= ideal_range[1]:
                brightness_score = 1.0
            else:
                distance_from_ideal = min(
                    abs(brightness - ideal_range[0]),
                    abs(brightness - ideal_range[1])
                )
                brightness_score = max(0, 1 - distance_from_ideal / 127.5)
                
            quality_level = 'excellent' if brightness_score > 0.9 else \
                           'good' if brightness_score > 0.7 else \
                           'fair' if brightness_score > 0.5 else 'poor'
                           
            return {
                'score': brightness_score,
                'brightness': brightness,
                'quality_level': quality_level,
                'details': f"Brightness: {brightness:.1f} ({quality_level})"
            }
        except Exception as e:
            return {'score': 0.0, 'error': str(e)}
            
    def check_noise_level(self, image_path: str) -> Dict:
        """ノイズレベルチェック"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Unable to load image")
                
            # ガウシアンフィルターを適用して元画像と比較
            filtered = cv2.GaussianBlur(image, (5, 5), 0)
            noise = cv2.absdiff(image, filtered)
            
            # ノイズレベル計算
            noise_level = np.mean(noise)
            
            # スコア計算（ノイズが少ないほど高スコア）
            noise_score = max(0, 1 - noise_level / 50)  # 50を閾値として調整
            
            quality_level = 'excellent' if noise_level < 5 else \
                           'good' if noise_level < 10 else \
                           'fair' if noise_level < 20 else 'poor'
                           
            return {
                'score': noise_score,
                'noise_level': noise_level,
                'quality_level': quality_level,
                'details': f"Noise: {noise_level:.2f} ({quality_level})"
            }
        except Exception as e:
            return {'score': 0.0, 'error': str(e)}
            
    def overall_quality_score(self, image_path: str) -> Dict:
        """総合品質スコア計算"""
        results = {}
        
        # 各項目の評価実行
        checks = [
            ('resolution', self.check_resolution),
            ('sharpness', self.check_sharpness),
            ('color_balance', self.check_color_balance),
            ('contrast', self.check_contrast),
            ('brightness', self.check_brightness),
            ('noise_level', self.check_noise_level)
        ]
        
        total_weighted_score = 0
        total_weight = 0
        
        for check_name, check_func in checks:
            try:
                result = check_func(image_path)
                results[check_name] = result
                
                if 'score' in result:
                    weight = self.weights.get(check_name, 0)
                    total_weighted_score += result['score'] * weight
                    total_weight += weight
                    
            except Exception as e:
                results[check_name] = {'score': 0.0, 'error': str(e)}
                
        # 総合スコア計算
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        # 品質レベル決定
        if overall_score >= 0.9:
            quality_grade = 'A'
            recommendation = 'Excellent quality - Ready for production use'
        elif overall_score >= 0.8:
            quality_grade = 'B'
            recommendation = 'Good quality - Minor improvements possible'
        elif overall_score >= 0.7:
            quality_grade = 'C'
            recommendation = 'Acceptable quality - Consider regeneration'
        elif overall_score >= 0.6:
            quality_grade = 'D'
            recommendation = 'Poor quality - Regeneration recommended'
        else:
            quality_grade = 'F'
            recommendation = 'Unacceptable quality - Must regenerate'
            
        return {
            'overall_score': overall_score,
            'quality_grade': quality_grade,
            'recommendation': recommendation,
            'meets_threshold': overall_score >= self.quality_threshold,
            'individual_scores': results,
            'summary': self.generate_quality_summary(results)
        }
        
    def generate_quality_summary(self, results: Dict) -> str:
        """品質評価サマリー生成"""
        summary_parts = []
        
        for check_name, result in results.items():
            if 'score' in result and 'quality_level' in result:
                summary_parts.append(f"{check_name}: {result['quality_level']}")
                
        return " | ".join(summary_parts)
        
    def batch_quality_check(self, image_paths: List[str]) -> Dict:
        """バッチ品質チェック"""
        batch_results = {}
        
        for path in image_paths:
            try:
                batch_results[path] = self.overall_quality_score(path)
            except Exception as e:
                batch_results[path] = {'error': str(e), 'overall_score': 0.0}
                
        # バッチ統計
        valid_scores = [
            result['overall_score'] 
            for result in batch_results.values() 
            if 'overall_score' in result
        ]
        
        batch_stats = {
            'total_images': len(image_paths),
            'successful_checks': len(valid_scores),
            'average_score': np.mean(valid_scores) if valid_scores else 0,
            'pass_rate': sum(1 for score in valid_scores if score >= self.quality_threshold) / len(valid_scores) if valid_scores else 0,
            'individual_results': batch_results
        }
        
        return batch_stats

# 使用例
if __name__ == "__main__":
    checker = ImageQualityChecker(
        min_resolution=(1024, 1024),
        quality_threshold=0.75
    )
    
    # 単一画像チェック
    test_image_path = "/content/outputs/test_image.jpg"
    result = checker.overall_quality_score(test_image_path)
    
    print(f"Overall Score: {result['overall_score']:.3f}")
    print(f"Quality Grade: {result['quality_grade']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Summary: {result['summary']}")
    
    # 個別スコア詳細
    for check_name, check_result in result['individual_scores'].items():
        if 'score' in check_result:
            print(f"  {check_name}: {check_result['score']:.3f} - {check_result.get('details', '')}")
