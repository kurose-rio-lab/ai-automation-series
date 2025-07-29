import requests
from PIL import Image
import io

class MultiPlatformDistributor:
    """マルチプラットフォーム配信自動化"""
    def __init__(self, api_keys):
        self.instagram_token = api_keys.get('instagram')
        self.twitter_keys = api_keys.get('twitter')
        self.facebook_token = api_keys.get('facebook')
        
    def optimize_for_platform(self, image_path, platform):
        """プラットフォーム別最適化"""
        image = Image.open(image_path)
        
        platform_specs = {
            'instagram_post': (1080, 1080),
            'instagram_story': (1080, 1920),
            'twitter_post': (1200, 675),
            'facebook_post': (1200, 630),
            'linkedin_post': (1200, 627)
        }
        
        if platform in platform_specs:
            target_size = platform_specs[platform]
            optimized = self.resize_with_aspect_ratio(image, target_size)
            return optimized
        
        return image
