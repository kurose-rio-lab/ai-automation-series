# elevenlabs_voice_automation.py
# ElevenLabs音声生成自動化システム

import os
import json
import asyncio
import aiohttp
import librosa
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import hashlib

# 音声設定クラス
@dataclass
class VoiceSettings:
    stability: float = 0.75  # 安定性 (0.0-1.0)
    similarity_boost: float = 0.75  # 類似性ブースト (0.0-1.0)
    style: float = 0.0  # スタイル強度 (0.0-1.0)
    use_speaker_boost: bool = True

# 感情タイプ列挙
class EmotionType(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    CALM = "calm"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"

