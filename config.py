# config.py

import os

# =========================
# CONFIG
# =========================
PROMPT_DIR = "./prompts"
MODEL_DIR = "./models"
LORA_DIR = "./lora"
PRESETS_FILE = "character_presets.json"
MODEL_MAPPING_FILE = "model_mapping.json"
DENOISE_STRENGTH = 0.35   
GUIDANCE_SCALE = 6.5      

NUM_STEPS_BASE = 32       
NUM_STEPS_UPSCALE = 20    

# Common negative prompt additions for consistent quality
DEFAULT_NEG = "modern technology, plastic, glasses, car, text, watermark, signature, blurry, low quality, bad anatomy, deformed hands, duplicate, anime girl tropes, oversaturated"

# Pre-computation steps to ensure directories exist
os.makedirs(PROMPT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)
