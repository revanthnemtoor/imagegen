import os
import gc
import json
import random
import torch
import traceback
import sys
import readchar
from datetime import datetime
from PIL import Image

import config
from device_manager import DeviceManager
from pipeline_manager import SDPipelineController
import argparse

from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    UniPCMultistepScheduler,
    FlowMatchEulerDiscreteScheduler
)

import questionary

def scan_models():
    return sorted([f for f in os.listdir(config.MODEL_DIR) if f.endswith((".safetensors", ".ckpt"))])

def scan_loras():
    if not os.path.exists(config.LORA_DIR):
        os.makedirs(config.LORA_DIR)
    return sorted([f for f in os.listdir(config.LORA_DIR) if f.endswith(".safetensors")])

def get_prompts():
    files = os.listdir(config.PROMPT_DIR)
    pos_files = sorted([f for f in files if f.endswith("pos.txt")])
    pairs = []
    for pos in pos_files:
        base = pos.replace("pos.txt", "")
        neg = base + "neg.txt"
        pos_path = os.path.join(config.PROMPT_DIR, pos)
        neg_path = os.path.join(config.PROMPT_DIR, neg)
        if os.path.exists(neg_path):
            pairs.append((pos_path, neg_path))
        else:
            print(f"⚠️ Missing negative prompt for {pos}")
    return pairs

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

import atexit
print("\033[?1049h", end="")
def restore_terminal():
    print("\033[?1049l", end="")
atexit.register(restore_terminal)

def draw_info(images, prompts, selected_idx):
    print("\033[2J\033[H", end="")
    print("🎨 IMAGE TUI — RESULTS (Modular)")
    print("=" * 40)
    for i, img in enumerate(images):
        marker = "▶" if i == selected_idx else " "
        name = os.path.basename(img)
        print(f"{marker} [{i}] {name}")
    print("\n" + "=" * 40)
    print(f"SELECTED: {os.path.basename(images[selected_idx])}")
    print(f"PROMPT:   {prompts[selected_idx][:100]}...")
    print("\nUse arrows | Enter = NEW SESSION | P = full prompt | O = open | Q = quit")

def interactive_view(images, prompts):
    if not images:
        print("❌ No images to display")
        return
    selected = 0
    while True:
        draw_info(images, prompts, selected)
        key = readchar.readkey()
        if key == readchar.key.RIGHT or key == readchar.key.DOWN:
            selected = min(selected + 1, len(images)-1)
        elif key == readchar.key.LEFT or key == readchar.key.UP:
            selected = max(selected - 1, 0)
        elif key == readchar.key.ENTER:
            return "RESTART"
        elif key.lower() == "o":
            print(f"\n📂 Opening {images[selected]}...")
            os.system(f"xdg-open {images[selected]} > /dev/null 2>&1 &")
        elif key.lower() == "p":
            print("\033[2J\033[H", end="")
            print("FULL PROMPT:\n")
            print(prompts[selected])
            input("\nPress Enter to return...")
        elif key.lower() == "q":
            return "QUIT"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON execution target struct")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        conf = json.load(f)

    # Validate output dir
    os.makedirs(conf["out_dir"], exist_ok=True)
    MODEL_PATH = conf["model_path"]
    LORA_DATA = conf["lora_data"]
    CHAR_BASE = conf["char_base"]
    WIDTH, HEIGHT = conf["res"]
    OUTPUT_DIR = conf["out_dir"]
    PROMPT_MODE = conf["prompt_mode"]
    
    if conf["manual_prompts"]:
        MANUAL_PROMPTS = (conf["manual_prompts"]["pos"], conf["manual_prompts"]["neg"])
    else:
        MANUAL_PROMPTS = ("", "")

    MANUAL_SEED = conf["seed"]
    LOOP_COUNT = conf["loops"]
    ENABLE_UPSCALE = conf["enable_upscale"]
    CFG = conf["overrides"]["cfg"]
    STEPS = conf["overrides"]["steps"]
    U_STEPS = conf["overrides"]["u_steps"]

    ARCH = SDPipelineController.detect_arch(MODEL_PATH) if conf["model_type"] == "AUTO" else conf["model_type"]

    sampler_str = conf["sampler"]
    if ARCH == "FLUX":
        SAMPLER_CLASS = FlowMatchEulerDiscreteScheduler
    else:
        mapping = {
            "DPM++ 2M Karras": DPMSolverMultistepScheduler,
            "Euler Discrete": EulerDiscreteScheduler,
            "Euler Ancestral": EulerAncestralDiscreteScheduler,
            "Heun Discrete": HeunDiscreteScheduler,
            "LMS Discrete": LMSDiscreteScheduler,
            "KDPM2 Discrete": KDPM2DiscreteScheduler,
            "KDPM2 Ancestral": KDPM2AncestralDiscreteScheduler,
            "UniPC Multistep": UniPCMultistepScheduler,
        }
        SAMPLER_CLASS = mapping.get(sampler_str, EulerDiscreteScheduler)

    try:
        if PROMPT_MODE == "AUTO":
            pairs = get_prompts()
            print(f"\n🧠 Found {len(pairs)} prompt pairs")
            if not pairs:
                print("❌ No prompts found (AUTO)!")
                import sys
                sys.exit(1)
        else:
            pairs = [(None, None)]

        if ENABLE_UPSCALE:
            print(f"🧠 Loading pipelines for {LOOP_COUNT} loops (Sharing VRAM for upscale)...")
            base_pipe, actual_arch = SDPipelineController.load_base_pipeline(MODEL_PATH, LORA_DATA, None, ARCH)
            upscale_pipe = SDPipelineController.load_upscale_pipeline(base_pipe, actual_arch)
        else:
            print(f"🧠 Loading base pipeline for {LOOP_COUNT} loops (Upscale Skipped)...")
            base_pipe, actual_arch = SDPipelineController.load_base_pipeline(MODEL_PATH, LORA_DATA, None, ARCH)

        images = []
        prompts_list = []

        for loop_idx in range(1, LOOP_COUNT + 1):
            for i, (pos_path, neg_path) in enumerate(pairs, 1):
                if PROMPT_MODE == "AUTO":
                    print(f"\n🚀 [Loop {loop_idx}/{LOOP_COUNT}] Generating {i}/{len(pairs)}")
                    p_body = load_text(pos_path)
                    n_body = load_text(neg_path)
                else:
                    print(f"\n🚀 [Loop {loop_idx}/{LOOP_COUNT}] Generating Manual Input")
                    p_body, n_body = MANUAL_PROMPTS

                prompt = f"{CHAR_BASE}, {p_body}" if CHAR_BASE.strip() else p_body
                negative = f"{config.DEFAULT_NEG}, {n_body}" if n_body and n_body.strip() else config.DEFAULT_NEG

                seed = random.randint(0, 2**32) if MANUAL_SEED == "RANDOM" else MANUAL_SEED
                
                device = DeviceManager.get_device_type()
                generator = torch.Generator(device).manual_seed(seed) if device != "cpu" else torch.Generator().manual_seed(seed)

                gc.collect()
                if device == "cuda": torch.cuda.empty_cache()

                print(f"  - Stage 1 ({WIDTH}x{HEIGHT}) | CFG: {CFG} | Steps: {STEPS}...")
                if actual_arch == "FLUX":
                    img = base_pipe(
                        prompt=prompt,
                        width=WIDTH,
                        height=HEIGHT,
                        num_inference_steps=STEPS,
                        guidance_scale=CFG,
                        generator=generator
                    ).images[0]
                else:
                    img = base_pipe(
                        prompt=prompt,
                        negative_prompt=negative,
                        width=WIDTH,
                        height=HEIGHT,
                        num_inference_steps=STEPS,
                        guidance_scale=CFG,
                        generator=generator
                    ).images[0]

                if ENABLE_UPSCALE:
                    print(f"  - Stage 2 (Upscale {WIDTH*1.5:.01f}x{HEIGHT*1.5:.01f}) | Steps: {U_STEPS}...")
                    gc.collect()
                    if device == "cuda": torch.cuda.empty_cache()

                    upscale_factor = 1.5 if (WIDTH * HEIGHT) > 512*512 else 2.0
                    u_width, u_height = int(WIDTH * upscale_factor), int(HEIGHT * upscale_factor)

                    if actual_arch == "FLUX":
                        img = upscale_pipe(
                            prompt=prompt,
                            image=img.resize((u_width, u_height), Image.LANCZOS),
                            strength=config.DENOISE_STRENGTH,
                            num_inference_steps=U_STEPS,
                            guidance_scale=CFG,
                            generator=generator
                        ).images[0]
                    else:
                        img = upscale_pipe(
                            prompt=prompt,
                            negative_prompt=negative,
                            image=img.resize((u_width, u_height), Image.LANCZOS),
                            strength=config.DENOISE_STRENGTH,
                            num_inference_steps=U_STEPS,
                            guidance_scale=CFG,
                            generator=generator
                        ).images[0]

                path = f"{OUTPUT_DIR}/output_{i}_L{loop_idx}.png"
                img.save(path)

                meta_path = f"{OUTPUT_DIR}/output_{i}_L{loop_idx}.txt"
                with open(meta_path, "w") as f:
                    f.write(f"PROMPT: {prompt}\nNEGATIVE: {negative}\nSEED: {seed}\nCFG: {CFG}\nSTEPS: {STEPS}\nUPSCALE_STEPS: {U_STEPS}\nLOOP: {loop_idx}/{LOOP_COUNT}\nMODEL: {os.path.basename(MODEL_PATH)}\n")
                    if LORA_DATA:
                        for lp, lw in LORA_DATA:
                            f.write(f"LORA: {os.path.basename(lp)} (Weight: {lw})\n")

                images.append(path)
                prompts_list.append(prompt)

                gc.collect()
                if device == "cuda": torch.cuda.empty_cache()

        print("🧹 Cleaning up VRAM...")
        del base_pipe
        if ENABLE_UPSCALE: del upscale_pipe
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()

        status = interactive_view(images, prompts_list)
        if status == "RESTART":
            sys.exit(42)
        else:
            sys.exit(0)
        
    except torch.cuda.OutOfMemoryError:
        restore_terminal()
        vram = DeviceManager.get_vram_gb()
        print("\n" + "!"*40)
        print("❌ CUDA OUT OF MEMORY ERROR")
        print(f"The model or resolution is too large for your GPU ({vram:.1f}GB VRAM limit).")
        print("Try lowering the resolution or using a non-SDXL model.")
        print("!"*40 + "\n")
        sys.exit(1)
    except Exception as e:
        restore_terminal()
        print("\n" + "!"*40)
        print("❌ AN ERROR OCCURRED DURING GENERATION:")
        traceback.print_exc()
        print("!"*40 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
