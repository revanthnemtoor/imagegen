# pipeline_manager.py
import os
import json
import torch
from safetensors import safe_open
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    FluxPipeline,
    FluxImg2ImgPipeline,
    FlowMatchEulerDiscreteScheduler
)
import config
from device_manager import DeviceManager

class SDPipelineController:
    """Manages the lifecycle of diffusers pipelines dynamically scaling to available hardware."""

    @staticmethod
    def detect_arch(path):
        """Scans model format headers to determine core architecture (FLUX/SDXL/SD1.5)."""
        if os.path.exists(config.MODEL_MAPPING_FILE):
            try:
                with open(config.MODEL_MAPPING_FILE, "r") as f:
                    mapping = json.load(f)
                    if path in mapping:
                        return mapping[path]
            except Exception:
                pass

        filename = os.path.basename(path).lower()
        if any(k in filename for k in ["flux", "schnell", "dev"]):
            return "FLUX"
        if any(k in filename for k in ["xl", "turbo", "lightning", "distill", "illustrious", "sdxl"]):
            return "SDXL"

        if path.endswith(".safetensors"):
            try:
                with safe_open(path, framework="pt") as f:
                    keys = f.keys()
                    if any("double_blocks" in k or "guidance_in" in k for k in keys):
                        return "FLUX"
                    if any("conditioner.embedders.1" in k for k in keys):
                        return "SDXL"
            except Exception:
                pass

        size_gb = os.path.getsize(path) / (1024**3)
        if size_gb > 15: return "FLUX"
        if size_gb > 6:  return "SDXL"
        return "SD15"

    @classmethod
    def load_base_pipeline(cls, path, lora_data=None, scheduler_class=None, forced_arch=None):
        """Builds the base Text2Image pipeline and calculates hardware bounds."""
        final_arch = forced_arch if (forced_arch and forced_arch != "AUTO") else cls.detect_arch(path)
        pipe = None

        if final_arch == "FLUX":
            print("🕒 Loading as FLUX.1 (Heavy!)...")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            try:
                pipe = FluxPipeline.from_single_file(
                    path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    ignore_mismatched_sizes=True
                )
            except Exception as e:
                print(f"⚠️ FLUX loading failed: {e}")

        elif final_arch == "SDXL":
            print("🕒 Loading as SDXL...")
            try:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    path, 
                    torch_dtype=torch.float16, 
                    use_safetensors=True,
                    config="stabilityai/stable-diffusion-xl-base-1.0",
                    low_cpu_mem_usage=True,
                    ignore_mismatched_sizes=True
                )
            except Exception as e:
                print(f"⚠️ SDXL loading failed: {e}")

        if pipe is None:
            print("🕒 Loading as Standard SD 1.5...")
            final_arch = "SD15"
            pipe = StableDiffusionPipeline.from_single_file(
                path, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

        # Apply LoRAs
        if lora_data:
            adapter_names, adapter_weights = [], []
            for i, (l_path, l_weight) in enumerate(lora_data):
                name = f"lora_{i}"
                print(f"📦 Applying LoRA: {os.path.basename(l_path)} (Weight: {l_weight})...")
                pipe.load_lora_weights(l_path, adapter_name=name)
                adapter_names.append(name)
                adapter_weights.append(l_weight)
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

        # Handle Scheduler
        if final_arch == "FLUX":
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler_class:
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)

        # Hardware Optimizations
        cls._apply_hardware_optimizations(pipe, final_arch)
        return pipe, final_arch

    @classmethod
    def load_upscale_pipeline(cls, base_pipe, arch):
        """Constructs an Img2Img pipeline inheriting resources directly from the base pipeline."""
        if arch == "FLUX":
            upscale_pipe = FluxImg2ImgPipeline.from_pipe(base_pipe)
        elif arch == "SDXL":
            upscale_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(base_pipe)
        else:
            upscale_pipe = StableDiffusionImg2ImgPipeline.from_pipe(base_pipe)

        cls._apply_hardware_optimizations(upscale_pipe, arch)
        return upscale_pipe

    @classmethod
    def _apply_hardware_optimizations(cls, pipe, arch):
        """Binds universal Diffusers hooks mapped dynamically out of DeviceManager."""
        pipe.enable_attention_slicing()
        
        strategy = DeviceManager.get_offload_strategy(arch)
        if strategy == "sequential":
            pipe.enable_sequential_cpu_offload() 
        else:
            pipe.enable_model_cpu_offload() 

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        try:
            pipe.vae.enable_slicing()
            pipe.enable_vae_tiling() 
        except Exception:
            pass
