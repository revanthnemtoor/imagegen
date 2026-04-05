import torch

class DeviceManager:
    _vram_cache = None
    _device_type = None

    @classmethod
    def get_device_type(cls):
        """Returns cuda, mps, or cpu dynamically."""
        if cls._device_type is not None:
            return cls._device_type
            
        if torch.cuda.is_available():
            cls._device_type = "cuda"
            # Log specific GPU name for diagnostics
            try:
                name = torch.cuda.get_device_name(0)
                print(f"🔍 Connected to: {name}")
            except Exception:
                pass
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cls._device_type = "mps"
        else:
            cls._device_type = "cpu"
            
        return cls._device_type

    @classmethod
    def get_vram_gb(cls):
        """Fetches the total VRAM available on the primary device."""
        if cls._vram_cache is not None:
            return cls._vram_cache

        device = cls.get_device_type()
        
        if device == "cuda":
            try:
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                cls._vram_cache = total_bytes / (1024**3)
            except Exception:
                # Fallback if properties fail
                cls._vram_cache = 0.0
        elif device == "mps":
            # MPS unified limits hard to fetch from basic torch attributes, assuming 8G
            cls._vram_cache = 8.0
        else:
            cls._vram_cache = 0.0

        return cls._vram_cache

    @classmethod
    def get_offload_strategy(cls, arch):
        """
        Dynamically returns the best VRAM optimization strategy string:
        'sequential' -> Layer-by-layer offload (for impossible models)
        'model'      -> Fully cached model offload (for fitting models)
        """
        vram = cls.get_vram_gb()
        
        if arch == "FLUX":
            # FLUX UNet is huge. Requires 16GB+ realistically to avoid sequential slicing
            return "sequential" if vram < 16.0 else "model"
        
        elif arch == "SDXL":
            # SDXL UNet is 5.2GB. Needs at least 6GB physical VRAM to sit cleanly in model_offload
            # Wait, earlier we found 3050 (4GB) could use model offload due to sys-mem fallback!
            # But the user just OOM'd! Therefore, if VRAM < 5.5, strictly enforce sequential.
            return "sequential" if vram < 5.5 else "model"

        else:
            # SD1.5 UNet is 1.7GB. Only needs sequential if running on an absolute potato
            return "sequential" if vram < 2.5 else "model"

    @classmethod
    def print_vram_warning(cls, required_gb):
        """Prints a targeted warning if the system cannot physically support the architecture."""
        vram = cls.get_vram_gb()
        
        # Check system RAM as well (12GB limit requested)
        import psutil
        sys_ram = psutil.virtual_memory().total / (1024**3)
        
        if vram < required_gb or sys_ram < 14.0:
            print("\n" + "!" * 40)
            print("⚠️ HARDWARE RESOURCES WARNING")
            if vram < required_gb:
                print(f"- GPU VRAM: {vram:.1f}GB (Architecture needs ~{required_gb:.1f}GB)")
            if sys_ram < 14.0:
                print(f"- System RAM: {sys_ram:.1f}GB (Tight! Recommend 16GB+)")
            
            print("\nEnabling high-efficiency low-VRAM mode...")
            print("!" * 40 + "\n")
