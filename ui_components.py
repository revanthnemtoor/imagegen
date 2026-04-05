import os
import random
import json
import questionary
from prompt_toolkit.key_binding import KeyBindings
import config

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

class TUIManager:
    """Encapsulates all Terminal User Interface elements and logic routing."""

    def __init__(self):
        self.kb = KeyBindings()
        self._setup_bindings()

    def _setup_bindings(self):
        @self.kb.add("backspace")
        def _(event):
            if hasattr(event.app, "current_buffer") and event.app.current_buffer.text:
                event.app.current_buffer.delete_before_cursor()
            else:
                event.app.exit(result="BACK")

    def select_model(self, models):
        choice = questionary.select(
            "Select model:",
            choices=[f"[{i}] {m}" for i, m in enumerate(models)] + ["[Back]"]
        ).ask()
        
        if choice in [None, "[Back]"]: return "BACK"
        if choice == "QUIT": return "QUIT"
        
        idx = int(choice.split("]")[0][1:])
        return os.path.join(config.MODEL_DIR, models[idx])

    def select_resolution(self):
        choice = questionary.select(
            "Aspect Ratio / Resolution:",
            choices=[
                "YouTube Standard (16:9) - 1024x576",
                "Portrait (9:16) - 576x1024",
                "Square (1:1) - 1024x1024",
                "Classic (512x512)",
                "Manual",
                "[Back]"
            ]
        ).ask()

        if choice is None: return "QUIT"
        if choice == "[Back]": return "BACK"

        if choice == "Manual":
            val = questionary.text("Enter resolution (e.g. 1024 576):", key_bindings=self.kb).ask()
            if val is None: return "QUIT"
            if val == "BACK": return "BACK"
            try:
                w, h = map(int, val.split())
                return w, h
            except Exception:
                return 1024, 1024

        mapping = {
            "YouTube Standard (16:9) - 1024x576": (1024, 576),
            "Portrait (9:16) - 576x1024": (576, 1024),
            "Square (1:1) - 1024x1024": (1024, 1024),
            "Classic (512x512)": (512, 512)
        }
        return mapping[choice]

    def select_model_type(self):
        choice = questionary.select(
            "Model Architecture:",
            choices=[
                "Auto-detect (Recommended)",
                "Flux.1 [dev/schnell]",
                "Stable Diffusion XL (SDXL)",
                "Stable Diffusion 1.5",
                "[Back]"
            ]
        ).ask()
        
        if choice is None: return "QUIT"
        if choice == "[Back]": return "BACK"
        
        if "Flux" in choice: return "FLUX"
        if "XL" in choice: return "SDXL"
        if "1.5" in choice: return "SD15"
        return "AUTO"

    def get_seed(self):
        val = questionary.text("Seed (Type a number or leave BLANK for random):").ask()
        if val is None: return "QUIT"
        if val == "BACK": return "BACK"
        
        val = val.strip().lower()
        if not val or val == "random":
            return "RANDOM"
        try:
            return int(val)
        except Exception:
            print("⚠️ Invalid seed number. Using random.")
            return "RANDOM"

    def select_loras(self, loras):
        if not loras: return []
        
        selected = questionary.checkbox(
            "Select LoRAs (Space to toggle, Enter to confirm):",
            choices=[f"{l}" for l in loras]
        ).ask()
        
        if selected is None: return "QUIT"
        if not selected: return []
        
        results = []
        for lora in selected:
            weight_val = questionary.text(f"Weight for {lora} (Default 0.75):", default="0.75", key_bindings=self.kb).ask()
            if weight_val is None: return "QUIT"
            if weight_val == "BACK": return "BACK"
            
            weight = float(weight_val) if weight_val.strip() else 0.75
            results.append((os.path.join(config.LORA_DIR, lora), weight))
        return results

    def get_character_base(self):
        presets = {}
        if os.path.exists(config.PRESETS_FILE):
            try:
                with open(config.PRESETS_FILE, "r") as f:
                    presets = json.load(f)
            except Exception:
                pass

        print("\n💡 Define your Consistent Character (e.g. 'A young Maurya warrior, black hair')")
        ps_choices = ["[New Character Description]", "[Back]"] + sorted(list(presets.keys()))
        choice = questionary.select("Select from Presets or Enter New:", choices=ps_choices).ask()

        if choice is None: return "QUIT"
        if choice == "[Back]": return "BACK"

        if choice == "[New Character Description]":
            char_base = questionary.text("Character Base Prefix:", key_bindings=self.kb).ask()
            if char_base is None: return "QUIT"
            if char_base == "BACK": return "BACK"
            
            save = questionary.confirm(f"Save as preset?").ask()
            if save is None: return "QUIT"

            if save:
                name = questionary.text("Preset name (e.g. 'Maurya Ruler'):", key_bindings=self.kb).ask()
                if name is None: return "QUIT"
                if name == "BACK": return "BACK"
                presets[name] = char_base
                with open(config.PRESETS_FILE, "w") as f:
                    json.dump(presets, f, indent=4)
                print(f"✅ Preset '{name}' saved.")
            return char_base
        
        return presets[choice]

    def select_output_folder(self):
        name = questionary.text("Output folder name:", default="output_run", key_bindings=self.kb).ask()
        if name is None: return "QUIT"
        if name == "BACK": return "BACK"
        
        path = f"./{name}"
        os.makedirs(path, exist_ok=True)
        return path

    def select_sampler(self, arch):
        sd_schedulers = {
            "DPM++ 2M Karras": DPMSolverMultistepScheduler,
            "Euler Discrete": EulerDiscreteScheduler,
            "Euler Ancestral": EulerAncestralDiscreteScheduler,
            "Heun Discrete": HeunDiscreteScheduler,
            "LMS Discrete": LMSDiscreteScheduler,
            "KDPM2 Discrete": KDPM2DiscreteScheduler,
            "KDPM2 Ancestral": KDPM2AncestralDiscreteScheduler,
            "UniPC Multistep": UniPCMultistepScheduler,
        }
        
        flux_schedulers = {
            "FlowMatch Euler": FlowMatchEulerDiscreteScheduler,
        }

        print(f"\n✨ Samplers for {arch} model:")
        if arch == "FLUX":
            choices = list(flux_schedulers.keys()) + ["[Back]"]
            mapping = flux_schedulers
        else:
            choices = list(sd_schedulers.keys()) + ["[Back]"]
            mapping = sd_schedulers
        
        choice = questionary.select("Select Sampler:", choices=choices).ask()
        
        if choice is None: return "QUIT"
        if choice == "[Back]": return "BACK"
        return mapping[choice]

    def get_prompt_input(self):
        choice = questionary.select("Prompt Source:", choices=["Auto (scan /prompts folder)", "Manual Entry", "[Back]"]).ask()
        
        if choice is None: return "QUIT"
        if choice == "[Back]": return "BACK"
        
        if choice == "Manual Entry":
            print("\n💡 Tip: Leave BLANK for empty string. Type 'black' for session default.")
            pos = questionary.text("Positive Prompt:", key_bindings=self.kb).ask()
            if pos is None: return "QUIT"
            if pos == "BACK": return "BACK"
            pos = "" if not pos else pos
            
            neg = questionary.text("Negative Prompt:", key_bindings=self.kb).ask()
            if neg is None: return "QUIT"
            if neg == "BACK": return "BACK"
            neg = "" if not neg else neg
            
            return "MANUAL", (pos, neg)
        
        return "AUTO", None

    def get_overrides(self):
        print("\n⚙️  Generation Overrides (BLANK=0/None | 'black'=Default Constant)")
        
        cfg_val = questionary.text(f"Guidance Scale (Typing 'black' uses {config.GUIDANCE_SCALE}):", key_bindings=self.kb).ask()
        if cfg_val is None: return "QUIT"
        if cfg_val == "BACK": return "BACK"
        cfg = config.GUIDANCE_SCALE if cfg_val.strip().lower() == 'black' else (float(cfg_val) if cfg_val.strip() else 1.0)
        
        steps_val = questionary.text(f"Steps (Typing 'black' uses {config.NUM_STEPS_BASE}):", key_bindings=self.kb).ask()
        if steps_val is None: return "QUIT"
        if steps_val == "BACK": return "BACK"
        steps = config.NUM_STEPS_BASE if steps_val.strip().lower() == 'black' else (int(steps_val) if steps_val.strip() else 20)
        
        upscale_steps_val = questionary.text(f"Upscale Steps (Typing 'black' uses {config.NUM_STEPS_UPSCALE}):", key_bindings=self.kb).ask()
        if upscale_steps_val is None: return "QUIT"
        if upscale_steps_val == "BACK": return "BACK"
        u_steps = config.NUM_STEPS_UPSCALE if upscale_steps_val.strip().lower() == 'black' else (int(upscale_steps_val) if upscale_steps_val.strip() else 10)
        
        return cfg, steps, u_steps

    def get_loop_count(self):
        val = questionary.text("Number of loops (Repeat entire prompt set):", default="1", key_bindings=self.kb).ask()
        if val is None: return "QUIT"
        if val == "BACK": return "BACK"
        try:
            return max(1, int(val))
        except Exception:
            return 1

    def get_upscale_choice(self):
        choice = questionary.select(
            "Enable Img2Img Upscale Layer? (Stage 2):",
            choices=["Yes (Higher Res)", "No (Faster execution)", "[Back]"]
        ).ask()
        
        if choice is None: return "QUIT"
        if choice == "[Back]": return "BACK"
        return True if "Yes" in choice else False
