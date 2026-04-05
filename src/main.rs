use inquire::{Select, Text, MultiSelect, Confirm};
use serde::{Deserialize, Serialize};
use std::fs;
use std::collections::HashMap;
use std::process::Command;

#[derive(Serialize, Deserialize, Debug)]
struct GenerationOverrides {
    cfg: f32,
    steps: u32,
    u_steps: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct ManualPrompts {
    pos: String,
    neg: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ExecutionTarget {
    model_path: String,
    model_type: String,
    lora_data: Vec<(String, f32)>,
    char_base: String,
    res: (u32, u32),
    out_dir: String,
    sampler: String,
    prompt_mode: String,
    manual_prompts: Option<ManualPrompts>,
    seed: String,
    loops: u32,
    enable_upscale: bool,
    overrides: GenerationOverrides,
}

struct ModelItem {
    display: String,
    filename: String,
    arch: String,
}

fn load_mapping() -> HashMap<String, String> {
    if let Ok(data) = fs::read_to_string("model_mapping.json") {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        HashMap::new()
    }
}

fn save_mapping(mapping: &HashMap<String, String>) {
    if let Ok(data) = serde_json::to_string_pretty(mapping) {
        let _ = fs::write("model_mapping.json", data);
    }
}

fn load_lora_triggers() -> HashMap<String, String> {
    if let Ok(data) = fs::read_to_string("lora_triggers.json") {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        HashMap::new()
    }
}

fn save_lora_triggers(mapping: &HashMap<String, String>) {
    if let Ok(data) = serde_json::to_string_pretty(mapping) {
        let _ = fs::write("lora_triggers.json", data);
    }
}

fn detect_arch(path: &str, mapping: &HashMap<String, String>) -> (String, bool) {
    if let Some(arch) = mapping.get(path) {
        return (arch.clone(), true);
    }
    
    let p = path.to_lowercase();
    if p.contains("flux") || p.contains("schnell") || p.contains("dev") {
        return ("FLUX".to_string(), false);
    }
    if p.contains("xl") || p.contains("turbo") || p.contains("lightning") || p.contains("distill") || p.contains("sdxl") {
        return ("SDXL".to_string(), false);
    }
    
    if let Ok(metadata) = fs::metadata(path) {
        let size_gb = metadata.len() as f64 / 1024_f64.powi(3);
        if size_gb > 15.0 { return ("FLUX".to_string(), false); }
        if size_gb > 6.0 { return ("SDXL".to_string(), false); }
    }
    
    ("SD15".to_string(), false)
}

fn scan_models(mapping: &HashMap<String, String>) -> Vec<ModelItem> {
    let mut models = Vec::new();
    if let Ok(entries) = fs::read_dir("./models") {
        for entry in entries.flatten() {
            if let Ok(file_name) = entry.file_name().into_string() {
                if file_name.ends_with(".safetensors") || file_name.ends_with(".ckpt") {
                    let path = format!("./models/{}", file_name);
                    let (arch, overridden) = detect_arch(&path, mapping);
                    let mut display = format!("[{}] {}", arch, file_name);
                    if overridden {
                        display.push_str(" (Overridden)");
                    }
                    models.push(ModelItem { display, filename: file_name, arch });
                }
            }
        }
    }
    models.sort_by(|a, b| a.display.cmp(&b.display));
    models
}

fn scan_loras() -> Vec<String> {
    let mut loras = Vec::new();
    let _ = fs::create_dir_all("./lora");
    if let Ok(entries) = fs::read_dir("./lora") {
        for entry in entries.flatten() {
            if let Ok(file_name) = entry.file_name().into_string() {
                if file_name.ends_with(".safetensors") {
                    loras.push(file_name);
                }
            }
        }
    }
    loras.sort();
    loras
}

fn main() {
    let mut mapping = load_mapping();
    let mut lora_triggers = load_lora_triggers();

    loop {
        println!("\x1B[2J\x1B[1;1H");
        println!("🦀 HYBRID IMAGE GENERATOR (RUST COMMANDER)");
        println!("===============================================\n");

        let models = scan_models(&mapping);
        if models.is_empty() {
            println!("❌ No models found in ./models/");
            return;
        }

    let display_opts: Vec<String> = models.iter().map(|m| m.display.clone()).collect();
    let model_choice = Select::new("Select model:", display_opts).prompt();
    let model_idx = match model_choice {
        Ok(choice) => models.iter().position(|m| m.display == choice).unwrap(),
        Err(_) => return,
    };
    
    let selected_model = &models[model_idx];
    let model_path = format!("./models/{}", selected_model.filename);
    let detected_arch = selected_model.arch.clone();

    let types = vec![
        format!("Auto-detect (Current: {})", detected_arch), 
        "Flux.1 [dev/schnell]".to_string(), 
        "Stable Diffusion XL (SDXL)".to_string(), 
        "Stable Diffusion 1.5".to_string()
    ];
    let type_raw = Select::new("Model Architecture (Select to OVERRIDE):", types).prompt();
    let model_type = match type_raw {
        Ok(t) if t.contains("Flux.1") => "FLUX".to_string(),
        Ok(t) if t.contains("XL") => "SDXL".to_string(),
        Ok(t) if t.contains("1.5") => "SD15".to_string(),
        Ok(_) => "AUTO".to_string(),
        Err(_) => return,
    };

    let mut final_arch = detected_arch.clone();
    if model_type != "AUTO" {
        final_arch = model_type.clone();
        if let Ok(true) = Confirm::new(&format!("Save {} as {} in architecture mapping?", selected_model.filename, final_arch)).with_default(true).prompt() {
            mapping.insert(model_path.clone(), final_arch.clone());
            save_mapping(&mapping);
            println!("✅ Mapping saved.");
        }
    }

    let loras = scan_loras();
    let mut lora_data = Vec::new();
    let mut triggered_words = Vec::new();

    if !loras.is_empty() {
        if let Ok(selected_loras) = MultiSelect::new("Select LoRAs (Space to toggle, Enter to confirm):", loras).prompt() {
            for lora in selected_loras {
                let weight = Text::new(&format!("Weight for {lora} (Default 0.75):")).prompt();
                let w = if let Ok(mut weight_str) = weight {
                    weight_str = weight_str.trim().to_string();
                    if weight_str.is_empty() { 0.75 } else { weight_str.parse().unwrap_or(0.75) }
                } else {
                    0.75
                };
                lora_data.push((format!("./lora/{}", lora), w));

                if let Some(t) = lora_triggers.get(&lora) {
                    if !t.is_empty() {
                        triggered_words.push(t.clone());
                    }
                } else {
                    let tw = Text::new(&format!("Trigger word for {} (leave blank for none):", lora)).prompt().unwrap_or_default();
                    if !tw.trim().is_empty() {
                        lora_triggers.insert(lora.clone(), tw.clone());
                        save_lora_triggers(&lora_triggers);
                        triggered_words.push(tw);
                    } else {
                        // Save empty string so we don't ask again next time
                        lora_triggers.insert(lora.clone(), "".to_string());
                        save_lora_triggers(&lora_triggers);
                    }
                }
            }
        }
    }

    let mut char_base = Text::new("Character Base Prefix (or leave blank for presets default):").prompt().unwrap_or_default();

    // Dynamically append LoRA triggers
    if !triggered_words.is_empty() {
        let joined_triggers = triggered_words.join(", ");
        if char_base.is_empty() {
            char_base = joined_triggers;
        } else {
            char_base = format!("{}, {}", joined_triggers, char_base);
        }
    }

    let resolutions = vec!["YouTube Standard (16:9) - 1024x576", "Portrait (9:16) - 576x1024", "Square (1:1) - 1024x1024", "Classic (512x512)"];
    let res_str = Select::new("Aspect Ratio / Resolution:", resolutions).prompt();
    let res = match res_str {
        Ok(r) if r.starts_with("YouTube") => (1024, 576),
        Ok(r) if r.starts_with("Portrait") => (576, 1024),
        Ok(r) if r.starts_with("Classic") => (512, 512),
        Ok(_) => (1024, 1024),
        Err(_) => return,
    };

    let out_dir = Text::new("Output folder name:").with_default("output_run").prompt().unwrap_or_else(|_| "output_run".to_string());
    let _ = fs::create_dir_all(&format!("./{}", out_dir));

    let sd_schedulers = vec!["DPM++ 2M Karras", "Euler Discrete", "Euler Ancestral", "Heun Discrete", "LMS Discrete", "KDPM2 Discrete", "KDPM2 Ancestral", "UniPC Multistep"];
    let flux_schedulers = vec!["FlowMatch Euler"];
    let schedulers = if final_arch == "FLUX" { flux_schedulers } else { sd_schedulers };
    let sampler = Select::new("Select Sampler:", schedulers).prompt().unwrap_or_else(|_| "Euler Discrete");

    let pts_src = vec!["Auto (scan /prompts folder)", "Manual Entry"];
    let mode_str = Select::new("Prompt Source:", pts_src).prompt().unwrap_or_else(|_| "Auto");
    
    let (prompt_mode, manual_prompts) = if mode_str.starts_with("Manual") {
        let pos = Text::new("Positive Prompt:").prompt().unwrap_or_default();
        let neg = Text::new("Negative Prompt:").prompt().unwrap_or_default();
        ("MANUAL".to_string(), Some(ManualPrompts { pos, neg }))
    } else {
        ("AUTO".to_string(), None)
    };

    let seed_str = Text::new("Seed (Type a number or leave BLANK for random):").prompt().unwrap_or_default();
    let seed = if seed_str.trim().is_empty() || seed_str.to_lowercase() == "random" { "RANDOM".to_string() } else { seed_str.to_string() };

    let loops_str = Text::new("Number of loops:").with_default("1").prompt().unwrap_or_else(|_| "1".to_string());
    let loops = loops_str.parse().unwrap_or(1);

    let upscale_flag = Select::new("Enable Img2Img Upscale Layer? (Stage 2):", vec!["Yes (Higher Res)", "No (Faster execution)"]).prompt().unwrap_or_else(|_| "Yes (Higher Res)");
    let enable_upscale = upscale_flag.starts_with("Yes");

    let cfg_str = Text::new("Guidance Scale (CFG):").with_default("6.5").prompt().unwrap_or_else(|_| "6.5".to_string());
    let cfg = cfg_str.parse().unwrap_or(6.5);
    
    let steps_str = Text::new("Steps:").with_default("32").prompt().unwrap_or_else(|_| "32".to_string());
    let steps = steps_str.parse().unwrap_or(32);

    let u_steps_str = Text::new("Upscale Steps:").with_default("20").prompt().unwrap_or_else(|_| "20".to_string());
    let u_steps = u_steps_str.parse().unwrap_or(20);

    let overrides = GenerationOverrides { cfg, steps, u_steps };

    let target = ExecutionTarget {
        model_path: model_path.clone(),
        model_type: final_arch.clone(),
        lora_data,
        char_base,
        res,
        out_dir: format!("./{}", out_dir),
        sampler: sampler.to_string(),
        prompt_mode,
        manual_prompts,
        seed,
        loops,
        enable_upscale,
        overrides,
    };

    let json_data = serde_json::to_string_pretty(&target).unwrap();
    fs::write("execution_target.json", json_data).unwrap();

        println!("\x1B[2J\x1B[1;1H");
        println!("🚀 Launching PyTorch Generator via Headless Bridge...");
        
        let mut child = Command::new("python")
            .arg("main.py")
            .arg("--config")
            .arg("execution_target.json")
            .spawn()
            .expect("Failed to start python");
            
        let status = child.wait().unwrap();
        let code = status.code().unwrap_or(1);

        // Success Check
        if code == 0 || code == 42 {
            if model_type != "AUTO" {
                println!("\n❔ Custom architecture mapped during this session.");
                if let Ok(true) = Confirm::new(&format!("Permanentize {} as {} in model_mapping.json?", selected_model.filename, final_arch)).with_default(true).prompt() {
                    mapping.insert(model_path.clone(), final_arch.clone());
                    save_mapping(&mapping);
                    println!("✅ Mapping successfully locked!");
                }
            }
        }

        if code == 42 {
            continue;
        } else if code == 0 {
            break;
        } else {
            // Failure Branch (OOM or otherwise)
            println!("\n⚠️ Python backend failed. Check the error trace above.");
            if let Ok(true) = Confirm::new("Press Enter to restart session, or N to quit").with_default(true).prompt() {
                continue;
            } else {
                break;
            }
        }
    }
}
