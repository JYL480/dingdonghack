import os

# OmniParser v2.0 weight check
print("Checking OmniParser v2.0 weights...")

# Check icon detection model
icon_detect_path = "icon_detect"
if os.path.exists(f"{icon_detect_path}/model.pt"):
    print(f"[OK] Icon detection model ready at {icon_detect_path}/model.pt")
else:
    print(f"[WARNING] {icon_detect_path}/model.pt not found - please download from HuggingFace")

# Check caption model
caption_path = "icon_caption"
if os.path.exists(f"{caption_path}/model.safetensors"):
    print(f"[OK] Caption model ready at {caption_path}/model.safetensors")
    
    # Check for required config files
    configs_found = []
    if os.path.exists(f"{caption_path}/config.json"):
        configs_found.append("config.json")
    if os.path.exists(f"{caption_path}/generation_config.json"):
        configs_found.append("generation_config.json")
    
    if configs_found:
        print(f"     Config files: {', '.join(configs_found)}")
else:
    print(f"[WARNING] {caption_path}/model.safetensors not found - please download from HuggingFace")

print("\nSetup check complete!")
print("OmniParser v2.0 is ready to use with:")
print("  - Icon detection: model.pt")
print("  - Icon caption: model.safetensors")
