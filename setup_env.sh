#!/bin/bash

# setup_env.sh
# Automated Environment Setup for Hybrid Image Generator

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}==============================================${NC}"
echo -e "${CYAN}   🚀 HYBRID IMAGE GEN ENVIRONMENT SETUP      ${NC}"
echo -e "${CYAN}==============================================${NC}"

# 1. Virtual Environment Check
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${RED}❌ ERROR: No active virtual environment detected.${NC}"
    echo -e "${YELLOW}Please activate your venv first, then run this script again.${NC}"
    echo -e "Example: source venv/bin/activate"
    exit 1
fi
echo -e "${GREEN}✅ Virtual Environment active: ${VIRTUAL_ENV}${NC}"

# 2. Python Version Check
PY_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PY_VER" != "3.11" ]]; then
    echo -e "${RED}❌ ERROR: Python 3.11 is required (Detected: ${PY_VER}).${NC}"
    echo -e "${YELLOW}PyTorch ROCm binaries are strictly optimized for Python 3.11.${NC}"
    echo -e "Please recreate your venv with python 3.11."
    exit 1
fi
echo -e "${GREEN}✅ Python version ${PY_VER} confirmed.${NC}"

# 3. Hardware Detection
GPU_TYPE="UNKNOWN"
if lspci | grep -i "nvidia" > /dev/null; then
    GPU_TYPE="NVIDIA"
elif lspci | grep -i "amd" > /dev/null || lspci | grep -i "radeon" > /dev/null; then
    GPU_TYPE="RADEON"
fi

echo -e "${CYAN}🔍 Detected Hardware: ${GPU_TYPE}${NC}"

# 4. Installation Loop
case $GPU_TYPE in
    "NVIDIA")
        echo -e "${YELLOW}📦 Installing NVIDIA CUDA Dependencies...${NC}"
        pip install --upgrade pip
        pip install -r requirements_nvidia.txt
        ;;
    "RADEON")
        echo -e "${YELLOW}📦 Installing AMD RADEON ROCm Dependencies...${NC}"
        pip install --upgrade pip
        pip install -r requirements_radeon.txt
        ;;
    *)
        echo -e "${RED}⚠️ UNKNOWN GPU DETECTED.${NC}"
        echo -e "Please select manually:"
        options=("NVIDIA CUDA" "AMD RADEON" "CPU ONLY / QUIT")
        select opt in "${options[@]}"
        do
            case $opt in
                "NVIDIA CUDA")
                    pip install -r requirements_nvidia.txt
                    break
                    ;;
                "AMD RADEON")
                    pip install -r requirements_radeon.txt
                    break
                    ;;
                *)
                    echo "Setup aborted."
                    exit 1
                    break
                    ;;
            esac
        done
        ;;
esac

echo -e "\n${GREEN}✨ SETUP COMPLETE!${NC}"
echo -e "${CYAN}Try running: cargo run${NC}"
