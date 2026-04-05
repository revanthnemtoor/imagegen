#!/bin/bash

# fix_rocm_libs.sh
# Fixes "libamdhip64.so: cannot enable executable stack" on Arch/CachyOS

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🛠️  Fixing ROCm Binary Compatibility (Clear Execstack)...${NC}"

# Check for patchelf
if ! command -v patchelf &> /dev/null; then
    echo -e "${RED}❌ ERROR: 'patchelf' is not installed.${NC}"
    echo -e "Please run: ${YELLOW}sudo pacman -S patchelf${NC}"
    exit 1
fi

# Find and fix libraries
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}❌ ERROR: Virtual environment folder not found at ./$VENV_DIR${NC}"
    exit 1
fi

# Target specific torch libs known for this issue
TARGET_LIBS=$(find "$VENV_DIR" -name "*.so*" | grep -E "libamdhip64|libhiprtc|libhsa-runtime64|librccl")

if [ -z "$TARGET_LIBS" ]; then
    echo -e "${YELLOW}⚠️ No specific ROCm libraries found in $VENV_DIR. Scanning all .so files...${NC}"
    TARGET_LIBS=$(find "$VENV_DIR" -name "*.so*")
fi

COUNT=0
for lib in $TARGET_LIBS; do
    if [ -f "$lib" ]; then
        echo -e "  - Patching: ${CYAN}${lib}${NC}"
        patchelf --clear-execstack "$lib" 2>/dev/null
        ((COUNT++))
    fi
done

echo -e "${GREEN}✅ Successfully patched $COUNT libraries.${NC}"
echo -e "${YELLOW}You should now be able to run 'cargo run' on your AMD GPU.${NC}"
