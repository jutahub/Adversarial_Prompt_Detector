#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}LLM Attack Detector${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
}

print_menu() {
    echo "Select an action:"
    echo "  1) Install dependencies"
    echo "  2) Train model"
    echo "  3) Run prediction"
    echo "  4) Collect dataset"
    echo "  5) Dataset info"
    echo "  6) Visualize dataset"
    echo "  7) Full setup and training"
    echo "  8) Exit"
    echo
}

install_dependencies() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r "$PROJECT_DIR/requirements.txt"
    
    echo -e "${GREEN}Dependencies installed${NC}"
}

train_model() {
    echo -e "${YELLOW}Starting model training...${NC}"
    
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo -e "${RED}Virtual environment not found${NC}"
        echo "Install dependencies first (option 1)"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    python "$PROJECT_DIR/src/train.py"
    
    echo -e "${GREEN}Training completed${NC}"
}

run_prediction() {
    echo -e "${YELLOW}Starting prediction...${NC}"
    
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo -e "${RED}Virtual environment not found${NC}"
        echo "Install dependencies first (option 1)"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    python "$PROJECT_DIR/src/predict.py"
}

collect_dataset() {
    echo -e "${YELLOW}Starting dataset collection...${NC}"
    
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo -e "${RED}Virtual environment not found${NC}"
        echo "Install dependencies first (option 1)"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    python "$PROJECT_DIR/src/collect_data.py"
}

dataset_info() {
    echo -e "${YELLOW}Loading dataset info...${NC}"
    
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo -e "${RED}Virtual environment not found${NC}"
        echo "Install dependencies first (option 1)"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    python "$PROJECT_DIR/src/dataset_info.py"
}

visualize_dataset() {
    echo -e "${YELLOW}Starting dataset visualization...${NC}"
    
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo -e "${RED}Virtual environment not found${NC}"
        echo "Install dependencies first (option 1)"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    python "$PROJECT_DIR/src/visualize_data.py"
}

full_setup() {
    echo -e "${YELLOW}Full setup and training...${NC}"
    install_dependencies
    train_model
}

print_header

if [ $# -gt 0 ]; then
    case "$1" in
        install)
            install_dependencies
            exit 0
            ;;
        train)
            train_model
            exit 0
            ;;
        predict)
            run_prediction
            exit 0
            ;;
        collect)
            collect_dataset
            exit 0
            ;;
        info)
            dataset_info
            exit 0
            ;;
        visualize)
            visualize_dataset
            exit 0
            ;;
        full)
            full_setup
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            echo "Available commands: install, train, predict, collect, info, visualize, full"
            exit 1
            ;;
    esac
fi

while true; do
    print_menu
    read -p "Enter number (1-8): " choice
    
    case $choice in
        1)
            install_dependencies
            ;;
        2)
            train_model
            ;;
        3)
            run_prediction
            ;;
        4)
            collect_dataset
            ;;
        5)
            dataset_info
            ;;
        6)
            visualize_dataset
            ;;
        7)
            full_setup
            ;;
        8)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
done
