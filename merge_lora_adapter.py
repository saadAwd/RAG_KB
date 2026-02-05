#!/usr/bin/env python3
"""
Script to merge LoRA adapter into base model for GGUF conversion.

Usage:
    python merge_lora_adapter.py
"""

import os
import sys
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
BASE_MODEL = "humain-ai/ALLaM-7B-Instruct-preview"
ADAPTER_PATH = Path("./Model/Allam7B-Physiology-RAG-finetuned-final")
OUTPUT_PATH = Path("./Model/Allam7B-Physiology-RAG-finetuned-merged")

def main():
    print("=" * 60)
    print("LoRA Adapter Merger for GGUF Conversion")
    print("=" * 60)
    
    # Check if adapter exists
    if not ADAPTER_PATH.exists():
        print(f"❌ Error: Adapter path not found: {ADAPTER_PATH}")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 Step 1: Loading base model: {BASE_MODEL}")
    print("   This may take a few minutes...")
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("   ✅ Base model loaded")
    except Exception as e:
        print(f"   ❌ Error loading base model: {e}")
        sys.exit(1)
    
    print(f"\n🔗 Step 2: Loading LoRA adapter from {ADAPTER_PATH}")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            str(ADAPTER_PATH),
        )
        print("   ✅ Adapter loaded")
    except Exception as e:
        print(f"   ❌ Error loading adapter: {e}")
        sys.exit(1)
    
    print(f"\n🔀 Step 3: Merging adapter into base model...")
    print("   This may take 5-10 minutes...")
    try:
        merged_model = model.merge_and_unload()
        print("   ✅ Adapter merged")
    except Exception as e:
        print(f"   ❌ Error merging adapter: {e}")
        sys.exit(1)
    
    print(f"\n💾 Step 4: Saving merged model to {OUTPUT_PATH}")
    try:
        merged_model.save_pretrained(str(OUTPUT_PATH))
        print("   ✅ Model saved")
    except Exception as e:
        print(f"   ❌ Error saving model: {e}")
        sys.exit(1)
    
    print(f"\n📝 Step 5: Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(ADAPTER_PATH),
            trust_remote_code=True
        )
        tokenizer.save_pretrained(str(OUTPUT_PATH))
        print("   ✅ Tokenizer saved")
    except Exception as e:
        print(f"   ❌ Error saving tokenizer: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Merged model saved to:")
    print(f"   {OUTPUT_PATH.absolute()}")
    print("\n📋 Next steps:")
    print("   1. Convert to GGUF using llama.cpp:")
    print("      python convert-hf-to-gguf.py \\")
    print(f"        {OUTPUT_PATH} \\")
    print(f"        --outfile ./Model/Allam7B-Physiology-RAG-finetuned.gguf \\")
    print("        --outtype q4_k_m")
    print("   2. Update generator to use GGUF format")
    print("=" * 60)

if __name__ == "__main__":
    main()
