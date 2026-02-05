"""
Knowledge base loaders for RAG.

This module provides utilities to read the processed JSONL files.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(THIS_DIR)

# Processed data directory
PROCESSED_DIR = os.path.join(PACKAGE_ROOT, "data", "processed")


def _iter_jsonl(path: str) -> Iterable[Dict]:
    """Yield JSON objects line-by-line from a JSONL file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [WARNING] Skipping invalid JSON at line {line_num} in {path}: {e}")
                    continue
    except UnicodeDecodeError as e:
        print(f"  [WARNING] UTF-8 decode error in {path}, trying with error handling...")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e2:
                    print(f"  [WARNING] Skipping invalid JSON at line {line_num} in {path}: {e2}")
                    continue


def _iter_kb_chunks(filename: str = "kb_chunks.jsonl") -> Iterable[Dict]:
    """Iterate over KB chunks from the chunks file."""
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"KB chunks file not found: {path}")
    return _iter_jsonl(path)

