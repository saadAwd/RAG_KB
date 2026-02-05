#!/bin/bash
# Automated script to build vector_db on GPU, verify CPU compatibility, and prepare for Git LFS upload

set -e  # Exit on error

echo "=========================================="
echo "Vector DB Build & Upload Automation"
echo "=========================================="

# Step 1: Build on GPU
echo ""
echo "[1/4] Building vector database on GPU..."
cd "$(dirname "$0")/.." || cd ~/RAG_KB || exit 1

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Build with GPU
python rebuild_vector_db.py --device cuda --force

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build completed on GPU"

# Step 2: Verify CPU compatibility
echo ""
echo "[2/4] Verifying CPU compatibility..."
CUDA_VISIBLE_DEVICES="" python -c "
from kb_retriever.vector_db import VectorDB
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
try:
    db = VectorDB.load(device='cpu')
    stats = db.get_stats()
    print(f'✅ CPU compatibility verified: {stats[\"total_chunks\"]} chunks')
except Exception as e:
    print(f'❌ CPU compatibility test failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ CPU compatibility test failed!"
    echo "   The vector_db built on GPU is not compatible with CPU."
    echo "   You may need to rebuild on CPU instead."
    exit 1
fi

echo "✅ CPU compatibility verified"

# Step 3: Prepare Git LFS
echo ""
echo "[3/4] Preparing Git LFS..."

# Initialize Git LFS if not already
git lfs install

# Track vector_db files
git lfs track "data/processed/vector_db/**" 2>/dev/null || true
git lfs track "data/processed/vector_db/**/*" 2>/dev/null || true

# Add .gitattributes
if [ -f .gitattributes ]; then
    git add .gitattributes
fi

echo "✅ Git LFS configured"

# Step 4: Check what will be uploaded
echo ""
echo "[4/4] Checking files to upload..."
echo ""
echo "Vector DB size:"
du -sh data/processed/vector_db/

echo ""
echo "Files tracked by Git LFS:"
git lfs ls-files | grep vector_db || echo "No files tracked yet"

echo ""
echo "=========================================="
echo "✅ Ready for upload!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the files above"
echo "2. Run: git add data/processed/vector_db/"
echo "3. Run: git commit -m 'Add CPU-compatible vector database'"
echo "4. Run: git push origin main"
echo ""
echo "Note: Upload may take 10-20 minutes due to large file size (~1.9GB)"
echo ""

