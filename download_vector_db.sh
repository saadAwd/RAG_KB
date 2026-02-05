#!/bin/bash
# Automated script to download vector_db from Git LFS on VM

set -e  # Exit on error

echo "=========================================="
echo "Downloading Vector DB from Git LFS"
echo "=========================================="

cd "$(dirname "$0")/.." || cd ~/RAG_KB || exit 1

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Step 1: Ensure Git LFS is installed
echo ""
echo "[1/3] Checking Git LFS..."
if ! command -v git-lfs &> /dev/null; then
    echo "Installing Git LFS..."
    sudo apt-get update
    sudo apt-get install -y git-lfs
fi

git lfs install

# Step 2: Pull with LFS
echo ""
echo "[2/3] Pulling vector database from Git LFS..."
echo "This may take 10-20 minutes due to large file size (~1.9GB)..."

git lfs pull

# Or if files are already in repo:
git pull origin main
git lfs pull

# Step 3: Verify
echo ""
echo "[3/3] Verifying vector database..."

python -c "
from kb_retriever.vector_db import VectorDB
try:
    db = VectorDB.load(device='cpu')
    stats = db.get_stats()
    print(f'✅ Vector database loaded successfully!')
    print(f'   Total chunks: {stats[\"total_chunks\"]}')
except Exception as e:
    print(f'❌ Error loading vector database: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS: Vector database is ready!"
    echo "=========================================="
    echo ""
    echo "You can now start the app:"
    echo "  sudo systemctl start kb-retriever"
    echo ""
else
    echo ""
    echo "❌ Verification failed. Check the error above."
    exit 1
fi

