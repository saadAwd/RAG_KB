"""
Main entry point for KB Retriever Gradio app.

Usage:
    python app.py                    # Run locally
    python app.py --server-name 0.0.0.0  # Allow external access
    python app.py --share            # Create public link
"""

from kb_retriever.gradio_retriever_test import main

if __name__ == "__main__":
    main()

