#!/usr/bin/env bash
# Build the code-only zip to upload to Colab.
# Excludes .cache/ (150 MB of embeddings -- Colab rebuilds it faster than you
# can upload it) and results/ (regenerated there).
set -euo pipefail
cd "$(dirname "$0")"
rm -f pe_toy_bundle.zip
zip -qr pe_toy_bundle.zip \
    pe_toy colab spec.md README.md SUMMARY.md make_bundle.sh \
    -x '*__pycache__*' '*.pyc'
echo "wrote $(pwd)/pe_toy_bundle.zip ($(du -h pe_toy_bundle.zip | cut -f1))"
