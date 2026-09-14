#!/usr/bin/env bash
# Build the zip to upload to Colab.
# Includes results/ and figures/ (13 MB) so the notebook's default 'resume'
# mode finds the shipped runs, exactly as a git clone would.
# Excludes .cache/ (150 MB of embeddings -- Colab rebuilds it faster than you
# can upload it).
set -euo pipefail
cd "$(dirname "$0")"
rm -f pe_toy_bundle.zip
zip -qr pe_toy_bundle.zip \
    pe_toy colab results figures spec.md README.md SUMMARY.md make_bundle.sh \
    -x '*__pycache__*' '*.pyc'
echo "wrote $(pwd)/pe_toy_bundle.zip ($(du -h pe_toy_bundle.zip | cut -f1))"
