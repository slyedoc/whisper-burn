#!/bin/bash

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 tiny"
    echo "Example: $0 tiny.en"
    exit 1
fi

MODEL_NAME=$1

echo "=== Step 1: Converting HuggingFace model ==="
start_time=$(date +%s.%3N)
uv run scripts/convert_huggingface_model.py openai/whisper-${MODEL_NAME} models/${MODEL_NAME}/${MODEL_NAME}.pt
end_time=$(date +%s.%3N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo "Step 1 completed in ${elapsed}s"

echo "=== Step 2: Dumping model weights ==="
start_time=$(date +%s.%3N)
uv run scripts/dump.py models/${MODEL_NAME}/${MODEL_NAME}.pt models/${MODEL_NAME}
end_time=$(date +%s.%3N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo "Step 2 completed in ${elapsed}s"

echo "=== Step 3: Converting to Burn format ==="
start_time=$(date +%s.%3N)
cargo run --release --bin convert -- --model ${MODEL_NAME}
end_time=$(date +%s.%3N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo "Step 3 completed in ${elapsed}s"

echo "=== Step 4: Testing transcription ==="
start_time=$(date +%s.%3N)
cargo run --release --bin transcribe -- --model ${MODEL_NAME} audio16k.wav
end_time=$(date +%s.%3N)
elapsed=$(echo "$end_time - $start_time" | bc)
echo "Step 4 completed in ${elapsed}s"

echo "=== All steps completed for model: ${MODEL_NAME} ==="
