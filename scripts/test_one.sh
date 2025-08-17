uv run scripts/convert_huggingface_model.py openai/whisper-tiny models/tiny/model.pt
uv run scripts/dump.py models/tiny/model.pt models/tiny
cargo run --release --bin convert -- --model tiny
cargo run --release --bin transcribe -- --model tiny audio16k.wav 
