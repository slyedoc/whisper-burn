import sys
import os
import urllib.request

from multiple_datasets.hub_default_utils import convert_hf_whisper 

if len(sys.argv) != 3:
    print("Usage: python3 script.py <huggingface_repo> <output_model_file>")
    sys.exit(1)
    
hf_repo = sys.argv[1]
out_file = sys.argv[2]

# Create parent directory if it doesn't exist
os.makedirs(os.path.dirname(out_file), exist_ok=True)

# Download and convert the model
convert_hf_whisper(hf_repo, out_file)

# Download the tokenizer.json directly from HuggingFace
model_dir = os.path.dirname(out_file)
tokenizer_url = f"https://huggingface.co/{hf_repo}/resolve/main/tokenizer.json"
tokenizer_path = os.path.join(model_dir, "tokenizer.json")

print(f"Downloading tokenizer from: {tokenizer_url}")
urllib.request.urlretrieve(tokenizer_url, tokenizer_path)
print(f"Tokenizer saved to {tokenizer_path}")