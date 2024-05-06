# Whisper Burn: Rust Implementation of OpenAI's Whisper Transcription Model

**Whisper Burn** is a Rust implementation of OpenAI's Whisper transcription model using the Rust deep learning framework, Burn.

## License

This project is licensed under the terms of the MIT license.

## Model Files

The OpenAI Whisper models that have been converted to work in burn are available in the whisper-burn space on Hugging Face. You can find them at [https://huggingface.co/Gadersd/whisper-burn](https://huggingface.co/Gadersd/whisper-burn).

If you have a custom fine-tuned model you can easily convert it to burn's format. Here is an example of converting OpenAI's tiny en model. The tinygrad dependency of the dump.py script should be installed from source not with pip.

```
# Download the tiny_en tokenizer
wget https://huggingface.co/Gadersd/whisper-burn/resolve/main/tiny_en/tokenizer.json

cd python
wget https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt
python3 dump.py tiny.en.pt tiny_en
mv tiny_en ../
cd ../
cargo run --release --bin convert tiny_en
```

However, if you want to convert a model from HuggingFace an extra conversion step is needed.

```
# Download the repo and convert it to .pt
python3 python/convert_huggingface_model.py openai/whisper-tiny tiny.pt

# Now it can be dumped
python3 python/dump.py tiny.pt tiny
cargo run --release --bin convert tiny

# Don't forget the tokenizer
wget https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json
```

#### 1. Clone the Repository

Clone the repository to your local machine using the following command:

```
git clone https://github.com/Gadersd/whisper-burn.git
```

Then, navigate to the project folder:

```
cd whisper-burn
```

#### 2. Download Whisper Tiny English Model

Use the following commands to download the Whisper tiny English model:

```
wget https://huggingface.co/Gadersd/whisper-burn/resolve/main/tiny_en/tiny_en.cfg
wget https://huggingface.co/Gadersd/whisper-burn/resolve/main/tiny_en/tiny_en.mpk.gz
wget https://huggingface.co/Gadersd/whisper-burn/resolve/main/tiny_en/tokenizer.json
```

#### 3. Run the Application

**Requirements**

- The audio file must be have a sample rate of 16k and be single-channel.
- `sox`. For Mac `brew install sox`

```
sox audio.wav -r 16000 -c 1 audio16k.wav
```
Now transcribe.

```
# this uses wgpu backend
cargo run --release --bin transcribe tiny_en audio16k.wav en transcription.txt
```

This usage assumes that "audio16k.wav" is the audio file you want to transcribe, and "tiny_en" is the model to use. Please adjust according to your specific needs.

Enjoy using **Whisper Burn**!

## Update as of 05/06/2024

This repository has been updated to use **Burn version 13**, which brings significant performance upgrades and many bug fixes for **wgpu**. As a result, the repository has been modified to use wgpu by default, as it should work on most machines regardless of the operating system or GPU type.

If you wish to swap out backends, you can easily do so by changing what is loaded in the `main.rs` file of the binary you are working with.

## New Binaries

We have added two new binaries, both aimed at (near) real-time transcription.

### Streaming Mode

To start the project in streaming mode, which takes in audio input from your microphone and transcribes it on the fly, run the following command:

```
cargo run --release --bin stream tiny en
```

### Real-Time Translation
We are also working on a new binary for real-time translation. This feature is currently a work in progress, so stay tuned for updates!

## Project File Structure

The project has a specific file structure that must be followed for the application to run correctly.

At the root of the project directory, there should be a `models` folder. This folder should contain various subfolders, each representing a different Whisper model. 

The name of each subfolder should match the name you want to pass into the `cargo run xxx` command. 

For example, your file structure may look like this:
```
.
├── models
│   ├── large-v2
│   ├── medium
│   ├── small
│   └── tiny
|       ├──tiny.cfg
|       ├──tiny.mpk
|       ├──tokenizer.json
├── python
│   ├── ...
├── src
│   ├── ...
├── Cargo.lock
├── Cargo.toml
├── README.md
```
If your file structure looks like this then you can run
```
cargo run --release --bin stream tiny en
```
where 'tiny' is the same name as one of the subfolders in the models folder.