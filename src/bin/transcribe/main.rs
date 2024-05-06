use std::collections::HashMap;
use std::iter;

use whisper_stream::helper::*;
use whisper_stream::model::*;
use whisper_stream::transcribe::waveform_to_text;
use whisper_stream::{token, token::Language};

use strum::IntoEnumIterator;

use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    config::Config,
    module::Module,
    record::{
        DefaultRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder, RecorderError,
    },
    tensor::{self, backend::Backend, Data, Float, Int, Tensor},
};
use num_traits::ToPrimitive;
use whisper_stream::audio::prep_audio;
use whisper_stream::token::{Gpt2Tokenizer, SpecialToken};
use std::{env, fs, process};
use hound::{self, SampleFormat};

fn load_audio_waveform<B: Backend>(filename: &str) -> hound::Result<(Vec<f32>, usize)> {
    let mut reader = hound::WavReader::open(filename)?;
    let spec = reader.spec();

    let duration = reader.duration() as usize;
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as usize;
    let bits_per_sample = spec.bits_per_sample;
    let sample_format = spec.sample_format;

    assert_eq!(sample_rate, 16000, "The audio sample rate must be 16k.");
    assert_eq!(channels, 1, "The audio must be single-channel.");

    let max_int_val = 2_u32.pow(spec.bits_per_sample as u32 - 1) - 1;

    let floats = match sample_format {
        SampleFormat::Float => reader.into_samples::<f32>().collect::<hound::Result<_>>()?,
        SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|s| s as f32 / max_int_val as f32))
            .collect::<hound::Result<_>>()?,
    };

    return Ok((floats, sample_rate));
}

fn main() {
    let tensor_device = WgpuDevice::default();

    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        eprintln!(
            "Usage: {} <model name> <audio file> <lang> <transcription file>",
            args[0]
        );
        process::exit(1);
    }

    let wav_file = &args[2];
    let text_file = &args[4];

    let lang_str = &args[3];
    let lang = match Language::iter().find(|lang| lang.as_str() == lang_str) {
        Some(lang) => lang,
        None => {
            eprintln!("Invalid language abbreviation: {}", lang_str);
            process::exit(1);
        }
    };

    let model_name = &args[1];

    println!("Loading waveform...");
    let (waveform, sample_rate) = match load_audio_waveform::<Wgpu>(wav_file) {
        Ok((w, sr)) => (w, sr),
        Err(e) => {
            eprintln!("Failed to load audio file: {}", e);
            process::exit(1);
        }
    };

    let (bpe, whisper_config, whisper) = load_model::<Wgpu>(&model_name, &tensor_device);

    let (text, tokens) = match waveform_to_text(&whisper, &bpe, lang, waveform, sample_rate, false) {
        Ok((text, tokens)) => (text, tokens),
        Err(e) => {
            eprintln!("Error during transcription: {}", e);
            process::exit(1);
        }
    };

    fs::write(text_file, text).unwrap_or_else(|e| {
        eprintln!("Error writing transcription file: {}", e);
        process::exit(1);
    });

    println!("Transcription finished.");
}

fn load_model<B: Backend>(
    model_name: &str,
    tensor_device_ref: &B::Device,
) -> (Gpt2Tokenizer, WhisperConfig, Whisper<B>) {
    let bpe = match Gpt2Tokenizer::new(model_name) {
        Ok(bpe) => bpe,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {}", e);
            process::exit(1);
        }
    };

    let whisper_config =
        match WhisperConfig::load(&format!("models/{}/{}.cfg", model_name, model_name)) {
            Ok(config) => config,
            Err(e) => {
                eprintln!("Failed to load whisper config: {}", e);
                process::exit(1);
            }
        };

    println!("Loading model...");
    let whisper: Whisper<B> = {
        match NamedMpkFileRecorder::<FullPrecisionSettings>::new()
            .load(
                format!("models/{}/{}", model_name, model_name).into(),
                tensor_device_ref,
            )
            .map(|record| whisper_config.init(tensor_device_ref).load_record(record))
        {
            Ok(whisper_model) => whisper_model,
            Err(e) => {
                eprintln!("Failed to load whisper model file: {}", e);
                process::exit(1);
            }
        }
    };

    let whisper = whisper.to_device(&tensor_device_ref);

    (bpe, whisper_config, whisper)
}