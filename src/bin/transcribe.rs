
use whisper_stream::model::*;
use whisper_stream::transcribe::waveform_to_text;
use whisper_stream::token::Language;

use strum::IntoEnumIterator;
use clap::Parser;
use log::debug;

use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    config::Config,
    module::Module,
    record::{
        FullPrecisionSettings, NamedMpkFileRecorder, Recorder,
    },
    tensor::backend::Backend,
};
use whisper_stream::token::Gpt2Tokenizer;
use std::process;
use hound::{self, SampleFormat};

#[derive(Parser)]
#[command(name = "transcribe")]
#[command(about = "Transcribe audio files using Whisper models")]
struct Args {
    /// Audio file to transcribe
    audio_file: String,
    
    /// Model name to use for transcription
    #[arg(short, long, default_value = "tiny_en")]
    model: String,
    
    /// Language code (e.g., 'en', 'es', 'fr')
    #[arg(short, long, default_value = "en")]
    language: String,
}

fn load_audio_waveform<B: Backend>(filename: &str) -> hound::Result<(Vec<f32>, usize)> {
    let reader = hound::WavReader::open(filename)?;
    let spec = reader.spec();

    let _duration = reader.duration() as usize;
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as usize;
    let _bits_per_sample = spec.bits_per_sample;
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
    let args = Args::parse();
    let tensor_device = WgpuDevice::default();

    let lang = match Language::iter().find(|lang| lang.as_str() == &args.language) {
        Some(lang) => lang,
        None => {
            eprintln!("Invalid language abbreviation: {}", &args.language);
            process::exit(1);
        }
    };

    debug!("Loading waveform...");
    let (waveform, sample_rate) = match load_audio_waveform::<Wgpu>(&args.audio_file) {
        Ok((w, sr)) => (w, sr),
        Err(e) => {
            eprintln!("Failed to load audio file: {}", e);
            process::exit(1);
        }
    };

    let (bpe, _whisper_config, whisper) = load_model::<Wgpu>(&args.model, &tensor_device);

    let (text, _tokens) = match waveform_to_text(&whisper, &bpe, lang, waveform, sample_rate, false) {
        Ok((text, tokens)) => (text, tokens),
        Err(e) => {
            eprintln!("Error during transcription: {}", e);
            process::exit(1);
        }
    };

    println!("{}", text);
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

    debug!("Loading model...");
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