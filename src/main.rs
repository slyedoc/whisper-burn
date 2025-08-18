#![recursion_limit = "512"]

pub mod audio;
pub mod beam;
pub mod helper;
pub mod model;
pub mod token;
pub mod transcribe;
use cfg_if::cfg_if;

use burn::{
    config::Config,
    module::Module,
    record::{DefaultRecorder, Recorder},
};
use clap::{Parser, Subcommand};
use std::process;
use strum::IntoEnumIterator;

use crate::{token::Language, transcribe::waveform_to_text};

#[derive(Parser)]
#[command(name = "whisper-burn")]
#[command(about = "Whisper speech recognition using Burn framework")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Transcribe audio files to text
    Transcribe {
        /// Audio file to transcribe
        audio_file: String,

        /// Model name to use for transcription
        #[arg(short, long, default_value = "tiny")]
        model: String,

        /// Model folder path
        #[arg(long, default_value = "models")]
        model_folder: String,

        /// Language code (e.g., 'en', 'es', 'fr')
        #[arg(short, long, default_value = "en")]
        language: String,
    },
    /// Convert model from PyTorch to Burn format
    Convert {
        /// Model name to convert
        #[arg(short, long, default_value = "tiny")]
        model: String,

        /// Model folder path  
        #[arg(long, default_value = "models")]
        model_folder: String,
    },
}

cfg_if!(
    if #[cfg(feature = "ndarray")] {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};
        type B = NdArray;
    } else if #[cfg(feature = "wgpu")] {
        use burn::backend::wgpu::{Wgpu, WgpuDevice};
        type B = Wgpu;
    } else if #[cfg(feature = "cuda")] {
        use burn::backend::cuda::{Cuda, CudaDevice};
        type B = Cuda;
    } else if #[cfg(feature = "tch-gpu")] {
        use burn::backend::libtorch::{LibTorch, LibTorchDevice};
        type B = LibTorch;
    } else if #[cfg(feature = "tch-cpu")] {
        use burn::backend::libtorch::{LibTorch, LibTorchDevice};
        type B = LibTorch;
    }
);

fn main() {
    let cli = Cli::parse();

    cfg_if!(
        if #[cfg(feature = "ndarray")] {
            println!("Using NdArray backend");
            let device = NdArrayDevice::default();
        } else if #[cfg(feature = "wgpu")] {
            println!("Using Wgpu backend");
            let device = WgpuDevice::default();
        } else if #[cfg(feature = "cuda")] {
            println!("Using Cuda backend");
            let device = CudaDevice::default();
        } else if #[cfg(feature = "tch-gpu")] {
            println!("Using LibTorch GPU backend");
            #[cfg(not(target_os = "macos"))]
            let device = LibTorchDevice::Cuda(0);
            #[cfg(target_os = "macos")]
            let device = LibTorchDevice::Mps;
        } else if #[cfg(feature = "tch-cpu")] {
            println!("Using LibTorch CPU backend");            
            let device = LibTorchDevice::Cpu;
        } else {
            panic!("No backend feature enabled. Please enable one of the backend features: ndarray, wgpu, cuda, tch-gpu, tch-cpu.");
        }
    );

    match cli.command {
        Commands::Transcribe {
            audio_file,
            model,
            model_folder,
            language,
        } => {
            let lang = match Language::iter().find(|lang| lang.as_str() == language) {
                Some(lang) => lang,
                None => {
                    eprintln!("Invalid language abbreviation: {}", &language);
                    process::exit(1);
                }
            };

            let (waveform, sample_rate) = match load_audio_waveform::<B>(&audio_file) {
                Ok((w, sr)) => (w, sr),
                Err(e) => {
                    eprintln!("Failed to load audio file: {}", e);
                    process::exit(1);
                }
            };

            let (bpe, _whisper_config, whisper) = load_model::<B>(&model, &model_folder, &device);
            let (text, _tokens) =
                match waveform_to_text(&whisper, &bpe, lang, waveform, sample_rate, false) {
                    Ok((text, tokens)) => (text, tokens),
                    Err(e) => {
                        eprintln!("Error during transcription: {}", e);
                        process::exit(1);
                    }
                };
            println!("{}", text);
        }
        Commands::Convert {
            model,
            model_folder,
        } => {
            let model_path = format!("{}/{}", model_folder, model);
            let config_file = format!("{}/config.json", model_path);
            let pytorch_model_path = format!("{}/model.pt", model_path);

            println!("Loading PyTorch model from: {}", pytorch_model_path);
            let (whisper, whisper_config) = model::load::load_extracted::<B>(&model_path).unwrap();

            // Save as Burn format
            let burn_model_path = format!("{}/model.mpk", model_path);
            println!("Saving Burn model to: {}", burn_model_path);
            DefaultRecorder::new()
                .record(whisper.into_record(), burn_model_path.into())
                .unwrap();

            println!("Saving config: {}", config_file);
            if let Err(e) = whisper_config.save(&config_file) {
                eprintln!("Error saving config for {}: {}", config_file, e);
                return;
            }

            println!("Model conversion completed successfully!");
        }
    }
}

fn load_audio_waveform<B: burn::tensor::backend::Backend>(
    filename: &str,
) -> hound::Result<(Vec<f32>, usize)> {
    use hound::SampleFormat;

    let reader = hound::WavReader::open(filename)?;
    let spec = reader.spec();

    let _duration = reader.duration() as usize;
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as usize;
    let _bits_per_sample = spec.bits_per_sample;

    let samples: Result<Vec<_>, _> = match spec.sample_format {
        SampleFormat::Float => reader.into_samples::<f32>().collect(),
        SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|s| s as f32 / i32::MAX as f32))
            .collect(),
    };

    let mut samples = samples?;

    if channels > 1 {
        samples = samples.into_iter().step_by(channels).collect();
    }

    let floats: Vec<f32> = samples;

    return Ok((floats, sample_rate));
}

fn load_model<B: burn::tensor::backend::Backend>(
    model_name: &str,
    model_dir: &str,
    device: &B::Device,
) -> (
    token::Gpt2Tokenizer,
    model::WhisperConfig,
    model::Whisper<B>,
) {
    use burn::{
        config::Config,
        module::Module,
        record::{FullPrecisionSettings, NamedMpkFileRecorder},
    };
    use model::*;
    use token::Gpt2Tokenizer;

    let model_path = format!("{}/{}", model_dir, model_name);

    // Load tokenizer
    dbg!(&model_path);
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    println!("Loading tokenizer from: {}", tokenizer_path);
    let bpe = Gpt2Tokenizer::new(&tokenizer_path).unwrap();

    // Load config
    let config_path = format!("{}/config.json", model_path);
    let whisper_config = WhisperConfig::load(&config_path).unwrap();

    // Load model
    let model_file_path = format!("{}/model.mpk", model_path);
    println!("Loading model: {}", model_file_path);
    let whisper: Whisper<B> = WhisperConfig::init(&whisper_config, device);
    let whisper = whisper
        .load_file(
            &model_file_path,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .unwrap();

    (bpe, whisper_config, whisper)
}
