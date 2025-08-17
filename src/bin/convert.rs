#![recursion_limit = "512"]

use whisper_stream::model::{load::*, *};

use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    config::Config,
    module::Module,
    record::{self, DefaultRecorder, Recorder},
    tensor::
        backend::Backend
    ,
};
use clap::Parser;


fn save_whisper<B: Backend>(whisper: Whisper<B>, name: &str) -> Result<(), record::RecorderError> {
    DefaultRecorder::new().record(whisper.into_record(), name.into())
}



#[derive(Parser)]
#[command(name = "transcribe")]
#[command(about = "Transcribe audio files using Whisper models")]
struct Args {
    
    /// Model name to use for transcription
    #[arg(short, long, default_value = "tiny_en")]
    model: String,

    #[arg(short, long, default_value = "models")]
    model_dir: String,
}

fn main() {
    println!("Whisper Model Converter");

    let args = Args::parse();

    let model_name = args.model.as_str();
    let model_dir = args.model_dir.as_str();

    let model_path = format!("{}/{}", model_dir, model_name);    

    let model_file = format!("{}/{}/model.mpk", model_dir, model_name);    
    //dbg!(load_path);


    let _device = WgpuDevice::default();

    let (whisper, whisper_config): (Whisper<Wgpu>, WhisperConfig) = match load_whisper(model_path.as_str())
    {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error loading model {}: {}", model_name, e);
            return;
        }
    };

    println!("Saving model: {}", model_file);
    if let Err(e) = save_whisper(whisper, &model_file) {
        eprintln!("Error saving model {}: {}", model_name, e);
        return;
    }
        
    let config_path = format!("{}/config.cfg", model_path);
    println!("Saving config: {}", config_path);
    if let Err(e) = whisper_config.save(&config_path) {
        eprintln!("Error saving config for {}: {}", model_name, e);
        return;
    }

    println!("Finished.");
}
