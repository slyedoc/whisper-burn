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

fn save_whisper<B: Backend>(whisper: Whisper<B>, name: &str) -> Result<(), record::RecorderError> {
    DefaultRecorder::new().record(whisper.into_record(), name.into())
}

use std::env;

fn main() {
    let model_name = match env::args().nth(1) {
        Some(name) => name,
        None => {
            eprintln!("Model dump folder not provided");
            return;
        }
    };

    let device = WgpuDevice::default();

    let (whisper, whisper_config): (Whisper<Wgpu>, WhisperConfig) = match load_whisper(&model_name)
    {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error loading model {}: {}", model_name, e);
            return;
        }
    };

    println!("Saving model...");
    if let Err(e) = save_whisper(whisper, &model_name) {
        eprintln!("Error saving model {}: {}", model_name, e);
        return;
    }

    println!("Saving config...");
    if let Err(e) = whisper_config.save(&format!("{}.cfg", model_name)) {
        eprintln!("Error saving config for {}: {}", model_name, e);
        return;
    }

    println!("Finished.");
}
