use anyhow::{Error as E, Result};
use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    config::Config,
    module::Module,
    record::{DefaultRecorder, FullPrecisionSettings, NamedMpkGzFileRecorder, Recorder, RecorderError},
    tensor::{self, backend::Backend, Data, Float, Int, Tensor},
};
use chrono::Local;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{self, SampleFormat};
use num_traits::ToPrimitive;
use std::{
    collections::HashMap,
    env,
    fs::OpenOptions,
    io,
    io::Write,
    iter, process,
    sync::{Arc, Mutex},
};
use strum::IntoEnumIterator;
use whisper_stream::{
    audio::prep_audio,
    helper::*,
    model::*,
    token,
    token::Language,
    token::{Gpt2Tokenizer, SpecialToken},
    transcribe::waveform_to_text,
};

fn main() {
    //COMMAND LINE
    let (model_name, wav_file, text_file, lang) = parse_args();

    let tensor_device = WgpuDevice::default();
    let (bpe, whisper_config, whisper) = load_model::<Wgpu>(&model_name, &tensor_device);

    //START AUDIO SERVER
    // Set up the input device and stream with the default input config.
    let audio_host = cpal::default_host();
    let audio_device = audio_host
        .default_input_device()
        .expect("Failed to get default input device");

    let audio_config = audio_device
        .default_input_config()
        .expect("Failed to get default input config");

    let channel_count = audio_config.channels() as usize;

    let audio_ring_buffer = Arc::new(Mutex::new(Vec::new()));
    let audio_ring_buffer_2 = audio_ring_buffer.clone();

    //loop to record the audio data forever (until the user stops the program)
    std::thread::spawn(move || loop {
        let data = record_audio(&audio_device, &audio_config, 1000).unwrap();
        audio_ring_buffer.lock().unwrap().extend_from_slice(&data);
        let max_len = data.len() * 4;
        let data_len = data.len();

        let mut audio_buffer = audio_ring_buffer.lock().unwrap();
        if audio_buffer.len() > max_len {
            let old_data_end = audio_buffer[audio_buffer.len() - data_len..].to_vec();
            *audio_buffer = Vec::new();
            audio_buffer.extend(old_data_end);
            audio_buffer.extend(data);
        }
    });

    // loop to process the audio data forever (until the user stops the program)
    println!("Transcribing audio...");
    let file = Arc::new(Mutex::new(
        OpenOptions::new()
            .append(true)
            .open("audio.txt")
            .expect("Failed to open output.txt"),
    ));
    for (i, _) in iter::repeat(()).enumerate() {
        std::thread::sleep(std::time::Duration::from_millis(2000));
        let data = audio_ring_buffer_2.lock().unwrap().clone();
        let audio_vectors: Vec<_> = data[..data.len() / channel_count as usize]
            .iter()
            .map(|v| *v as f32 / 32768.)
            .collect();

        //RUN INFERENCE
        let vector_length = audio_vectors.len();
        let (text, tokens) = match waveform_to_text(&whisper, &bpe, lang, audio_vectors, 16000) {
            Ok((text, tokens)) => (text, tokens),
            Err(e) => {
                eprintln!("Error during transcription: {}", e);
                process::exit(1);
            }
        };

        let output = format!(
            "{}, {}, {}, {}",
            i,
            text,
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            vector_length
        );
        println!("{}", output);

        let mut file = file.lock().unwrap();
        writeln!(file, "{}\n", output).unwrap_or_else(|e| {
            eprintln!("Error writing transcription file: {}", e);
            process::exit(1);
        });
    }
}

fn parse_args() -> (String, String, String, Language) {
    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        eprintln!(
            "Usage: {} <model name> <audio file> <lang> <transcription file>",
            args[0]
        );
        process::exit(1);
    }

    let model_name = args[1].clone();
    let wav_file = args[2].clone();
    let text_file = args[4].clone();

    let lang_str = &args[3];
    let lang = match Language::iter().find(|lang| lang.as_str() == lang_str) {
        Some(lang) => lang,
        None => {
            eprintln!("Invalid language abbreviation: {}", lang_str);
            process::exit(1);
        }
    };

    (model_name, wav_file, text_file, lang)
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

    println!("{:?}", std::env::current_dir());
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
        println!("{}", format!("models/{}/{}", model_name, model_name));
    
        match NamedMpkGzFileRecorder::<FullPrecisionSettings>::new()
            .load(
                format!("models/{}/{}", model_name, model_name).into(),
                tensor_device_ref,
            )
            .map(|record| whisper_config.init(tensor_device_ref).load_record(record)) {
            Ok(whisper_model) => whisper_model,
            Err(e) => {
                eprintln!("Failed to load whisper model file: {}", e);
                process::exit(1);
            }
        }
    };
    // let whisper: Whisper<B> = {
    //     println!("{}", format!("models/{}/{}", model_name, model_name));
    
    //     match DefaultRecorder::new()
    //         .load(
    //             format!("models/{}/{}", model_name, model_name).into(),
    //             tensor_device_ref,
    //         )
    //         .map(|record| whisper_config.init(tensor_device_ref).load_record(record)) {
    //         Ok(whisper_model) => whisper_model,
    //         Err(e) => {
    //             eprintln!("Failed to load whisper model file: {}", e);
    //             process::exit(1);
    //         }
    //     }
    // };

    let whisper = whisper.to_device(&tensor_device_ref);

    (bpe, whisper_config, whisper)
}

fn record_audio(
    audio_device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    milliseconds: u64,
) -> Result<Vec<i16>> {
    let writer = Arc::new(Mutex::new(Vec::new()));
    let writer_2 = writer.clone();
    let stream = audio_device.build_input_stream(
        &config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let processed = data
                .iter()
                .map(|v| (v * 32768.0) as i16)
                .collect::<Vec<i16>>();
            writer_2.lock().unwrap().extend_from_slice(&processed);
        },
        move |err| {
            eprintln!("an error occurred on stream: {}", err);
        },
        None,
    )?;
    stream.play()?;
    std::thread::sleep(std::time::Duration::from_millis(milliseconds));
    drop(stream);
    let data = writer.lock().unwrap().clone();
    let step = 3;
    let data: Vec<i16> = data.iter().step_by(step).copied().collect();
    Ok(data)
}
