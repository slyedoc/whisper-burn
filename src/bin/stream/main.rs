use anyhow::{Error as E, Result};
use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    config::Config,
    module::Module,
    record::{
        DefaultRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder, RecorderError,
    },
    tensor::{self, backend::Backend, Data, Float, Int, Tensor},
};
use chrono::Local;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{self, SampleFormat};
use num_traits::ToPrimitive;
use std::{
    collections::{HashMap, VecDeque},
    env,
    fs::{OpenOptions, File},
    io,
    io::Write,
    iter, process,
    sync::{Arc, Mutex, Condvar},
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
use webrtc_vad::{Vad, VadMode, SampleRate};
use dasp::signal::{self, Signal};
use dasp::interpolate::linear::Linear;
use dasp::ring_buffer::Bounded;
fn main() {
    //COMMAND LINE
    let (model_name, wav_file, text_file, lang) = parse_args();

    let tensor_device = WgpuDevice::default();
    let (bpe, whisper_config, whisper) = load_model::<Wgpu>(&model_name, &tensor_device);

    let file = Arc::new(Mutex::new(
        OpenOptions::new()
            .append(true)
            .open("audio.txt")
            .expect("Failed to open output.txt"),
    ));

    let audio_queue_and_notifier = Arc::new((Mutex::new(VecDeque::<f32>::new()), Condvar::new()));


    let audio_queue_and_notifier1 = Arc::clone(&audio_queue_and_notifier);
    std::thread::spawn(move || {
        record_audio(audio_queue_and_notifier1)
    });


    let audio_queue_and_notifier2 = Arc::clone(&audio_queue_and_notifier);
    std::thread::spawn(move || {
        process_audio_data(audio_queue_and_notifier2, file, whisper, bpe, lang);
    });

    loop {
        std::thread::sleep(std::time::Duration::from_millis(1000));
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

fn record_audio(
    audio_queue_and_notifier: Arc<(Mutex<VecDeque<f32>>, Condvar)>,
) {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("Failed to get default input device");
    let config = device.default_input_config().expect("Failed to get default input config");
    let sample_rate = config.sample_rate().0 as f32;

    // Create a stream with the default input format
    let stream = device.build_input_stream(
        &config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            handle_input_data(data, &sample_rate);
        },
        move |err| eprintln!("Error: {}", err),
        None
    ).expect("Failed to build input stream");

    // Play the stream
    stream.play().expect("Failed to play stream");




    // let mut is_speech = false;
    // let stream = audio_device.build_input_stream(
    //     &audio_config.config(),
    //     move |data: &[f32], _: &cpal::InputCallbackInfo| {
    //         println!("{:?}", &data);
    //         let (lock, cvar) = &*audio_queue_and_notifier;
    //         let mut writer = lock.lock().unwrap();

    //         // Clone and convert data to i16 for VAD
    //         let vad_data_i16: Vec<i16> = data.iter().map(|x| *x as i16).collect();
    //         let mut vad = Vad::new_with_rate_and_mode(SampleRate::Rate48kHz, VadMode::Quality);
    //         let is_speech_segment_result = vad.is_voice_segment(&vad_data_i16);
    //         let is_speech_segment = match is_speech_segment_result {
    //             Ok(s) => s,
    //             Err(e) => {
    //                 eprintln!("Failed to check if voice segment: {:?}", e);
    //                 return;
    //             }
    //         };

    //         //println!("{:?}", is_speech_segment);
    //         if is_speech_segment {
    //             if is_speech {
    //                 for &sample in data {
    //                     writer.push_back(sample);
    //                 }
    //                 cvar.notify_one(); // Notify the inference thread that there's data
    //             } else {
    //                 is_speech = true;
    //             }
    //         } else if is_speech {
    //             is_speech = false;
    //         }
    //     },
    //     move |err| {
    //         eprintln!("an error occurred on stream: {}", err);
    //     },
    //     None,
    // ).expect("Failed to build input stream.");
    // stream.play();
    // loop {
    //     println!("hello from loop2");
    //     std::thread::sleep(std::time::Duration::from_millis(1000));
    // }
}

fn process_audio_data(
    audio_queue_and_notifier: Arc<(Mutex<VecDeque<f32>>, Condvar)>,
    file: Arc<Mutex<File>>,
    whisper: Whisper<Wgpu>,
    bpe: Gpt2Tokenizer,
    lang: Language,
) {
    for (i, _) in iter::repeat(()).enumerate() {
        let (lock, cvar) = &*audio_queue_and_notifier;
        let mut audio_data_vectors = lock.lock().unwrap();
        while audio_data_vectors.is_empty() {
            audio_data_vectors = cvar.wait(audio_data_vectors).unwrap();
        }
        let processed_len = audio_data_vectors.len();
        let mut audio_data_vectors_clone_for_inference: Vec<f32> = audio_data_vectors.clone().into();

        //RUN INFERENCE
        let (text, tokens) = match waveform_to_text(&whisper, &bpe, lang, audio_data_vectors_clone_for_inference, 16000, true) {
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
            &processed_len,
            Local::now().format("%Y-%m-%d %H:%M:%S"),
        );
        println!("{}", output);

        let mut file = file.lock().unwrap();
        writeln!(file, "{}\n", output).unwrap_or_else(|e| {
            eprintln!("Error writing transcription file: {}", e);
            process::exit(1);
        });

        // Remove the processed data from the buffer
        audio_data_vectors.drain(0..processed_len);    
        //audio_data_vectors.shrink_to_fit(); //REVISIT IF THIS IS NECESSARY
    }
}

fn handle_input_data(input_data: &[f32], sample_rate: &f32) {
    // Create a signal from the input data
    let signal = signal::from_iter(input_data.iter().cloned());

    // Create a linear interpolator
    let interpolator = Linear::new(0.0, 0.0);

    // Create a source from the signal and the interpolator
    let mut source = signal.from_hz_to_samples(sample_rate).upsample(interpolator);

    // Create a ring buffer
    let mut ring_buffer = Bounded::from(vec![0i16; 16000]);

    // Fill the ring buffer with samples from the source
    for _ in 0..16000 {
        let sample = source.next();
        let sample_i16 = i16::from_sample(sample);
        ring_buffer.push(sample_i16);
    }

    // Now, `ring_buffer` contains `i16` samples at a 16 kHz sample rate
}