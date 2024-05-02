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
use rtrb::{Consumer, RingBuffer};

const BUFFER_FRAME_COUNT: usize = 30;
const MINIMUM_SAMPLE_COUNT: usize = 1600 * 4; // @ 16kHz = 400ms

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

    let audio_queue_and_notifier = Arc::new((Mutex::new(VecDeque::<i16>::new()), Condvar::new()));


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

fn normalize_audio_data_to_16k(input_data: &[f32], input_sample_rate: &f32) -> Vec<f32> {
    let target_sample_rate = 16000f32;
    let resample_ratio = target_sample_rate / input_sample_rate;
    let mut resampled_data = Vec::new();
    for i in 0..(input_data.len() as f32 * resample_ratio) as usize {
        let x = i as f32 / resample_ratio;
        let x1 = x.floor() as usize;
        let x2 = x.ceil() as usize;

        if x2 >= input_data.len() {
            break;
        }

        let y1 = input_data[x1];
        let y2 = input_data[x2];

        let y = y1 + (y2 - y1) * (x - x1 as f32);
        resampled_data.push(y);
    }

    resampled_data
}

fn process_audio_data(
    audio_queue_and_notifier: Arc<(Mutex<VecDeque<i16>>, Condvar)>,
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


        //LEAVING THIS FOR NOW AS THEIR IS STILL A BIT OF AUDIO DISTORTION AND WANT TO DEBUG LATER
        // let spec = hound::WavSpec {
        //     channels: 1,
        //     sample_rate: 16000, // adjust this to match your audio data
        //     bits_per_sample: 16, // adjust this to match your audio data
        //     sample_format: hound::SampleFormat::Int,
        // };
        
        // let mut writer = hound::WavWriter::create(format!("output_{}.wav", i), spec).unwrap();
        // let mut audio_data_vectors_clone_for_inference2: Vec<i16> = audio_data_vectors.clone().into();
        // for sample in audio_data_vectors_clone_for_inference2 {
        //     writer.write_sample(sample).unwrap(); // cast to i16, adjust this to match your audio data
        // }
        
        // writer.finalize().unwrap();


        //RUN INFERENCE
        let speech_segment_f32: Vec<f32> = audio_data_vectors.clone().into_iter().map(|x| x as f32 / 32767.0).collect();
        let (text, tokens) = match waveform_to_text(&whisper, &bpe, lang, speech_segment_f32, 16000, true) {
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

fn record_audio(
    audio_queue_and_notifier: Arc<(Mutex<VecDeque<i16>>, Condvar)>,
) {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("Failed to get default input device");
    let config = device.default_input_config().expect("Failed to get default input config");
    let sample_rate = config.sample_rate().0 as f32;
    let mut vad = Vad::new_with_rate(webrtc_vad::SampleRate::Rate16kHz);
    vad.set_mode(VadMode::Aggressive);
    let (lock, cvar) = &*audio_queue_and_notifier;

    // Create a stream with the default input format
    let (mut producer, mut consumer) = RingBuffer::<i16>::new(16384);
    let stream = device.build_input_stream(
        &config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let data_16k = normalize_audio_data_to_16k(data, &sample_rate);
            let vad_data_i16_16k: Vec<i16> = data_16k.iter().map(|x| (*x * 32767.0) as i16).collect();
            for sample in vad_data_i16_16k {
                producer.push(sample).expect("Failed to push sample to ring buffer");
            }
        },  
        move |err| eprintln!("Error: {}", err),
        None
    ).expect("Failed to build input stream");
    // Play the stream
    stream.play().expect("Failed to play stream");


    /*
        TODO This is a dirty hack and should be changed to an algorithm
        that transcribes in short segments and also concatenates those segments 
        checking the results against one another, the choice of length of small vs 
        long segment will be hard to figure out
    */
    let mut unactive_count = 0;
    let mut speaking = false;
    let mut speech_segment = Vec::<i16>::new();
    loop {
        if consumer.slots() > 160 {
            let mut audio_frame = Vec::<i16>::new();
            for _ in 0..160 {
                match consumer.pop() {
                    Ok(value) => {
                        audio_frame.push(value);
                    }
                    Err(err) => {
                        println!("Error: {}", err);
                        break;
                    },
                }
            }

            let speech_active = vad.is_voice_segment(&audio_frame).expect("Failed to check voice segment");
            if speaking {
                if speech_active {
                    speech_segment.extend(audio_frame);
                } else {
                    if unactive_count > BUFFER_FRAME_COUNT {
                        /* 
                            If more than 30 frames of unactive speech
                            then consider end of segment and 
                            send over the channel to transcribing service
                        */ 
                        speaking = false;
                        if speech_segment.len() > MINIMUM_SAMPLE_COUNT {
                            //HERE WE SHOULD RUN INFERENCE
                            let mut queue = lock.lock().unwrap();
                            for sample in speech_segment.clone() {
                                queue.push_back(sample);
                            }
                            cvar.notify_all();
                        }
                        speech_segment.clear();
                    } else {
                        unactive_count += 1;
                    }
                }
            } else {
                if speech_active {
                    speaking = true;
                    unactive_count = 0;
                    speech_segment.extend(audio_frame);
                }
            }
        }
    }
}
