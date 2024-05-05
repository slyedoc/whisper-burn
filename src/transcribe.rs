use crate::audio::{max_waveform_samples, prep_audio};
use crate::beam;
use crate::helper::*;
use crate::model::*;
use crate::token::{self, *};
use burn::{
    config::Config,
    module::Module,
    tensor::{
        self,
        activation::log_softmax,
        backend::{self, Backend},
        Data, ElementConversion, Float, Int, Tensor,
    },
};
use num_traits::ToPrimitive;
use std::{f32, iter, ops::Div};
use std::time::Instant;

pub fn waveform_to_text<B: Backend>(
    whisper: &Whisper<B>,
    bpe: &Gpt2Tokenizer,
    lang: Language,
    waveform: Vec<f32>,
    sample_rate: usize,
    streaming_mode: bool,
) -> token::Result<(String, Vec<usize>)> {
    let device = whisper.devices()[0].clone();

    let n_ctx_max_encoder = whisper.encoder_ctx_size();
    let padding = 30; //ADJUST THIS IF CHINKS ARE REPEATING THEMSELVES ENDLESSLY
    let n_waveform_samples_per_window = max_waveform_samples(n_ctx_max_encoder - padding);





    // let tensor_size = waveform.len();
    // let mel_tensor = Tensor::from_floats(
    //     tensor::Data::new(waveform, [tensor_size].into()),
    //     &device,
    // );
    // let dims = &mel_tensor.dims();

    // let mel = prep_audio(mel_tensor.unsqueeze(), sample_rate as f64);
    // println!("BEFORE PREP: {:?}, MEL AFTER PREP: {:?}", dims, mel.dims());

    // let (new_text, new_tokens) = mels_to_text(
    //     whisper,
    //     bpe,
    //     lang,
    //     mel,
    //     padding,
    //     streaming_mode,
    // )?;
    // println!("NEW TEXT: {:?}, NEW TOKENS: {:?}", &new_text, &new_tokens);

    // Ok((new_text, new_tokens))




    let mel_iter =
        waveform_to_mel_tensor(waveform, sample_rate, n_waveform_samples_per_window, device);

    let mut text = String::new();
    let mut tokens: Vec<usize> = Vec::new();

    //IN THE FOLLOWING CODE, WE WILL PRETTY MUCH ALWAYS ITERATE JUST ONCE, SINCE WE ARE SENDING SUCH SHORT CLIPS OF AUDIO. THIS MEANS FIND CHUNK OVERLAP IS NOT NECESSARY BUT CAN LEAVE IT FOR THE FUTURE
    for (i, mel) in mel_iter.enumerate() {
        println!("MEL DIMS: {:?}", &mel.dims());
        let (new_text, new_tokens) = mels_to_text(
            whisper,
            bpe,
            lang,
            mel,
            padding,
            streaming_mode,
        )?;
        //println!("NEW TEXT: {:?}, NEW TOKENS: {:?}", &new_text, &new_tokens);

        if let Some((prev_index, curr_index)) =
            find_chunk_overlap(&tokens[..], &new_tokens[..], 40, 3)
        {
            tokens.truncate(prev_index);
            tokens.extend(&new_tokens[curr_index..]);
        } else {
            tokens.extend(new_tokens);
        }

        text = bpe.decode(&tokens[..], true)?;
    }

    Ok((text, tokens))
}


fn waveform_to_mel_tensor<B: Backend>(
    waveform: Vec<f32>,
    sample_rate: usize,
    window_length_samples: usize,
    device: B::Device,
) -> impl Iterator<Item = Tensor<B, 3>> {
    let chunk_overlap = sample_rate * 3;
    let n_samples_per_tensor = window_length_samples;
    let shift = n_samples_per_tensor.saturating_sub(chunk_overlap).max(1);
    let iter_len = waveform.len().saturating_sub(1).div(shift) + 1;

    (0..iter_len).into_iter().map(move |i| {
        let start = i * shift;
        let end = (start + n_samples_per_tensor).min(waveform.len());

        let slice = &waveform[start..end];

        let waveform = Tensor::from_floats(
            tensor::Data::new(slice.to_vec(), [slice.len()].into()),
            &device,
        );
        println!("WAQVEFORM LEN: {:?}", &waveform.dims());

        let mels = prep_audio(waveform.unsqueeze(), sample_rate as f64);
        println!("MELS: {:?}", &mels);

        mels
    })
}

#[derive(Clone)]
struct BeamSearchToken {
    token: usize,
    log_prob: f64,
}

fn mels_to_text<B: Backend>(
    whisper: &Whisper<B>,
    bpe: &Gpt2Tokenizer,
    lang: Language,
    mels: Tensor<B, 3>,
    padding: usize,
    streaming_mode: bool,
) -> token::Result<(String, Vec<usize>)> {
    let device = mels.device();

    let n_ctx_max_encoder = whisper.encoder_ctx_size();
    let n_ctx_max_decoder = whisper.decoder_ctx_size();

    let [n_channel, n_mel, n_ctx] = mels.dims();
    if n_ctx + padding > n_ctx_max_encoder {
        println!(
            "Audio has length of {} which exceeds maximum length {}. It will be clipped.",
            n_ctx + padding,
            n_ctx_max_encoder
        );
    }

    // the zero padding helps whisper determine end of text
    let mels = Tensor::cat(
        vec![
            mels.slice([0..1, 0..n_mel, 0..(n_ctx).min(n_ctx_max_encoder - padding)]),
            Tensor::zeros([1, n_mel, padding], &device),
        ],
        2,
    );
    let encoder_output = whisper.forward_encoder(mels);

    let start_token = bpe.special_token(SpecialToken::StartofTranscript).unwrap();
    let transcription_token = bpe.special_token(SpecialToken::Transcribe).unwrap();
    let start_of_prev_token = bpe.special_token(SpecialToken::StartofPrev).unwrap();
    let lang_token = bpe.special_token(SpecialToken::Language(lang)).unwrap();
    let first_timestamp_token = bpe.special_token(SpecialToken::Timestamp(0.0)).unwrap();
    let end_token = bpe.special_token(SpecialToken::EndofText).unwrap();
    let notimestamp = bpe.special_token(SpecialToken::NoTimeStamps).unwrap();

    let mut initial_tokens = Vec::new();
    initial_tokens.extend([start_token, lang_token, transcription_token, notimestamp]);

    println!("{:?}", &initial_tokens);

    type BeamNode = beam::BeamNode<BeamSearchToken>;
    let initial_tokens = BeamNode {
        seq: initial_tokens
            .into_iter()
            .map(|tok| BeamSearchToken {
                token: tok,
                log_prob: 0.0,
            })
            .collect(),
        log_prob: 0.0,
    };

    let neg_infty = -f32::INFINITY;

    let vocab_size = bpe.vocab_size();
    let mut special_tokens_maskout: Vec<f32> = (0..vocab_size)
        .into_iter()
        .map(|token| {
            if bpe.is_special(token) {
                neg_infty
            } else {
                0.0
            }
        })
        .collect();
    //special_tokens_maskout[end_token] = 1.0;

    let special_tokens_maskout = Tensor::from_data(
        Data::new(special_tokens_maskout, [vocab_size].into()).convert(),
        &device,
    );

    let beamsearch_next = |beams: &[BeamNode]| {
        // convert tokens into tensor
        let max_seq_len = beams.iter().map(|beam| beam.seq.len()).max().unwrap_or(0);
        let flattened_tokens: Vec<_> = beams
            .iter()
            .flat_map(|beam| {
                let additional_tokens = max_seq_len - beam.seq.len();
                beam.seq
                    .iter()
                    .map(|btok| btok.token)
                    .chain(iter::once(0).cycle().take(additional_tokens))
            })
            .collect();

        let token_tensor = Tensor::from_ints(
            Data::from_usize(Data::new(
                flattened_tokens,
                [beams.len(), max_seq_len].into(),
            )),
            &device,
        );

        let logits =
            whisper.forward_decoder(token_tensor, encoder_output.clone().repeat(0, beams.len()));
        let logits = if max_seq_len > 5 {
            logits
        } else {
            logits + special_tokens_maskout.clone().unsqueeze()
        };
        let log_probs = log_softmax(logits, 2);

        let [n_batch, n_token, n_dict] = log_probs.dims();
        let beam_log_probs = beams.iter().enumerate().map(|(i, beam)| {
            let batch = i;
            let token_index = beam.seq.len() - 1;

            log_probs
                .clone()
                .slice([batch..batch + 1, token_index..token_index + 1])
                .flatten::<1>(0, 2)
                .into_data()
                .value
        });

        let continuations = beam_log_probs
            .zip(beams)
            .map(|(log_probs, beam)| {
                log_probs
                    .into_iter()
                    .map(|log_prob| log_prob.elem::<f64>())
                    .enumerate()
                    .map(|(token_id, log_prob)| {
                        (
                            BeamSearchToken {
                                token: token_id,
                                log_prob: log_prob,
                            },
                            beam.log_prob + log_prob,
                        )
                    })
                    .collect()
            })
            .collect();

        continuations
    };

    let beamsearch_is_finished = |toks: &[BeamSearchToken]| {
        if let Some(btok) = toks.last() {
            btok.token == end_token
        } else {
            false
        }
    };

    let beam_size = 5;
    let max_depth = 30;
    let mut tokens: Vec<_> = beam::beam_search(
        vec![initial_tokens],
        beamsearch_next,
        beamsearch_is_finished,
        beam_size,
        max_depth,
    )
    .into_iter()
    .map(|btok| btok.token)
    .collect();

    println!("TOKENS RIGHT BEFORE DECODING: {:?}", &tokens);
    let text = bpe.decode(&tokens[..], false)?;

    return Ok((text, tokens));
}





//HELPERS
fn find_chunk_overlap(
    prev_tokens: &[usize],
    curr_tokens: &[usize],
    max_n_offsets: usize,
    min_n_overlaps: usize,
) -> Option<(usize, usize)> {
    let mut max_overlap = 0;
    let mut max_overlap_indices = (0, 0);
    let n_offsets = prev_tokens.len().min(curr_tokens.len()).min(max_n_offsets);

    for offset in 0..n_offsets {
        let prev_start_index = prev_tokens.len() - 1 - offset;
        let mut overlap_iter = prev_tokens
            .iter()
            .skip(prev_start_index)
            .zip(curr_tokens.iter())
            .enumerate()
            .filter(|(_, (&old, &new))| old == new);

        let n_overlap = overlap_iter.clone().count();
        if n_overlap > max_overlap {
            max_overlap = n_overlap;

            let curr_overlap_index = overlap_iter.next().unwrap().0;
            let prev_overlap_index = prev_start_index + curr_overlap_index;
            max_overlap_indices = (prev_overlap_index, curr_overlap_index)
        }
    }

    if max_overlap >= min_n_overlaps {
        Some(max_overlap_indices)
    } else {
        None
    }
}
