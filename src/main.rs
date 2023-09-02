use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::Mutex;
use std::time::Instant;
use indicatif::{ProgressIterator, ProgressStyle};
use miniz_oxide::deflate::core::{compress, CompressorOxide, TDEFLFlush};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;
use rand::prelude::*;
use rayon::current_thread_index;
use once_cell::sync::Lazy;
use knn::PointCloud;

const N_CTX: usize = 16;
const BATCH_SIZE: usize = 1;
const N_STEPS: usize = 1_000;

static BUFFERS: Lazy<[Mutex<Box<[u8; 100]>>; 32]> = Lazy::new(|| {
    core::array::from_fn(|_| Mutex::new(Box::new([0u8; 100])))
});
static COMPRESSORS: Lazy<[Mutex<Box<CompressorOxide>>; 32]> = Lazy::new(|| {
    core::array::from_fn(|_| Mutex::new(Box::new(CompressorOxide::default())))
});

fn main() {
    Lazy::force(&BUFFERS);
    Lazy::force(&COMPRESSORS);

    let start = Instant::now();
    let text = fs::read_to_string("input.txt").unwrap()[..1000].to_string();

    // # here are all the unique characters that occur in this text
    let chars = {
        let chars_set: HashSet<char> = HashSet::from_iter(text.chars());
        let mut chars: Vec<char> = Vec::from_iter(chars_set);
        chars.sort();
        chars
    };

    // create a mapping from characters to integers
    let mut stoi : HashMap<char, u8> = HashMap::from_iter(
        chars.iter().enumerate().map(|(i, c)| (c.clone(), i as u8))
    );
    let mut itos : HashMap<u8, char> = HashMap::from_iter(
        chars.iter().enumerate().map(|(i, c)| (i as u8, c.clone()))
    );

    // encoder: take a string, output a list of integers
    let encode = |s: &str| s.chars().map(|c| *stoi.get(&c).unwrap()).collect::<Vec<u8>>();
    // decoder: take a list of integers, output a string
    let decode = |l: Vec<u8>| l.iter().map(|i| itos.get(i).unwrap()).collect::<String>();

    let data = encode(text.as_str());
    let n_vocab: i64 = chars.len() as i64;
    let n_train: i64 = data.len() as i64;

    println!("n_vocab = {n_vocab}");
    println!("n_ctx   = {N_CTX}");
    println!("n_train = {n_train}");


    let mut X = Vec::with_capacity(N_CTX * N_STEPS);
    let mut Y= Vec::with_capacity(N_CTX * N_STEPS);

    let before = Instant::now();
    for i in (0..N_STEPS).progress() {
        let (x, y) = get_data(&data);
        for token_idx in 0..N_CTX {
            let context = &x[0][..token_idx+1];
            let target = &y[0][token_idx];

            // println!("when context is {:?}, target is {}", decode(context.to_vec()), itos.get(target).unwrap());
            X.push((context, compressed_size(context)));
            Y.push(target);
        }
    }
    println!("Getting X,Y | Elapsed time: {:.2?}", before.elapsed());

    let before = Instant::now();
    let progress_style = ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {per_sec} {pos:>7}/{len:7} {msg}")
        .unwrap();
    let ncd_scores = (0..X.len())
        .into_iter() // TODO: why is it this slow ? parallel is even slower
        .progress()
        .with_style(progress_style)
        .map(|i| (0..X.len())
            .into_par_iter()
            .map(|j: usize| ncd(X[i].0, X[i].1, X[j].0, X[j].1))
                .collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();
    println!("ncd_scores | Elapsed time: {:.2?}", before.elapsed());

    generate(&pc, b"hello", 200, &X);

    println!("TOOK: {}", start.elapsed().as_secs_f64());
}

fn generate(knn: &PointCloud<Vec<f64>>, context: &[u8], max_new_tokens: usize, ncd_scores: &Vec<Vec<f64>>, X: &Vec<(&[u8], u64)>, Y: &Vec<u8>) {
    for _ in 0..max_new_tokens {
        let idx_cond = &context[context.len()-N_CTX..];
        let idx_cond_compressed = compressed_size(&idx_cond);

        // get the predictions
        let ncd_scores: Vec<f64> = X.iter().map(|(x, sz)| ncd(&idx_cond, idx_cond_compressed, &x, *sz)).collect();
        let neighboors = knn.get_nearest_k(ncd_scores, 7);

    }
}

fn get_data(data: &Vec<u8>) -> ([&[u8]; BATCH_SIZE], [&[u8]; BATCH_SIZE]) {
    let max_idx = data.len() - N_CTX;
    let ix: [usize; BATCH_SIZE] = core::array::from_fn(|x| (random::<f32>() * max_idx as f32) as usize);

    let x: [&[u8]; BATCH_SIZE] = ix.map(|i| &data[i..(i+N_CTX)]);
    let y: [&[u8]; BATCH_SIZE] = ix.map(|i| &data[(i+1)..(i+N_CTX+1)]);

    (x, y)
}

fn concat_with_space(x: &[u8], y: &[u8]) -> Vec<u8> {
    let mut s = Vec::with_capacity(x.len() + 1 + y.len());
    for _x in x {
        s.push(*_x);
    }
    s.push(' ' as u8);
    for _y in y {
        s.push(*_y);
    }

    s
}

fn compressed_size(x: &[u8]) -> u64 {
    let thread_id = current_thread_index().unwrap_or(0);
    let mut buffer = BUFFERS[thread_id].lock().unwrap();
    let mut compressor = COMPRESSORS[thread_id].lock().unwrap();
    compressor.reset();
    let (_, consumed, produced) = compress(&mut compressor, x, &mut buffer[..], TDEFLFlush::None);

    produced as u64
}

fn ncd(x: &[u8], x_compressed: u64, x2: &[u8], x2_compressed: u64) -> f64 {
    let xx2 = compressed_size(&concat_with_space(x, x2)[..]);
    (xx2 - x_compressed.min(x2_compressed)) as f64 / x_compressed.max(x2_compressed) as f64
}


