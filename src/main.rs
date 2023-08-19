use std::collections::{HashMap, HashSet};
use std::fs;
use std::thread::sleep;
use std::time::{Duration, Instant};
use miniz_oxide::deflate::compress_to_vec;
use indicatif::{ProgressIterator, ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::iter::{ParallelIterator, IntoParallelRefIterator};
use rayon::prelude::IntoParallelIterator;


fn main() {
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
    let n_ctx : i64 = 8;
    let n_train: i64 = data.len() as i64;

    println!("n_vocab = {n_vocab}");
    println!("n_ctx   = {n_ctx}");
    println!("n_train = {n_train}");


    let mut X = Vec::with_capacity((n_train*n_ctx) as usize);
    let mut Y= Vec::with_capacity((n_train*n_ctx) as usize);

    let before = Instant::now();
    let d = get_data(&data, n_ctx);
    println!("get_data | Elapsed time: {:.2?}", before.elapsed());

    let before = Instant::now();
    for (x, y) in d.0.iter().zip(d.1.iter()).progress() {
        for token_idx in 0..n_ctx as usize {
            let context = &x[..token_idx+1];
            let target = &y[token_idx];

            // println!("when context is {:?}, target is {}", decode(context.to_vec()), itos.get(target).unwrap());
            X.push((context, compress_to_vec(context, 6).len() as i64));
            Y.push(target);
        }
    }
    println!("X, Y | Elapsed time: {:.2?}", before.elapsed());

    let before = Instant::now();
    let progress_style = ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {per_sec} {pos:>7}/{len:7} {msg}")
        .unwrap();
    let ncd_scores = (0..X.len())
        .into_iter() // TODO: why is it this slow ? parallel is even slower
        .progress_count(X.len() as u64)
        .with_style(progress_style)
        .map(|i| (0..X.len())
            .into_iter()
            .map(|j: usize| ncd(X[i].0, X[i].1, X[j].0, X[j].1))
                .collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();
    println!("ncd_scores | Elapsed time: {:.2?}", before.elapsed());


    println!("TOOK: {}", start.elapsed().as_secs_f64());
}

fn get_data(data: &Vec<u8>, n_ctx: i64) -> (Vec<&[u8]>, Vec<&[u8]>) {
    let ix = Vec::from_iter(0..(data.len() - n_ctx as usize));
    let x = ix.par_iter().map(|&i| &data[i..(i+n_ctx as usize)]).collect();
    let y = ix.par_iter().map(|&i| &data[(i+1)..(i+n_ctx as usize+1)]).collect();

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

fn ncd(x: &[u8], x_compressed: i64, x2: &[u8], x2_compressed: i64) -> f64 {
    let xx2 = compress_to_vec(&*[x, x2].concat(), 6).len() as i64;
    (xx2 - x_compressed.min(x2_compressed)) as f64 / x_compressed.max(x2_compressed) as f64
}


