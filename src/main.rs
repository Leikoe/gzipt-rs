use std::collections::{HashMap, HashSet};
use std::fs;


fn main() {
    let text = fs::read_to_string("input.txt").unwrap();

    // # here are all the unique characters that occur in this text
    let chars = {
        let mut chars_set: HashSet<char> = HashSet::from_iter(text.chars());
        let mut chars: Vec<char> = Vec::from_iter(chars_set);
        chars.sort();
        chars
    };

    // create a mapping from characters to integers
    // stoi = { ch:i for i,ch in enumerate(chars) }
    // itos = { i:ch for i,ch in enumerate(chars) }
    // encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    // decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    // let mut stoi = HashMap::new();
    let mut stoi : HashMap<char, i64> = HashMap::from_iter(
        chars.iter().enumerate().map(|(i, c)| (c.clone(), i as i64))
    );
    let mut itos : HashMap<i64, char> = HashMap::from_iter(
        chars.iter().enumerate().map(|(i, c)| (i as i64, c.clone()))
    );

    let encode = |s: &str| s.chars().map(|c| stoi.get(&c).unwrap().clone()).collect::<Vec<i64>>();
    let decode = |l: Vec<i64>| l.iter().map(|i| itos.get(i).unwrap()).collect::<String>();

    dbg!(decode(encode("lol")));
    println!("Hello, world!");
}
