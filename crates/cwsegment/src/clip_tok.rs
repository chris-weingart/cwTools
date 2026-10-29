//! CLIP BPE tokenizer for SAM3 text prompts (Local mode), no Python.
//!
//! Same pipeline as facebook/sam3's tokenizer.json (CLIPTokenizer):
//! special tokens `<|startoftext|>` / `<|endoftext|>` matched in the raw text,
//! whitespace runs -> one space, lowercase, split into contractions ('s 't 're
//! 've 'm 'll 'd), letter runs, single digits and other-symbol runs (whitespace
//! dropped), GPT-2 byte-to-unicode mapping, BPE with `</w>` on the last symbol
//! of each word. Output: [BOS] + up to 30 ids + [EOS], padded with EOS to 32,
//! attention mask 1 over BOS..EOS.
//!
//! Not reproduced: NFC normalisation (typed and pasted text is NFC almost
//! always; decomposed accents can tokenize differently) and the exact Unicode
//! letter class (`\p{L}`; we use `char::is_alphabetic`, which also counts some
//! combining marks). Plain ASCII prompts are exact; the golden test
//! (tests/clip_tok.rs) covers ASCII, Latin accents, CJK, Cyrillic, Arabic, emoji.
//!
//! Files (from facebook/sam3, SAM License): `vocab.json` + `merges.txt` in the
//! cwSegment model folder.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

pub const TEXT_LEN: usize = 32;
pub const BOS: i64 = 49406;
pub const EOS: i64 = 49407;
const SPECIALS: [(&str, i64); 2] = [("<|startoftext|>", BOS), ("<|endoftext|>", EOS)];

pub struct ClipTokenizer {
    vocab: HashMap<String, i64>,
    ranks: HashMap<(String, String), usize>,
    byte_char: [char; 256],
    cache: Mutex<HashMap<String, Vec<i64>>>,
}

/// GPT-2 byte -> printable unicode char table.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let c = if printable {
            b
        } else {
            n += 1;
            255 + n
        };
        table[b as usize] = char::from_u32(c).unwrap_or('\u{FFFD}');
    }
    table
}

impl ClipTokenizer {
    /// `vocab.json` + `merges.txt` from `dir`.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let vp = dir.join("vocab.json");
        let mp = dir.join("merges.txt");
        let v = std::fs::read_to_string(&vp).map_err(|e| format!("{}: {e}", vp.display()))?;
        let m = std::fs::read_to_string(&mp).map_err(|e| format!("{}: {e}", mp.display()))?;
        Self::from_strings(&v, &m)
    }

    pub fn from_strings(vocab_json: &str, merges_txt: &str) -> Result<Self, String> {
        let raw: HashMap<String, i64> = serde_json::from_str(vocab_json).map_err(|e| format!("vocab.json: {e}"))?;
        let mut ranks = HashMap::new();
        for line in merges_txt.lines() {
            if line.starts_with("#version") || line.trim().is_empty() {
                continue;
            }
            let mut it = line.split(' ');
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                let r = ranks.len();
                ranks.entry((a.to_string(), b.to_string())).or_insert(r);
            }
        }
        if raw.len() < 49408 || ranks.is_empty() {
            return Err(format!("tokenizer files look incomplete ({} vocab, {} merges)", raw.len(), ranks.len()));
        }
        Ok(ClipTokenizer { vocab: raw, ranks, byte_char: bytes_to_unicode(), cache: Mutex::new(HashMap::new()) })
    }

    /// (input_ids, attention_mask), both TEXT_LEN long.
    pub fn encode(&self, text: &str) -> (Vec<i64>, Vec<i64>) {
        let mut ids = vec![BOS];
        for piece in split_specials(text) {
            match piece {
                Piece::Special(id) => ids.push(id),
                Piece::Text(t) => {
                    for w in pre_tokenize(&normalize(t)) {
                        ids.extend(self.bpe_word(&w));
                    }
                }
            }
        }
        ids.truncate(TEXT_LEN - 1);
        ids.push(EOS);
        let n = ids.len();
        let mut mask = vec![1i64; n];
        ids.resize(TEXT_LEN, EOS);
        mask.resize(TEXT_LEN, 0);
        (ids, mask)
    }

    fn bpe_word(&self, word: &str) -> Vec<i64> {
        if let Ok(c) = self.cache.lock() {
            if let Some(v) = c.get(word) {
                return v.clone();
            }
        }
        let mapped: String = word.bytes().map(|b| self.byte_char[b as usize]).collect();
        let mut sym: Vec<String> = mapped.chars().map(|c| c.to_string()).collect();
        if let Some(last) = sym.last_mut() {
            last.push_str("</w>");
        }
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..sym.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(sym[i].clone(), sym[i + 1].clone())) {
                    if best.map_or(true, |(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i0)) = best else { break };
            let (a, b) = (sym[i0].clone(), sym[i0 + 1].clone());
            let mut out = Vec::with_capacity(sym.len());
            let mut i = 0;
            while i < sym.len() {
                if i + 1 < sym.len() && sym[i] == a && sym[i + 1] == b {
                    out.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    out.push(std::mem::take(&mut sym[i]));
                    i += 1;
                }
            }
            sym = out;
        }
        let ids: Vec<i64> = sym.iter().map(|s| *self.vocab.get(s).unwrap_or(&EOS)).collect();
        if let Ok(mut c) = self.cache.lock() {
            c.insert(word.to_string(), ids.clone());
        }
        ids
    }
}

enum Piece<'a> {
    Text(&'a str),
    Special(i64),
}

fn split_specials(text: &str) -> Vec<Piece<'_>> {
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let hit = SPECIALS.iter().filter_map(|(s, id)| rest.find(s).map(|p| (p, s.len(), *id))).min_by_key(|h| h.0);
        match hit {
            Some((p, len, id)) => {
                if p > 0 {
                    out.push(Piece::Text(&rest[..p]));
                }
                out.push(Piece::Special(id));
                rest = &rest[p + len..];
            }
            None => {
                if !rest.is_empty() {
                    out.push(Piece::Text(rest));
                }
                return out;
            }
        }
    }
}

/// Whitespace runs -> " ", lowercase (tokenizer.json normalizer minus NFC).
fn normalize(t: &str) -> String {
    let mut s = String::with_capacity(t.len());
    let mut in_ws = false;
    for c in t.chars() {
        if c.is_whitespace() {
            if !in_ws {
                s.push(' ');
            }
            in_ws = true;
        } else {
            s.extend(c.to_lowercase());
            in_ws = false;
        }
    }
    s
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic() && !c.is_numeric()
}

/// CLIP split pattern: 's|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+ (matches kept).
fn pre_tokenize(s: &str) -> Vec<String> {
    let c: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch == '\'' {
            let rest: String = c[i + 1..c.len().min(i + 3)].iter().collect();
            let hit = ["re", "ve", "ll"].iter().find(|x| rest.starts_with(**x)).map(|x| x.len())
                .or_else(|| ["s", "t", "m", "d"].iter().find(|x| rest.starts_with(**x)).map(|_| 1));
            if let Some(n) = hit {
                out.push(c[i..i + 1 + n].iter().collect());
                i += 1 + n;
                continue;
            }
        }
        if ch.is_whitespace() {
            i += 1;
        } else if is_letter(ch) {
            let j = (i..c.len()).find(|&j| !is_letter(c[j])).unwrap_or(c.len());
            out.push(c[i..j].iter().collect());
            i = j;
        } else if ch.is_numeric() {
            out.push(ch.to_string());
            i += 1;
        } else {
            let j = (i..c.len()).find(|&j| c[j].is_whitespace() || is_letter(c[j]) || c[j].is_numeric())
                .unwrap_or(c.len());
            out.push(c[i..j].iter().collect());
            i = j;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_clip() {
        assert_eq!(pre_tokenize("it's the man's x-ray, 21"), ["it", "'s", "the", "man", "'s", "x", "-", "ray", ",", "2", "1"]);
        assert_eq!(pre_tokenize("rock'n'roll"), ["rock", "'", "n", "'", "roll"]);
        assert_eq!(normalize("  Red\t\nCAR "), " red car ");
    }

    #[test]
    fn byte_table_is_a_bijection() {
        let t = bytes_to_unicode();
        let mut v: Vec<char> = t.to_vec();
        v.sort();
        v.dedup();
        assert_eq!(v.len(), 256);
        assert_eq!(t[b'a' as usize], 'a');
        assert_eq!(t[b' ' as usize], '\u{120}');
    }
}
