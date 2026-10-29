//! F/B solve cache between renders.
//!
//! The solve is by far the expensive part of a render; despill, despill
//! gating and the Output mode are cheap per-pixel passes over (plate, F, B).
//! So the solved F/B (plus the band mask + crop it was solved on) is kept per
//! instance, keyed on everything that feeds the solve: frame time, render
//! window, a 128-bit hash of the Source pixels (raw bits, RGBA), and every
//! solve-affecting parameter. Changing an output-only parameter re-renders
//! from the cache; anything else misses and re-solves. Results are
//! deterministic, so a hit reproduces a fresh render bit-for-bit.
//!
//! Bounded: at most [`PER_INSTANCE`] entries per instance and a global byte
//! budget (`CWFILL_CACHE_MB`, default 1024; 0 disables the cache), evicting
//! least-recently-used. Safe under concurrent renders (the host may render
//! different instances in parallel — we declare the default instance-safe
//! render threading): one global mutex, held only for lookup/insert, never
//! across a solve; entries are shared immutable `Arc`s.

use crate::solver::Plane;
use std::sync::{Arc, Mutex, OnceLock};

/// Max cached frames per effect instance.
pub const PER_INSTANCE: usize = 3;

/// Default global budget in MiB (a 4K full-frame entry is ~210 MiB).
pub const DEFAULT_BUDGET_MB: usize = 1024;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Key {
    pub instance: usize,
    pub time_bits: u64,
    pub win: [i32; 4],
    pub content: [u64; 2],
    pub params: Vec<u32>,
}

impl Key {
    /// Placeholder for a render that bypasses the cache (never stored).
    pub fn uncached() -> Self {
        Key { instance: 0, time_bits: 0, win: [0; 4], content: [0; 2], params: Vec::new() }
    }
}

pub struct Entry {
    pub key: Key,
    /// Output mask (non-zero = solved pixel) over the window; None = all.
    pub mask: Option<Vec<u8>>,
    /// Solve crop within the window (x1, y1, x2, y2), exclusive max.
    pub rect: (i32, i32, i32, i32),
    /// Solved planes, crop-sized, FRAME row order (row 0 = top of the crop),
    /// in the solver's working encoding.
    pub fg: Plane,
    pub bg: Plane,
}

impl Entry {
    pub fn bytes(&self) -> usize {
        self.mask.as_ref().map_or(0, |m| m.len()) + (self.fg.data.len() + self.bg.data.len()) * 4
    }
}

/// LRU list of entries (MRU last). The process-wide instance lives behind a
/// mutex; tests build their own so they never race the render tests.
#[derive(Default)]
pub struct Lru {
    entries: Vec<Arc<Entry>>,
}

impl Lru {
    pub const fn new() -> Self {
        Lru { entries: Vec::new() }
    }

    pub fn get(&mut self, key: &Key) -> Option<Arc<Entry>> {
        let i = self.entries.iter().position(|e| &e.key == key)?;
        let e = self.entries.remove(i);
        self.entries.push(e.clone());
        Some(e)
    }

    pub fn put(&mut self, entry: Arc<Entry>, budget: usize) {
        if entry.bytes() > budget {
            return;
        }
        let inst = entry.key.instance;
        let c = &mut self.entries;
        c.retain(|e| e.key != entry.key);
        c.push(entry);
        // Per-instance cap (drop that instance's oldest).
        while c.iter().filter(|e| e.key.instance == inst).count() > PER_INSTANCE {
            let i = c.iter().position(|e| e.key.instance == inst).unwrap_or(0);
            c.remove(i);
        }
        // Global budget (drop globally oldest).
        while !c.is_empty() && c.iter().map(|e| e.bytes()).sum::<usize>() > budget {
            c.remove(0);
        }
    }

    pub fn purge_instance(&mut self, instance: usize) {
        self.entries.retain(|e| e.key.instance != instance);
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

static CACHE: Mutex<Lru> = Mutex::new(Lru::new());

pub fn budget_bytes() -> usize {
    static B: OnceLock<usize> = OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("CWFILL_CACHE_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_BUDGET_MB)
            .saturating_mul(1 << 20)
    })
}

pub fn enabled() -> bool {
    budget_bytes() > 0
}

fn lock() -> std::sync::MutexGuard<'static, Lru> {
    CACHE.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn get(key: &Key) -> Option<Arc<Entry>> {
    lock().get(key)
}

pub fn put_arc(entry: Arc<Entry>) {
    let budget = budget_bytes();
    lock().put(entry, budget)
}

pub fn purge_instance(instance: usize) {
    lock().purge_instance(instance)
}

pub fn clear() {
    lock().entries.clear();
}

/// 128-bit content hash accumulator: two independent multiply-mix lanes over
/// 64-bit words, order-sensitive. Not cryptographic — guards against stale
/// reuse when upstream pixels change, where accidental collisions between
/// real frames are the only concern.
#[derive(Clone, Copy)]
pub struct Hasher {
    a: u64,
    b: u64,
}

impl Hasher {
    pub fn new(seed: u64) -> Self {
        Hasher { a: seed ^ 0x243F_6A88_85A3_08D3, b: seed.rotate_left(32) ^ 0x1319_8A2E_0370_7344 }
    }
    #[inline(always)]
    pub fn word(&mut self, w: u64) {
        self.a = (self.a ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.a ^= self.a >> 29;
        self.b = (self.b.rotate_left(23) ^ w).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    }
    pub fn finish(mut self) -> [u64; 2] {
        for _ in 0..2 {
            self.a = (self.a ^ (self.a >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            self.b = (self.b ^ (self.b >> 33)).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        }
        [self.a ^ (self.a >> 33), self.b ^ (self.b >> 33)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(instance: usize, t: u64, px: usize) -> Arc<Entry> {
        Arc::new(Entry {
            key: Key { instance, time_bits: t, win: [0; 4], content: [0; 2], params: vec![] },
            mask: None,
            rect: (0, 0, 1, 1),
            fg: Plane::new(1, px, 3),
            bg: Plane::new(1, px, 3),
        })
    }

    #[test]
    fn cache_bounds_and_lru() {
        let k = |i, t| Key { instance: i, time_bits: t, win: [0; 4], content: [0; 2], params: vec![] };
        // Per-instance cap: 5 frames of instance 7 → only the last 3 remain.
        let mut c = Lru::new();
        for t in 0..5 {
            c.put(entry(7, t, 10), usize::MAX);
        }
        assert_eq!(c.len(), PER_INSTANCE);
        assert!(c.get(&k(7, 0)).is_none() && c.get(&k(7, 1)).is_none());
        assert!(c.get(&k(7, 2)).is_some());
        // Budget: each entry = 2*10*3*4 = 240 bytes; budget 500 keeps 2, LRU out.
        let mut c = Lru::new();
        c.put(entry(1, 0, 10), 500);
        c.put(entry(2, 0, 10), 500);
        assert!(c.get(&k(1, 0)).is_some()); // touch 1 → 2 is now LRU
        c.put(entry(3, 0, 10), 500);
        assert!(c.get(&k(2, 0)).is_none());
        assert!(c.get(&k(1, 0)).is_some());
        // Oversized entry is never stored; purge drops an instance.
        c.put(entry(4, 0, 1000), 500);
        assert!(c.get(&k(4, 0)).is_none());
        c.purge_instance(1);
        assert_eq!(c.len(), 1);
        // Re-putting the same key replaces, never duplicates.
        c.put(entry(3, 0, 10), 500);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn hasher_is_order_and_value_sensitive() {
        let h = |ws: &[u64]| {
            let mut x = Hasher::new(1);
            for &w in ws {
                x.word(w);
            }
            x.finish()
        };
        assert_ne!(h(&[1, 2]), h(&[2, 1]));
        assert_ne!(h(&[1, 2]), h(&[1, 3]));
        assert_eq!(h(&[5, 6, 7]), h(&[5, 6, 7]));
    }
}
