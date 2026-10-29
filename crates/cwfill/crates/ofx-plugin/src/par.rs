//! Minimal std-only parallelism helpers (no rayon: no global pool living inside
//! the host process, no extra dependency).
//!
//! Every helper runs its jobs through a shared atomic job counter, and the
//! CALLING thread works the queue too. So if the host refuses to give us a
//! thread (`spawn_scoped` fails), the work still completes on the threads we
//! did get — never a deadlock, never a lost job. With `panic = "abort"` a
//! panic anywhere takes the host down, so the closures handed in here must not
//! panic (the solver/render kernels are plain arithmetic on pre-sized buffers).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Thread cap. A 2x8-core Xeon (32 threads) measured no gain past 16 for
/// the pipelined solve, and a host like Flame has its own work to do.
pub const MAX_DEFAULT_THREADS: usize = 16;

/// Worker threads to use for one render. `CWFILL_THREADS=<n>` overrides
/// (1 = fully sequential; clamped to 1..=64); otherwise
/// `min(available_parallelism, 16)`. Read once per process.
pub fn default_threads() -> usize {
    static T: OnceLock<usize> = OnceLock::new();
    *T.get_or_init(|| {
        if let Some(n) = std::env::var("CWFILL_THREADS").ok().and_then(|s| s.trim().parse::<usize>().ok()) {
            return n.clamp(1, 64);
        }
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(MAX_DEFAULT_THREADS)
            .max(1)
    })
}

/// Run `job(i)` for every `i in 0..n_jobs` on up to `threads` threads
/// (including the caller). Job order across threads is unspecified, so jobs
/// must be independent (or synchronise themselves, as the pipelined solver
/// does — it relies on jobs being CLAIMED in increasing order, which this
/// guarantees).
pub fn run_jobs<F>(n_jobs: usize, threads: usize, job: F)
where
    F: Fn(usize) + Sync,
{
    let threads = threads.max(1).min(n_jobs.max(1));
    if threads <= 1 {
        for i in 0..n_jobs {
            job(i);
        }
        return;
    }
    let next = AtomicUsize::new(0);
    let work = || loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= n_jobs {
            break;
        }
        job(i);
    };
    std::thread::scope(|s| {
        for _ in 1..threads {
            // A failed spawn just means fewer helpers; the queue still drains.
            let _ = std::thread::Builder::new().spawn_scoped(s, &work);
        }
        work();
    });
}

/// Split `data` into consecutive chunks of `chunk_len` elements (last one may
/// be short) and call `f(chunk_index, chunk)` for each, in parallel.
pub fn chunks_mut<T, F>(data: &mut [T], chunk_len: usize, threads: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let chunk_len = chunk_len.max(1);
    if threads <= 1 || data.len() <= chunk_len {
        for (i, c) in data.chunks_mut(chunk_len).enumerate() {
            f(i, c);
        }
        return;
    }
    let slots: Vec<Mutex<Option<&mut [T]>>> = data.chunks_mut(chunk_len).map(|c| Mutex::new(Some(c))).collect();
    run_jobs(slots.len(), threads, |i| {
        let chunk = slots[i].lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(c) = chunk {
            f(i, c);
        }
    });
}

/// Rows per parallel job for an image `rows` tall, so each thread gets a few
/// jobs (load balance) without per-job overhead dominating.
pub fn rows_per_job(rows: usize, threads: usize) -> usize {
    if threads <= 1 {
        return rows.max(1);
    }
    (rows / (threads * 4)).max(1)
}

/// Call `f(y0, y1)` over consecutive row ranges covering `0..rows`, in
/// parallel. Closures write through [`SendPtr`] into disjoint rows.
pub fn row_chunks<F>(rows: usize, threads: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    let rpj = rows_per_job(rows, threads);
    let n = (rows + rpj - 1) / rpj;
    run_jobs(n, threads, |j| f(j * rpj, ((j + 1) * rpj).min(rows)));
}

/// Threads worth using for `pixels` of light per-pixel work: below ~64K pixels
/// thread spawn costs more than it saves.
pub fn threads_for(pixels: usize, threads: usize) -> usize {
    if pixels < 65_536 {
        1
    } else {
        threads.max(1)
    }
}

/// Raw pointer that may be shared across the scoped worker threads. Every use
/// site guarantees disjoint (or happens-before ordered) access.
#[derive(Clone, Copy)]
pub struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_jobs_covers_every_job_once() {
        for &t in &[1usize, 2, 4, 16] {
            let hits: Vec<AtomicUsize> = (0..1000).map(|_| AtomicUsize::new(0)).collect();
            run_jobs(hits.len(), t, |i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "threads={t}");
        }
    }

    #[test]
    fn chunks_mut_writes_every_chunk() {
        let mut v = vec![0usize; 1003];
        chunks_mut(&mut v, 10, 8, |ci, c| {
            for (k, x) in c.iter_mut().enumerate() {
                *x = ci * 10 + k;
            }
        });
        assert!(v.iter().enumerate().all(|(i, &x)| i == x));
    }
}
