//! CUDA keeper regression (Linux, Flame's ORT 1.22 + cuDNN 9): the cwSegment
//! path (`build_session`) builds a GPU session of a conv model, runs it (so
//! cuDNN is initialised), drops it (Model Folder / Use GPU switch), rebuilds,
//! runs, drops again and exits. Without the keeper, dropping the process's
//! last CUDA session can abort in cuDNN's exit handlers. Run as many processes:
//!   B=$(cargo test --release -p cwtools-core --test keeper --no-run ... )
//!   for i in $(seq 20); do CWTOOLS_KEEPER_MODEL=<sam3 encoder.onnx> $B --ignored -q; echo rc=$?; done
//! (CWTOOLS_NO_CUDA_KEEPER=1 disables the keeper for contrast.)

#[test]
#[ignore]
fn drop_and_rebuild_gpu_sessions() {
    use cwtools_core::ort_rt::{build_session, init_ort};
    use ort::value::Tensor;
    init_ort("").unwrap(); // cwtools.conf ort_lib / auto-detect
    let p = std::path::PathBuf::from(std::env::var("CWTOOLS_KEEPER_MODEL").expect("set CWTOOLS_KEEPER_MODEL (SAM3 encoder.onnx)"));
    for _ in 0..2 {
        let (mut s, gpu) = build_session(&p, true).unwrap();
        assert!(gpu, "CUDA EP not available");
        let x = Tensor::from_array((vec![1i64, 3, 1008, 1008], vec![0.5f32; 3 * 1008 * 1008])).unwrap();
        let name = s.inputs[0].name.clone();
        s.run(ort::inputs![name.as_str() => x]).unwrap();
        drop(s); // "switch Model Folder": every GPU session dropped
    }
}
