//! Convention guard: the cwTools menu order in Flame.
//!
//! Flame sorts the OFX menu by plugin IDENTIFIER, not by label (cwProbe test:
//! "1 cwProbe Numbered" sorted as `...cwProbe.Numbered`). The identifiers carry
//! the order Segment, Matte, Refine, Fill; labels stay plain. cwFill keeps its
//! released identifier (`dev.weingart.cwFill`): digits sort before letters, so
//! cw1/cw2/cw3 come first. Changing an identifier breaks saved Flame setups.

use std::path::Path;

const ORDER: &[(&str, &str)] = &[
    ("cwsegment/src/lib.rs", "dev.weingart.cw1Segment"),
    ("cwmatte/src/lib.rs", "dev.weingart.cw2Matte"),
    ("cwrefine/src/lib.rs", "dev.weingart.cw3Refine"),
    ("cwfill/crates/ofx-plugin/src/lib.rs", "dev.weingart.cwFill"),
];

#[test]
fn identifiers_give_the_menu_order() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    for (file, id) in ORDER {
        let src = std::fs::read_to_string(crates.join(file)).unwrap();
        let line = src.lines().find(|l| l.contains("static PLUGIN_IDENTIFIER")).unwrap_or_else(|| panic!("{file}: no PLUGIN_IDENTIFIER"));
        assert!(line.contains(&format!("b\"{id}\\0\"")), "{file}: expected {id}, got {line}");
    }
    // Flame's sort: plain and case-insensitive must agree
    for w in ORDER.windows(2) {
        assert!(w[0].1 < w[1].1, "{} !< {}", w[0].1, w[1].1);
        assert!(w[0].1.to_lowercase() < w[1].1.to_lowercase(), "{} !< {} (case-insensitive)", w[0].1, w[1].1);
    }
}
