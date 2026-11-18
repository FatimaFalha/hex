//! ggml's Metal tensor API (Metal 4 / MetalPerformancePrimitives matmuls).
//! ggml turns it on by itself only on M5, M6, A19 and A20 GPUs, and on an
//! Apple M5 under macOS 26.0 its kernels return wrong numbers: every model
//! (Parakeet Unified, Parakeet v2, Whisper) then hears no speech or prints
//! garbage, while the same clip transcribes correctly with the tensor API
//! off. The simdgroup kernels ggml uses on every other Mac are correct and
//! fast enough for dictation, so Hex turns the tensor API off before ggml
//! starts.
//!
//! `main` calls [`disable`] first, before any thread exists. A user who sets
//! `GGML_METAL_TENSOR_DISABLE` keeps their own value, and
//! `HEX_METAL_TENSOR=1` leaves ggml's own choice alone, to compare a
//! fixed macOS or ggml.

const GGML_DISABLE: &str = "GGML_METAL_TENSOR_DISABLE";
const HEX_OPT_IN: &str = "HEX_METAL_TENSOR";

/// Whether Hex should set `GGML_METAL_TENSOR_DISABLE`, given the current
/// values of it and of `HEX_METAL_TENSOR`.
fn should_disable(ggml_disable: Option<&str>, hex_opt_in: Option<&str>) -> bool {
    if ggml_disable.is_some() {
        return false;
    }
    !matches!(hex_opt_in.map(str::trim), Some("1" | "true" | "on" | "yes"))
}

/// Turns ggml's Metal tensor API off for this process and its children.
pub fn disable() {
    let ggml_disable = std::env::var(GGML_DISABLE).ok();
    let hex_opt_in = std::env::var(HEX_OPT_IN).ok();
    if should_disable(ggml_disable.as_deref(), hex_opt_in.as_deref()) {
        // SAFETY: `main` calls this before any other thread exists to read
        // the environment.
        unsafe { std::env::set_var(GGML_DISABLE, "1") };
    }
}

#[cfg(test)]
mod tests {
    use super::should_disable;

    #[test]
    fn disables_by_default() {
        assert!(should_disable(None, None));
    }

    #[test]
    fn keeps_a_value_the_user_set() {
        assert!(!should_disable(Some("1"), None));
        assert!(!should_disable(Some(""), None));
    }

    #[test]
    fn opt_in_leaves_ggml_to_decide() {
        assert!(!should_disable(None, Some("1")));
        assert!(!should_disable(None, Some("true")));
        assert!(should_disable(None, Some("0")));
    }
}
