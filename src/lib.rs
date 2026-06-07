//! Hodge decomposition of musical disagreements.
//!
//! Decomposes pairwise disagreements in an ensemble into:
//! - Gradient component: resolvable differences (can be fixed by adjustment)
//! - Curl component: cyclic disagreements (A > B > C > A)
//! - Harmonic component: irreconcilable creative tension

pub mod graph;
pub mod decomposition;
pub mod gradient;
pub mod curl;
pub mod harmonic;
pub mod consensus;
