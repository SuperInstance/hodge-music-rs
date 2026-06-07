//! Harmonic component: irreconcilable creative tension.
//!
//! These disagreements are neither resolvable (gradient) nor cyclic (curl).
//! They represent genuine creative differences that must be embraced.

use crate::graph::DisagreementGraph;
use crate::decomposition::hodge_decompose;

/// Harmonic analysis for creative tension.
pub struct HarmonicAnalysis;

impl HarmonicAnalysis {
    /// Compute the harmonic energy (creative tension) in the graph.
    pub fn creative_tension_energy(graph: &DisagreementGraph) -> f64 {
        let components = hodge_decompose(graph);
        components.harmonic_energy()
    }

    /// Fraction of total disagreement that is creative tension.
    pub fn creative_tension_fraction(graph: &DisagreementGraph) -> f64 {
        let components = hodge_decompose(graph);
        components.harmonic_fraction()
    }

    /// Classify the ensemble's disagreement state.
    pub fn classify(graph: &DisagreementGraph) -> DisagreementState {
        let components = hodge_decompose(graph);
        let gf = components.gradient_fraction();
        let cf = components.curl_fraction();
        let hf = components.harmonic_fraction();

        if hf > 0.5 {
            DisagreementState::CreativeTension
        } else if cf > 0.5 {
            DisagreementState::CyclicConflict
        } else if gf > 0.5 {
            DisagreementState::Resolvable
        } else {
            DisagreementState::Mixed
        }
    }

    /// Per-agent creative tension score.
    pub fn agent_tension(graph: &DisagreementGraph) -> Vec<f64> {
        let components = hodge_decompose(graph);
        let mut tension = vec![0.0; graph.n_agents];
        let deg = graph.degrees();

        for (i, e) in graph.edges.iter().enumerate() {
            if i < components.harmonic.len() {
                let h = components.harmonic[i].abs();
                tension[e.from] += h;
                tension[e.to] += h;
            }
        }

        for i in 0..graph.n_agents {
            if deg[i] > 0 {
                tension[i] /= deg[i] as f64;
            }
        }

        tension
    }

    /// Check if creative tension is healthy (moderate level).
    pub fn is_healthy_tension(graph: &DisagreementGraph, min: f64, max: f64) -> bool {
        let fraction = Self::creative_tension_fraction(graph);
        fraction >= min && fraction <= max
    }

    /// Suggest how to use creative tension productively.
    pub fn tension_quality(graph: &DisagreementGraph) -> TensionQuality {
        let components = hodge_decompose(graph);
        let hf = components.harmonic_fraction();

        if hf < 0.1 {
            TensionQuality::None
        } else if hf < 0.3 {
            TensionQuality::Spice
        } else if hf < 0.6 {
            TensionQuality::Productive
        } else if hf < 0.8 {
            TensionQuality::Intense
        } else {
            TensionQuality::Overwhelming
        }
    }
}

/// Disagreement state classification.
#[derive(Debug, Clone, PartialEq)]
pub enum DisagreementState {
    /// Mostly resolvable differences.
    Resolvable,
    /// Mostly cyclic conflicts.
    CyclicConflict,
    /// Mostly creative tension.
    CreativeTension,
    /// Mixed, no dominant component.
    Mixed,
}

/// Quality of creative tension.
#[derive(Debug, Clone, PartialEq)]
pub enum TensionQuality {
    /// No creative tension.
    None,
    /// Light spice — just enough to be interesting.
    Spice,
    /// Productive — drives creativity.
    Productive,
    /// Intense — high creative energy.
    Intense,
    /// Overwhelming — may need mediation.
    Overwhelming,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creative_tension_empty() {
        let g = DisagreementGraph::new(3);
        let energy = HarmonicAnalysis::creative_tension_energy(&g);
        assert!((energy).abs() < 1e-10);
    }

    #[test]
    fn test_classify_resolvable() {
        let mut g = DisagreementGraph::new(2);
        g.add_edge(0, 1, 2.0);
        let state = HarmonicAnalysis::classify(&g);
        // With 2 agents and 1 edge, should be mostly gradient/resolvable
        assert!(matches!(state, DisagreementState::Resolvable | DisagreementState::Mixed));
    }

    #[test]
    fn test_agent_tension() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let tension = HarmonicAnalysis::agent_tension(&g);
        assert_eq!(tension.len(), 3);
    }

    #[test]
    fn test_healthy_tension() {
        let mut g = DisagreementGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let is_healthy = HarmonicAnalysis::is_healthy_tension(&g, 0.0, 1.0);
        assert!(is_healthy);
    }

    #[test]
    fn test_tension_quality() {
        let mut g = DisagreementGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let quality = HarmonicAnalysis::tension_quality(&g);
        assert!(matches!(quality, TensionQuality::None | TensionQuality::Spice | TensionQuality::Productive));
    }
}
