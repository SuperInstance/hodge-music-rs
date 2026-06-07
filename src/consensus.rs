//! Consensus measurement from Hodge decomposition.

use crate::graph::DisagreementGraph;
use crate::decomposition::hodge_decompose;

/// Consensus metrics derived from disagreement decomposition.
pub struct ConsensusMeasure;

impl ConsensusMeasure {
    /// Overall consensus level (0 = no agreement, 1 = full agreement).
    pub fn consensus_level(graph: &DisagreementGraph) -> f64 {
        if graph.edges.is_empty() {
            return 1.0; // No disagreements = full consensus
        }
        let total = graph.total_disagreement();
        if total == 0.0 {
            return 1.0;
        }
        let components = hodge_decompose(graph);
        // Consensus is high when gradient fraction is high (resolvable)
        // and harmonic fraction is low
        1.0 - components.harmonic_fraction()
    }

    /// Consensus level considering only the gradient (resolvable) component.
    pub fn effective_consensus(graph: &DisagreementGraph) -> f64 {
        let components = hodge_decompose(graph);
        components.gradient_fraction()
    }

    /// Per-agent agreement score.
    pub fn agent_agreement(graph: &DisagreementGraph) -> Vec<f64> {
        let mut agreement = vec![1.0; graph.n_agents];
        let deg = graph.degrees();

        for e in &graph.edges {
            agreement[e.from] -= e.weight.abs();
            agreement[e.to] -= e.weight.abs();
        }

        // Normalize by degree
        for i in 0..graph.n_agents {
            if deg[i] > 0 {
                let max_disagreement = graph.edges.iter()
                    .filter(|e| e.weight.abs() > 0.0)
                    .map(|e| e.weight.abs())
                    .fold(0.0_f64, f64::max)
                    .max(1e-10);
                agreement[i] = (agreement[i] / (deg[i] as f64 * max_disagreement)).clamp(0.0, 1.0);
            }
        }

        agreement
    }

    /// Whether the ensemble can reach consensus (no irreconcilable disagreements).
    pub fn can_reach_consensus(graph: &DisagreementGraph, tolerance: f64) -> bool {
        let components = hodge_decompose(graph);
        components.harmonic_fraction() <= tolerance
    }

    /// Estimate steps to consensus based on gradient magnitude.
    pub fn consensus_distance(graph: &DisagreementGraph) -> f64 {
        graph.total_disagreement()
    }

    /// Pairwise consensus matrix.
    pub fn pairwise_consensus(graph: &DisagreementGraph) -> Vec<Vec<f64>> {
        let n = graph.n_agents;
        let mut matrix = vec![vec![1.0; n]; n];

        for e in &graph.edges {
            let max_weight = graph.edges.iter()
                .map(|edge| edge.weight.abs())
                .fold(0.0_f64, f64::max)
                .max(1e-10);
            let consensus = 1.0 - (e.weight.abs() / max_weight);
            matrix[e.from][e.to] = consensus;
            matrix[e.to][e.from] = consensus;
        }

        matrix
    }

    /// Identify the pair with the most creative (irreconcilable) disagreement.
    pub fn most_creative_pair(graph: &DisagreementGraph) -> Option<(usize, usize)> {
        let components = hodge_decompose(graph);
        let mut best_pair = None;
        let mut best_harmonic = 0.0;

        for (i, e) in graph.edges.iter().enumerate() {
            if i < components.harmonic.len() && components.harmonic[i].abs() > best_harmonic {
                best_harmonic = components.harmonic[i].abs();
                best_pair = Some((e.from, e.to));
            }
        }

        best_pair
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consensus_full() {
        let g = DisagreementGraph::new(3);
        assert!((ConsensusMeasure::consensus_level(&g) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_partial() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let level = ConsensusMeasure::consensus_level(&g);
        assert!(level > 0.0 && level <= 1.0);
    }

    #[test]
    fn test_agent_agreement() {
        let mut g = DisagreementGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let agree = ConsensusMeasure::agent_agreement(&g);
        assert_eq!(agree.len(), 2);
    }

    #[test]
    fn test_can_reach_consensus() {
        let mut g = DisagreementGraph::new(2);
        g.add_edge(0, 1, 1.0);
        // Single edge, no cycles — should be reachable
        assert!(ConsensusMeasure::can_reach_consensus(&g, 0.5));
    }

    #[test]
    fn test_pairwise_consensus() {
        let mut g = DisagreementGraph::new(2);
        g.add_edge(0, 1, 0.5);
        let matrix = ConsensusMeasure::pairwise_consensus(&g);
        assert_eq!(matrix.len(), 2);
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_distance() {
        let mut g = DisagreementGraph::new(2);
        g.add_edge(0, 1, 2.0);
        assert!((ConsensusMeasure::consensus_distance(&g) - 2.0).abs() < 1e-10);
    }
}
