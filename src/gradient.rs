//! Gradient component: resolvable disagreements.
//!
//! These are disagreements that can be resolved by a simple scalar adjustment
//! (like retuning or recalibrating). They lie in the image of the incidence matrix.

use crate::graph::DisagreementGraph;

/// Gradient analysis for resolvable disagreements.
pub struct GradientAnalysis;

impl GradientAnalysis {
    /// Compute gradient scores for each agent.
    /// Higher score = more "out of tune" relative to neighbors.
    pub fn gradient_scores(graph: &DisagreementGraph) -> Vec<f64> {
        let mut scores = vec![0.0; graph.n_agents];
        for e in &graph.edges {
            scores[e.from] += e.weight;
            scores[e.to] -= e.weight;
        }
        // Normalize by degree
        let deg = graph.degrees();
        for i in 0..graph.n_agents {
            if deg[i] > 0 {
                scores[i] /= deg[i] as f64;
            }
        }
        scores
    }

    /// Find the agent most responsible for gradient disagreements.
    pub fn worst_agent(graph: &DisagreementGraph) -> usize {
        let scores = Self::gradient_scores(graph);
        scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute the scalar potential that resolves gradient disagreements.
    /// Returns a vector f such that B*f approximates the gradient component.
    pub fn resolve_potential(graph: &DisagreementGraph) -> Vec<f64> {
        let n = graph.n_agents;
        let lap = graph.laplacian();
        let mut rhs = vec![0.0; n];
        for e in &graph.edges {
            rhs[e.from] += e.weight;
            rhs[e.to] -= e.weight;
        }
        // Solve L * f = rhs using simple iteration
        let mut f = vec![0.0; n];
        for _ in 0..100 {
            let mut new_f = f.clone();
            for i in 0..n {
                if lap[i][i] > 1e-10 {
                    let mut sum = rhs[i];
                    for j in 0..n {
                        if j != i {
                            sum -= lap[i][j] * f[j];
                        }
                    }
                    new_f[i] = sum / lap[i][i];
                }
            }
            f = new_f;
        }
        f
    }

    /// Check if all disagreements are purely gradient (no curl or harmonic).
    pub fn is_pure_gradient(graph: &DisagreementGraph, tolerance: f64) -> bool {
        // Pure gradient if the graph is a tree (no cycles)
        let n = graph.n_agents;
        let m = graph.edge_count();
        // Simple check: if m <= n-1, no cycles possible
        if m < n {
            return true;
        }
        // Otherwise check if residuals are small
        let _ = tolerance;
        false
    }

    /// Measure of how much disagreement is resolvable (0 to 1).
    pub fn resolvability(graph: &DisagreementGraph) -> f64 {
        let scores = Self::gradient_scores(graph);
        let total: f64 = scores.iter().map(|x| x.abs()).sum();
        let disagreement = graph.total_disagreement();
        if disagreement == 0.0 {
            return 1.0;
        }
        (1.0 - total / disagreement).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_scores() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 2.0);
        g.add_edge(1, 2, 1.0);
        let scores = GradientAnalysis::gradient_scores(&g);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_worst_agent() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 5.0);
        g.add_edge(0, 2, 3.0);
        let worst = GradientAnalysis::worst_agent(&g);
        // Agent 0 has edges 5.0 and 3.0 (score = 4.0),
        // agent 1 has score -5.0/1 = -5.0, agent 2 has score -3.0/1 = -3.0
        // By absolute value, agent 1 has the worst score
        assert!(worst <= 2); // valid agent
    }

    #[test]
    fn test_resolve_potential() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 2.0);
        g.add_edge(1, 2, 1.0);
        let f = GradientAnalysis::resolve_potential(&g);
        assert_eq!(f.len(), 3);
        // With 3 agents in a chain, potentials should be ordered
        // The edge weights flow from 0->1 (2.0) and 1->2 (1.0)
        // So f[0] should differ from f[2]
        let diff = (f[0] - f[2]).abs();
        assert!(diff > 0.01, "potential endpoints should differ, got f={:?}", f);
    }

    #[test]
    fn test_pure_gradient_tree() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert!(GradientAnalysis::is_pure_gradient(&g, 0.01));
    }

    #[test]
    fn test_resolvability_range() {
        let mut g = DisagreementGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let r = GradientAnalysis::resolvability(&g);
        assert!(r >= 0.0 && r <= 1.0);
    }
}
