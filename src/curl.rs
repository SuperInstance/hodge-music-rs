//! Curl component: cyclic disagreements.
//!
//! These are disagreements that form cycles: A disagrees with B, B with C, C with A.
//! They cannot be resolved by simple adjustment but can be broken by removing edges.

use crate::graph::DisagreementGraph;

/// Curl analysis for cyclic disagreements.
pub struct CurlAnalysis;

impl CurlAnalysis {
    /// Find all 3-cycles (triangles) in the graph.
    /// Returns list of (agent_a, agent_b, agent_c) forming cycles.
    pub fn find_triangles(graph: &DisagreementGraph) -> Vec<(usize, usize, usize)> {
        let n = graph.n_agents;
        let mut adj = vec![vec![false; n]; n];
        for e in &graph.edges {
            adj[e.from][e.to] = true;
        }

        let mut triangles = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if !adj[i][j] && !adj[j][i] {
                    continue;
                }
                for k in (j + 1)..n {
                    let ij = adj[i][j] || adj[j][i];
                    let jk = adj[j][k] || adj[k][j];
                    let ki = adj[k][i] || adj[i][k];
                    if ij && jk && ki {
                        triangles.push((i, j, k));
                    }
                }
            }
        }
        triangles
    }

    /// Compute circulation around a triangle.
    /// Positive = clockwise cycle, negative = counterclockwise.
    pub fn circulation(graph: &DisagreementGraph, tri: (usize, usize, usize)) -> f64 {
        let (a, b, c) = tri;
        let mut circ = 0.0;
        for e in &graph.edges {
            // Forward edges (a->b, b->c, c->a) add to circulation
            let is_forward = (e.from == a && e.to == b)
                || (e.from == b && e.to == c)
                || (e.from == c && e.to == a);
            // Reverse edges subtract
            let is_reverse = (e.from == b && e.to == a)
                || (e.from == c && e.to == b)
                || (e.from == a && e.to == c);

            if is_forward {
                circ += e.weight;
            } else if is_reverse {
                circ -= e.weight;
            }
        }
        circ
    }

    /// Total curl energy across all cycles.
    pub fn total_curl_energy(graph: &DisagreementGraph) -> f64 {
        let triangles = Self::find_triangles(graph);
        triangles.iter()
            .map(|tri| {
                let c = Self::circulation(graph, *tri);
                c * c
            })
            .sum()
    }

    /// Detect if the graph has cyclic disagreements.
    pub fn has_cycles(graph: &DisagreementGraph) -> bool {
        !Self::find_triangles(graph).is_empty()
    }

    /// Break cycles by finding the edge with highest cyclic contribution.
    /// Returns the index of the edge to remove.
    pub fn suggest_edge_to_break(graph: &DisagreementGraph) -> Option<usize> {
        let triangles = Self::find_triangles(graph);
        if triangles.is_empty() {
            return None;
        }

        let mut best_edge = 0;
        let mut best_score = 0.0;

        for (idx, e) in graph.edges.iter().enumerate() {
            let mut score = 0.0;
            for tri in &triangles {
                let (a, b, c) = *tri;
                if (e.from == a || e.from == b || e.from == c) &&
                   (e.to == a || e.to == b || e.to == c) {
                    score += e.weight.abs();
                }
            }
            if score > best_score {
                best_score = score;
                best_edge = idx;
            }
        }

        Some(best_edge)
    }

    /// Compute the curl vector for each edge.
    /// Edges that participate in cycles get non-zero values.
    pub fn curl_vector(graph: &DisagreementGraph) -> Vec<f64> {
        let triangles = Self::find_triangles(graph);
        let mut curl = vec![0.0; graph.edges.len()];

        for tri in &triangles {
            let circ = Self::circulation(graph, *tri);
            if circ.abs() < 1e-10 {
                continue;
            }
            let third = circ / 3.0;
            let (a, b, c) = *tri;
            for (idx, e) in graph.edges.iter().enumerate() {
                let in_triangle = (e.from == a || e.from == b || e.from == c) &&
                                  (e.to == a || e.to == b || e.to == c);
                if in_triangle {
                    let sign = if (e.from == a && e.to == b) ||
                                  (e.from == b && e.to == c) ||
                                  (e.from == c && e.to == a) {
                        1.0
                    } else {
                        -1.0
                    };
                    curl[idx] += sign * third;
                }
            }
        }
        curl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_triangles() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 0, 1.0);
        let tris = CurlAnalysis::find_triangles(&g);
        assert_eq!(tris.len(), 1);
        assert_eq!(tris[0], (0, 1, 2));
    }

    #[test]
    fn test_no_triangles() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let tris = CurlAnalysis::find_triangles(&g);
        assert!(tris.is_empty());
    }

    #[test]
    fn test_circulation() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 2.0);
        g.add_edge(1, 2, 3.0);
        g.add_edge(2, 0, -1.0);
        let circ = CurlAnalysis::circulation(&g, (0, 1, 2));
        assert!((circ - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_has_cycles() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 0, 1.0);
        assert!(CurlAnalysis::has_cycles(&g));
    }

    #[test]
    fn test_suggest_break() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 5.0);
        g.add_edge(2, 0, 1.0);
        let suggestion = CurlAnalysis::suggest_edge_to_break(&g);
        assert!(suggestion.is_some());
        // Edge 1 (weight 5.0) should be the one to break
        assert_eq!(suggestion.unwrap(), 1);
    }
}
