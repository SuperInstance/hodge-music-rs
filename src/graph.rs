//! Musical disagreement graph.

/// An edge in the disagreement graph.
#[derive(Debug, Clone)]
pub struct DisagreementEdge {
    pub from: usize,
    pub to: usize,
    /// Signed disagreement value. Positive = from wants more than to.
    pub weight: f64,
}

impl DisagreementEdge {
    pub fn new(from: usize, to: usize, weight: f64) -> Self {
        Self { from, to, weight }
    }
}

/// Musical disagreement graph over an ensemble.
#[derive(Debug, Clone)]
pub struct DisagreementGraph {
    /// Number of agents (vertices).
    pub n_agents: usize,
    /// Directed, weighted edges representing disagreements.
    pub edges: Vec<DisagreementEdge>,
}

impl DisagreementGraph {
    /// Create a new graph with n_agents and no edges.
    pub fn new(n_agents: usize) -> Self {
        Self { n_agents, edges: Vec::new() }
    }

    /// Add a disagreement edge.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        assert!(from < self.n_agents && to < self.n_agents, "agent index out of bounds");
        self.edges.push(DisagreementEdge::new(from, to, weight));
    }

    /// Get all edges involving a given agent.
    pub fn edges_for(&self, agent: usize) -> Vec<&DisagreementEdge> {
        self.edges.iter().filter(|e| e.from == agent || e.to == agent).collect()
    }

    /// Compute the degree (number of edges) for each agent.
    pub fn degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.n_agents];
        for e in &self.edges {
            deg[e.from] += 1;
            deg[e.to] += 1;
        }
        deg
    }

    /// Total disagreement magnitude.
    pub fn total_disagreement(&self) -> f64 {
        self.edges.iter().map(|e| e.weight.abs()).sum()
    }

    /// Build the signed incidence matrix (n_agents x n_edges).
    /// For edge j from u to v: B[u][j] = +1, B[v][j] = -1, rest 0.
    pub fn incidence_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n_agents;
        let m = self.edges.len();
        let mut b = vec![vec![0.0; m]; n];
        for (j, e) in self.edges.iter().enumerate() {
            b[e.from][j] = 1.0;
            b[e.to][j] = -1.0;
        }
        b
    }

    /// Build the graph Laplacian (n_agents x n_agents).
    /// L = D - A where D is degree matrix and A is adjacency.
    pub fn laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.n_agents;
        let mut l = vec![vec![0.0; n]; n];
        for e in &self.edges {
            l[e.from][e.from] += 1.0;
            l[e.to][e.to] += 1.0;
            l[e.from][e.to] -= 1.0;
            l[e.to][e.from] -= 1.0;
        }
        l
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_graph() {
        let g = DisagreementGraph::new(3);
        assert_eq!(g.n_agents, 3);
        assert!(g.edges.is_empty());
    }

    #[test]
    fn test_add_edges() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 2.0);
        g.add_edge(1, 2, -1.0);
        assert_eq!(g.edges.len(), 2);
    }

    #[test]
    fn test_degrees() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let deg = g.degrees();
        assert_eq!(deg[0], 1);
        assert_eq!(deg[1], 2);
        assert_eq!(deg[2], 1);
    }

    #[test]
    fn test_total_disagreement() {
        let mut g = DisagreementGraph::new(2);
        g.add_edge(0, 1, 3.0);
        g.add_edge(1, 0, -2.0);
        assert!((g.total_disagreement() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_incidence_matrix() {
        let mut g = DisagreementGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let b = g.incidence_matrix();
        assert!((b[0][0] - 1.0).abs() < 1e-10);
        assert!((b[1][0] - (-1.0)).abs() < 1e-10);
    }
}
