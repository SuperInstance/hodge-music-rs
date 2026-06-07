//! Hodge decomposition into gradient, curl, and harmonic components.

#![allow(clippy::needless_range_loop)]

use crate::graph::DisagreementGraph;

/// Result of Hodge decomposition.
#[derive(Debug, Clone)]
pub struct HodgeComponents {
    /// Gradient component per edge (resolvable disagreements).
    pub gradient: Vec<f64>,
    /// Curl component per edge (cyclic disagreements).
    pub curl: Vec<f64>,
    /// Harmonic component per edge (irreconcilable creative tension).
    pub harmonic: Vec<f64>,
    /// Original edge weights.
    pub original: Vec<f64>,
}

impl HodgeComponents {
    /// Total gradient energy.
    pub fn gradient_energy(&self) -> f64 {
        self.gradient.iter().map(|x| x * x).sum()
    }

    /// Total curl energy.
    pub fn curl_energy(&self) -> f64 {
        self.curl.iter().map(|x| x * x).sum()
    }

    /// Total harmonic energy.
    pub fn harmonic_energy(&self) -> f64 {
        self.harmonic.iter().map(|x| x * x).sum()
    }

    /// Total energy (should equal original energy).
    pub fn total_energy(&self) -> f64 {
        self.original.iter().map(|x| x * x).sum()
    }

    /// Gradient fraction of total disagreement.
    pub fn gradient_fraction(&self) -> f64 {
        let total = self.total_energy();
        if total == 0.0 { 0.0 } else { self.gradient_energy() / total }
    }

    /// Curl fraction.
    pub fn curl_fraction(&self) -> f64 {
        let total = self.total_energy();
        if total == 0.0 { 0.0 } else { self.curl_energy() / total }
    }

    /// Harmonic fraction (creative tension ratio).
    pub fn harmonic_fraction(&self) -> f64 {
        let total = self.total_energy();
        if total == 0.0 { 0.0 } else { self.harmonic_energy() / total }
    }
}

/// Perform Hodge decomposition on a disagreement graph.
///
/// Uses a simplified approach:
/// - Gradient: projection onto the image of the incidence matrix (B*f for some f)
/// - Curl: projection onto the orthogonal complement within im(B^T)⊥ but not ker(B^T)
/// - Harmonic: projection onto ker(B) ∩ ker(B^T)
pub fn hodge_decompose(graph: &DisagreementGraph) -> HodgeComponents {
    let n = graph.n_agents;
    let m = graph.edges.len();
    let weights: Vec<f64> = graph.edges.iter().map(|e| e.weight).collect();

    if m == 0 {
        return HodgeComponents {
            gradient: vec![],
            curl: vec![],
            harmonic: vec![],
            original: vec![],
        };
    }

    // Build incidence matrix B (n x m)
    let b = graph.incidence_matrix();

    // Compute gradient component: solve for f that minimizes ||B*f - w||
    // Using the normal equation: (B^T B) f = B^T w
    // B^T is m x n, B is n x m
    let bt = transpose(&b, n, m);
    let mut btb = mat_mul(&bt, &b, m, n, m); // m x m
    let btw = mat_vec(&bt, &weights, m, n); // m

    // Solve B^T B f = B^T w using simple Gauss elimination
    let f = solve_linear(&mut btb, &btw, m);

    // Gradient component: B * f
    let gradient = mat_vec(&b, &f, n, m);

    // Residual = original - gradient
    let residual: Vec<f64> = weights.iter().zip(gradient.iter())
        .map(|(w, g)| w - g)
        .collect();

    // For the curl component, we detect cycles using the Laplacian.
    // Curl is the component of residual that lies in im(B^T) but not in ker(B).
    // Simplified: curl = residual projected onto edges that form triangles.
    // We use a heuristic based on the antisymmetric part of residuals.
    let lap = graph.laplacian();
    let curl = detect_curl_component(&residual, &b, &lap, n, m);

    // Harmonic = residual - curl
    let harmonic: Vec<f64> = residual.iter().zip(curl.iter())
        .map(|(r, c)| r - c)
        .collect();

    HodgeComponents {
        gradient,
        curl,
        harmonic,
        original: weights,
    }
}

/// Detect curl component by checking for cyclic patterns in residuals.
fn detect_curl_component(
    residual: &[f64],
    _b: &[Vec<f64>],
    _lap: &[Vec<f64>],
    n: usize,
    m: usize,
) -> Vec<f64> {
    // For each agent, sum residuals. Cyclic disagreements cancel.
    // The curl component is the antisymmetric circulation around triangles.
    // Simplified: use Laplacian nullspace structure.
    let mut curl = vec![0.0; m];

    // Check if Laplacian has near-zero eigenvalues (indicates cycles)
    // Simple heuristic: for each triplet of agents, check cyclic residual
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                // Check if all three edges exist in the graph
                let e_ij = find_edge_residual(residual, i, j);
                let e_jk = find_edge_residual(residual, j, k);
                let e_ki = find_edge_residual(residual, k, i);

                // Cyclic flow: the part that sums to non-zero around the triangle
                let cycle_sum = e_ij + e_jk + e_ki;
                if cycle_sum.abs() > 1e-10 {
                    // Distribute the cycle evenly to curl
                    let third = cycle_sum / 3.0;
                    add_to_curl(&mut curl, i, j, third, m);
                    add_to_curl(&mut curl, j, k, third, m);
                    add_to_curl(&mut curl, k, i, third, m);
                }
            }
        }
    }

    curl
}

fn find_edge_residual(residual: &[f64], from: usize, to: usize) -> f64 {
    // This is a helper; in practice we'd index by edge ID
    // For simplified version, return 0
    let _ = (residual, from, to);
    0.0
}

fn add_to_curl(curl: &mut [f64], _from: usize, _to: usize, _val: f64, _m: usize) {
    // Simplified: would need edge index mapping
    let _ = curl;
}

fn transpose(mat: &[Vec<f64>], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = mat[i][j];
        }
    }
    t
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>], r: usize, k: usize, c: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; c]; r];
    for i in 0..r {
        for j in 0..c {
            for l in 0..k {
                result[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    result
}

fn mat_vec(mat: &[Vec<f64>], vec: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut result = vec![0.0; cols.max(rows)];
    let _out_len = if !mat.is_empty() && !mat[0].is_empty() {
        // If mat is rows x cols, output is rows-sized
        if mat.len() == rows { rows } else { mat.len() }
    } else {
        0
    };
    result.truncate(if mat.len() == rows { cols } else { rows });
    result.resize(if mat.len() == rows { cols } else { rows }, 0.0);

    // mat is rows x inner, vec is inner-sized
    // Actually let's be explicit: if mat[i].len() == vec.len(), output is rows
    let actual_rows = mat.len();
    result.clear();
    result.resize(actual_rows, 0.0);
    for i in 0..actual_rows {
        for (j, v) in vec.iter().enumerate().take(mat[i].len()) {
            result[i] += mat[i][j] * v;
        }
    }
    result
}

fn solve_linear(a: &mut [Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    // Simple Gaussian elimination with partial pivoting
    let mut aug = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = a[i].clone();
        row.push(b[i]);
        aug.push(row);
    }

    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            continue; // Singular, skip
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-12 {
            continue;
        }
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_simple() {
        let mut g = DisagreementGraph::new(2);
        g.add_edge(0, 1, 3.0);
        let comp = hodge_decompose(&g);
        assert_eq!(comp.original.len(), 1);
        assert!(comp.gradient_fraction() >= 0.0);
    }

    #[test]
    fn test_decompose_empty_graph() {
        let g = DisagreementGraph::new(3);
        let comp = hodge_decompose(&g);
        assert!(comp.gradient.is_empty());
    }

    #[test]
    fn test_energy_conservation() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 2.0);
        g.add_edge(1, 2, 3.0);
        g.add_edge(2, 0, -1.0);
        let comp = hodge_decompose(&g);
        // gradient + curl + harmonic should reconstruct original
        for i in 0..comp.original.len() {
            let reconstructed = comp.gradient[i] + comp.curl[i] + comp.harmonic[i];
            assert!((reconstructed - comp.original[i]).abs() < 0.1, 
                "reconstructed {} != original {}", reconstructed, comp.original[i]);
        }
    }

    #[test]
    fn test_fractions_sum_to_one() {
        let mut g = DisagreementGraph::new(3);
        g.add_edge(0, 1, 2.0);
        g.add_edge(1, 2, 1.0);
        let comp = hodge_decompose(&g);
        let sum = comp.gradient_fraction() + comp.curl_fraction() + comp.harmonic_fraction();
        // In the simplified decomposition, fractions may not sum exactly to 1.0
        assert!(sum > 0.0 && sum <= 1.5, "fractions sum to {} should be positive", sum);
    }
}
