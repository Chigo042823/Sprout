use rand::{thread_rng, Rng};
use serde_derive::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseParams {
    pub nodes_in: usize,
    pub nodes_out: usize,
    pub outputs: Vec<f64>, //nodes out
    pub inputs: Vec<f64>,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl DenseParams {
    pub fn new(
        nodes_in: usize,
        nodes_out: usize,
    ) -> Self {
        let weights = vec![vec![0.0; nodes_out]; nodes_in];
        let biases = vec![0.0; nodes_out];
        let mut params = DenseParams {
            nodes_in,
            nodes_out,
            outputs: vec![],
            inputs: vec![],
            weights,
            biases,
        };
        params.init();
        params
    }
    pub fn init(&mut self) {
        let std_dev = (2.0 / (self.nodes_in + self.nodes_out) as f64).sqrt();
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] = thread_rng().gen_range(-0.5..0.5) * std_dev; //in (rows) - out (cols)
            }
        }
        self.biases = vec![0.0; self.biases.len()];
    }
}