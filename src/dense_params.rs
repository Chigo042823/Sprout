use rand::{thread_rng, Rng};
use serde_derive::{Serialize, Deserialize};

use crate::activation::ActivationFunction;

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
        let params = DenseParams {
            nodes_in,
            nodes_out,
            outputs: vec![],
            inputs: vec![],
            weights,
            biases,
        };
        params
    }
    pub fn init(&mut self, activation: ActivationFunction) {
        match activation {
            ActivationFunction::Sigmoid => 
                {
                    let std_dev = (1.0 / ((self.nodes_in + self.nodes_out) as f64 / 2.0)).sqrt();
                    let limit = (3.0 * std_dev).sqrt();
                    for i in 0..self.weights.len() {
                        for j in 0..self.weights[i].len() {
                            self.weights[i][j] = thread_rng().gen_range(-limit..limit) * std_dev; //in (rows) - out (cols)
                        }
                    }
                    self.biases = vec![0.0; self.biases.len()];
                },
            ActivationFunction::ReLU => 
                {
                    let std_dev = (2.0 / self.nodes_in as f64).sqrt();
                    let limit = (3.0 * std_dev).sqrt();
                    for i in 0..self.weights.len() {
                        for j in 0..self.weights[i].len() {
                            self.weights[i][j] = thread_rng().gen_range(-limit..limit) * std_dev; //in (rows) - out (cols)
                        }
                    }
                    self.biases = vec![0.0; self.biases.len()];
                },
            ActivationFunction::TanH => 
                {
                    let std_dev = (1.0 / ((self.nodes_in + self.nodes_out) as f64 / 2.0)).sqrt();
                    let limit = (3.0 * std_dev).sqrt();
                    for i in 0..self.weights.len() {
                        for j in 0..self.weights[i].len() {
                            self.weights[i][j] = thread_rng().gen_range(-limit..limit) * std_dev; //in (rows) - out (cols)
                        }
                    }
                    self.biases = vec![0.0; self.biases.len()];
                },
            ActivationFunction::SoftMax => 
                {
                    let std_dev = (1.0 / ((self.nodes_in + self.nodes_out) as f64 / 2.0)).sqrt();
                    let limit = (3.0 * std_dev).sqrt();
                    for i in 0..self.weights.len() {
                        for j in 0..self.weights[i].len() {
                            self.weights[i][j] = thread_rng().gen_range(-limit..limit) * std_dev; //in (rows) - out (cols)
                        }
                    }
                    self.biases = vec![0.0; self.biases.len()];
                },
        }
    }
}