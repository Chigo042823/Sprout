use std::vec;

use rand::{thread_rng, Rng};

use crate::activation::{Activation, ActivationFunction};
use serde_derive::*;

pub enum LayerType {
    Dense,
    Convolutional,
}
pub trait Layer {
    fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64>;
    fn backward(&mut self, loss_gradient: Vec<f64>, learning_rate: f64) -> Vec<f64>;
    fn get_weights(&self) -> Vec<Vec<f64>>;
    fn get_biases(&self) -> Vec<f64>;
    fn get_outputs(&self) -> Vec<f64>;
    fn get_layer_type(&self) -> LayerType;
    fn get_nodes(&self) -> usize;
    fn get_input_nodes(&self) -> usize;
    fn set_params(&mut self, weights: Vec<Vec<f64>>, biases: Vec<f64>);
    fn reset(&mut self);
    fn init(&mut self);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    pub nodes_in: usize,
    pub nodes_out: usize,
    pub outputs: Vec<f64>, //nodes out
    pub inputs: Vec<f64>,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: Activation,
}

impl DenseLayer {
    pub fn new(nodes_in: usize, nodes_out: usize, activation_fn: ActivationFunction) -> Self {
        let weights = vec![vec![0.0; nodes_out]; nodes_in];
        let biases = vec![0.0; nodes_out];

        let mut layer = DenseLayer {
            nodes_in,
            nodes_out,
            outputs: vec![],
            inputs: vec![],
            weights,
            biases,
            activation: Activation::new(activation_fn),
        };
        layer.init();
        layer
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        self.inputs = inputs.clone();
        let mut weighted_inputs = self.biases.clone();
        for i in 0..self.nodes_out {
            for j in 0..self.nodes_in {
                weighted_inputs[i] += inputs[j] * self.weights[j][i];
            }
        }

        let mut activation = vec![0.0; self.nodes_out];
        for i in 0..self.nodes_out {
            activation[i] = self.activation.function(weighted_inputs[i]);
        }
        self.outputs = activation.clone();
        activation
    }

    fn backward(&mut self, errors: Vec<f64>, learning_rate: f64) -> Vec<f64> {
        let mut delta_output = errors.clone();
        for i in 0..delta_output.len() {
            delta_output[i] *= self.activation.derivative(self.outputs[i].clone());
        }

        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] -= learning_rate * (self.inputs[i] * delta_output[j]);
            }
        }

        for i in 0..self.biases.len() {
            self.biases[i] -= learning_rate * delta_output[i];
        }

        let mut next_delta = vec![0.0; self.nodes_in];
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                next_delta[i] += self.weights[i][j] * delta_output[j] * self.inputs[i];
            }   
        }

        next_delta
    }

    fn get_weights(&self) -> Vec<Vec<f64>> {
        self.weights.clone()
    }

    fn get_biases(&self) -> Vec<f64> {
        self.biases.clone()
    }

    fn get_nodes(&self) -> usize {
        self.nodes_out.clone()
    }

    fn get_input_nodes(&self) -> usize {
        self.nodes_in.clone()
    }

    fn get_outputs(&self) -> Vec<f64> {
        self.outputs.clone()
    }

    fn get_layer_type(&self) -> LayerType {
        LayerType::Dense
    }

    fn init(&mut self) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] = thread_rng().gen_range(-0.1..0.1); //in (rows) - out (cols)
            }
        }
        for i in 0..self.biases.len() {
            self.biases[i] = thread_rng().gen_range(-0.1..0.1); //nodes out
        }
    }

    fn reset(&mut self) {
        self.init();
        self.inputs = vec![];
        self.outputs = vec![];
    }

    fn set_params(&mut self, weights: Vec<Vec<f64>>, biases: Vec<f64>) {
        self.weights = weights;
        self.biases = biases;
    }
}