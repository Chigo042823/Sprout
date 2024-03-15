use std::vec;

use rand::{thread_rng, Rng};

use crate::activation::{Activation, ActivationFunction};
use serde_derive::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Dense,
    Convolutional,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub nodes_in: usize,
    pub nodes_out: usize,
    pub outputs: Vec<f64>, //nodes out
    pub inputs: Vec<f64>,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: Activation,
    pub layer_type: LayerType
}

impl Layer {
    pub fn new(nodes_in: usize, nodes_out: usize, layer_type: LayerType, activation_fn: ActivationFunction) -> Self {
        let weights = vec![vec![0.0; nodes_out]; nodes_in];
        let biases = vec![0.0; nodes_out];

        let mut layer = Layer {
            nodes_in,
            nodes_out,
            outputs: vec![],
            inputs: vec![],
            weights,
            biases,
            activation: Activation::new(activation_fn),
            layer_type
        };
        layer.init();
        layer
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
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

    pub fn backward(&mut self, errors: Vec<f64>, learning_rate: f64) -> Vec<f64> {
        let mut delta_output = errors.clone();
        for i in 0..delta_output.len() {
            delta_output[i] *= self.activation.derivative(self.outputs[i].clone());
            // delta_output[i] = delta_output[i].min(5.0);
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
                next_delta[i] += (self.weights[i][j] * delta_output[j] * self.inputs[i]);
            }   
        }

        next_delta
    }

    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        self.weights.clone()
    }

    pub fn get_biases(&self) -> Vec<f64> {
        self.biases.clone()
    }

    pub fn get_nodes(&self) -> usize {
        self.nodes_out.clone()
    }

    pub fn get_input_nodes(&self) -> usize {
        self.nodes_in.clone()
    }

    pub fn get_outputs(&self) -> Vec<f64> {
        self.outputs.clone()
    }

    pub fn get_layer_type(&self) -> LayerType {
        self.layer_type.clone()
    }

    pub fn init(&mut self) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] = thread_rng().gen_range(-0.2..0.2); //in (rows) - out (cols)
            }
        }
        for i in 0..self.biases.len() {
            self.biases[i] = thread_rng().gen_range(-0.2..0.2); //nodes out
        }
    }

    pub fn reset(&mut self) {
        self.init();
        self.inputs = vec![];
        self.outputs = vec![];
    }

    pub fn set_params(&mut self, weights: Vec<Vec<f64>>, biases: Vec<f64>) {
        self.weights = weights;
        self.biases = biases;
    }
}