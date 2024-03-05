use rand::{thread_rng, Rng};

use crate::activation::{Activation, ActivationFunction};

pub trait Layer {
    fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64>;
    fn backward(&mut self,inputs: Vec<f64>, loss_gradient: Vec<f64>, learning_rate: f64) -> Vec<f64>;
    fn get_weights(&self) -> Vec<Vec<f64>>;
    fn get_biases(&self) -> Vec<f64>;
    fn get_outputs(&self) -> Vec<f64>;
}

pub struct DenseLayer {
    pub nodes_in: usize,
    pub nodes_out: usize,
    pub outputs: Vec<f64>, //nodes out
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: Activation
}

impl DenseLayer {
    pub fn new(nodes_in: usize, nodes_out: usize, activation_fn: ActivationFunction) -> Self {
        let mut weights = vec![vec![0.0; nodes_out]; nodes_in];
        let mut biases = vec![0.0; nodes_out];

        for i in 0..nodes_out {
            for j in 0..nodes_in {
                weights[j][i] = thread_rng().gen_range(-1.0..1.0); //in (rows) - out (cols)
            }
            biases[i] = thread_rng().gen_range(-1.0..1.0); //nodes out
        }

        DenseLayer {
            nodes_in,
            nodes_out,
            outputs: vec![],
            weights,
            biases,
            activation: Activation::new(activation_fn),
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let weighted_inputs = &mut self.biases;
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

    fn backward(&mut self, inputs: Vec<f64>, mut loss_gradient: Vec<f64>, learning_rate: f64) -> Vec<f64> {
        let mut weight_gradients = vec![vec![0.0; self.nodes_out]; self.nodes_in];
        let bias_gradients = loss_gradient.clone();

        for i in 0..weight_gradients.len() {
            for j in 0..weight_gradients[i].len() {
                weight_gradients[i][j] = loss_gradient[j] * self.activation.derivative(self.outputs[j]) * inputs[i];
            }
        }

        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] -= learning_rate * weight_gradients[i][j];
            }
        }

        for i in 0..self.biases.len() {
            self.biases[i] += learning_rate * (bias_gradients[i] * self.activation.derivative(self.outputs[i]));
        }

        let mut next_loss_gradient = vec![0.0; self.nodes_in];

        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                next_loss_gradient[i] += self.weights[i][j] * loss_gradient[j];
            }
        }

        next_loss_gradient
    }

    fn get_weights(&self) -> Vec<Vec<f64>> {
        self.weights.clone()
    }

    fn get_biases(&self) -> Vec<f64> {
        self.biases.clone()
    }

    fn get_outputs(&self) -> Vec<f64> {
        self.outputs.clone()
    }
}