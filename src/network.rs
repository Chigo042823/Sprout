use std::time;

use crate::layer::Layer;


pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
    pub learning_rate: f64,
    pub batch_size: usize,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>, learning_rate: f64, batch_size: usize) -> Self {
        Network {
            layers,
            learning_rate,
            batch_size,
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut current: Vec<f64> = inputs; 
        for i in 0..self.layers.len() {
            current = self.layers[i].forward(current);
        }
        current
    }

    pub fn backward(&mut self, loss_gradient: Vec<f64>) {
        let mut delta_output = loss_gradient.clone();

        for i in (0..self.layers.len()).rev() {
            let layer = self.layers[i].as_mut();
            delta_output = layer.backward(delta_output.clone(), self.learning_rate);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        if inputs.len() % self.batch_size != 0 {
            panic!("Inputs: {} not divisible by Batch Size: {}", inputs.len(), self.batch_size);
        }

        for i in 0..epochs { // Epochs
            if i % 1000 == 0 {
                println!("Progress: {}%", 100.0 * (i as f32 / epochs as f32));
            }
            for j in (0..inputs.len()).step_by(self.batch_size) { // Batches
                let batches = inputs.len() as f64 / self.batch_size as f64;

                let batch_inputs: Vec<_> = inputs[j..j+self.batch_size].iter().collect();
                let batch_targets: Vec<_> = targets[j..j+self.batch_size].iter().collect();
                let mut batch_outputs = vec![vec![0.0; batch_targets[0].len()]; self.batch_size];
                
                for k in 0..batch_outputs.len() {
                    let outputs: Vec<_> = self.forward(batch_inputs[k].clone());
                    batch_outputs[k] = outputs;
                }

                let mut loss_gradient = vec![0.0; batch_targets[0].len()];

                for k in 0..batch_targets.len() {
                    for l in 0..batch_targets[k].len() {
                        loss_gradient[l] += (2.0 * (batch_outputs[k][l] - batch_targets[k][l])) / batches;
                    }
                }
                
                // println!("Epoch: {} / {epochs} || Cost: {}", i + 1, self.get_cost(&targets[j], &outputs));

                self.backward(loss_gradient.clone());
            }
        }
    }

    pub fn get_cost(&self, targets: &Vec<f64>, outputs: &Vec<f64>) -> f64 {
        let mut cost = 0.0;
        for i in 0..targets.len() {
            cost += (outputs[i] -  targets[i]).powf(2.0);
        }
        cost / targets.len() as f64
    }

    pub fn print_weights(&self) {
        for i in 0..self.layers.len() {
            println!("{:#?}", self.layers[i].get_weights());
        }
    }

    pub fn print_biases(&self) {
        for i in 0..self.layers.len() {
            println!("{:#?}", self.layers[i].get_biases());
        }
    }
}