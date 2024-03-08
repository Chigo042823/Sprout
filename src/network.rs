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
        for i in 0..epochs { // Epochs
            if i % 1000 == 0 {
                println!("Progress: {}%", 100.0 * (i as f32 / epochs as f32));
            }
            let samples = inputs.len();
            let batches = f64::ceil(samples as f64 / self.batch_size as f64);

            for j in 0..batches as usize { // Batches
                let batch_start = j * self.batch_size;
                let batch_end = (batch_start + self.batch_size).min(samples);
                let mut batch_targets: Vec<Vec<f64>> = vec![];
                let mut batch_outputs: Vec<Vec<f64>> = vec![];

                for k in batch_start..batch_end {
                    batch_outputs.push(self.forward(inputs[k].clone()));
                    batch_targets.push(targets[k].clone());
                }

                let mut loss_gradient: Vec<f64> = vec![0.0; batch_targets[0].len()];

                for k in 0..batch_targets.len() { // Each Sample
                    for l in 0..batch_targets[k].len() { // Each Output Node
                        loss_gradient[l] += (2.0 * (batch_outputs[k][l] - batch_targets[k][l])) / self.batch_size as f64;
                    }
                }
                
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