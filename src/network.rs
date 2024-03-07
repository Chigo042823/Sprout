use std::time;

use crate::layer::Layer;


pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
    pub learning_rate: f64
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>, learning_rate: f64) -> Self {
        Network {
            layers,
            learning_rate,
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut current: Vec<f64> = inputs; 
        for i in 0..self.layers.len() {
            current = self.layers[i].forward(current);
        }
        current
    }

    pub fn backward(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
        if outputs.len() != targets.len() {
            println!("Ouputs: {} and targets: {} are not compatible", outputs.len(), targets.len());
            panic!();
        }

        let mut delta_output = Vec::with_capacity(outputs.len());
        for i in 0..outputs.len() {
            delta_output.push(2.0 * (outputs[i] - targets[i]));
        }

        for i in (0..self.layers.len()).rev() {
            let layer = self.layers[i].as_mut();
            delta_output = layer.backward(delta_output.clone(), self.learning_rate);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        for i in 0..epochs {
            for j in 0..inputs.len() {
                let ftime = time::Instant::now();
                let outputs = self.forward(inputs[j].clone());
                println!("Epoch: {} / {epochs} || Cost: {}", i + 1, self.get_cost(&targets[j], &outputs));
                let btime = time::Instant::now();
                self.backward(outputs, targets[j].clone());
                println!("Forward: {:?} Backward: {:?}", ftime.elapsed(), btime.elapsed());
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