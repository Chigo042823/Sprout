use rand::seq::SliceRandom;

use crate::layer::Layer;
use std::{file, fs::File, io::{Read, Write}};

pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub cost: f64,
    pub print_progress: bool
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>, learning_rate: f64, batch_size: usize) -> Self {
        Network {
            layers,
            learning_rate,
            batch_size,
            cost: 0.0,
            print_progress: false
        }
    }

    pub fn print_progress(&mut self, value: bool) {
        self.print_progress = value
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

    pub fn train(&mut self, mut data: Vec<[Vec<f64>; 2]>, epochs: usize) {
        for i in 0..epochs { // Epochs
            if i % 1000 == 0 && self.print_progress {
                println!("Progress: {}%", 100.0 * (i as f32 / epochs as f32));
            }
            let samples = data.len();
            let batches = f64::ceil(samples as f64 / self.batch_size as f64);

            for j in 0..batches as usize { // Batches
                let batch_start = j * self.batch_size;
                let batch_end = (batch_start + self.batch_size).min(samples);
                let samples_per_batch = batch_end - batch_start;
                let mut batch_targets: Vec<Vec<f64>> = vec![];
                let mut batch_outputs: Vec<Vec<f64>> = vec![];

                for k in batch_start..batch_end {
                    batch_outputs.push(self.forward(data[k][0].clone()));
                    batch_targets.push(data[k][1].clone());
                }

                let mut loss_gradient: Vec<f64> = vec![0.0; batch_targets[0].len()];

                for k in 0..batch_targets.len() { // Each Sample
                    self.cost += Self::get_cost(&batch_targets[k], &batch_outputs[k]);
                    for l in 0..batch_targets[k].len() { // Each Output Node
                        loss_gradient[l] += (2.0 * (batch_outputs[k][l] - batch_targets[k][l])) * (1.0 / samples_per_batch as f64);
                    }
                }
                
                self.backward(loss_gradient.clone());

                Self::shuffle_vector(&mut data);
            }
            self.cost /= data.len() as f64;
        }
        if self.print_progress {
            println!("Training Complete");
        }
    }

    pub fn reset(&mut self) {
        for i in 0..self.layers.len() {
            self.layers[i].reset();
        }
    }

    pub fn get_cost(targets: &Vec<f64>, outputs: &Vec<f64>) -> f64 {
        let mut cost = 0.0;
        for i in 0..targets.len() {
            cost += (outputs[i] -  targets[i]).powf(2.0);
        }
        cost
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

    pub fn get_weights(&self) -> Vec<Vec<Vec<f64>>>{
        let mut weights = vec![];
        for i in 0..self.layers.len() {
            weights.push(self.layers[i].get_weights());
        }
        weights
    }

    pub fn get_biases(&self) -> Vec<Vec<f64>> {
        let mut biases = vec![];
        for i in 0..self.layers.len() {
            biases.push(self.layers[i].get_biases());
        }
        biases
    }

    pub fn get_nodes(&self) -> Vec<usize>{
        let mut nodes = vec![];
        nodes.push(self.layers[0].get_input_nodes());
        for i in 0..self.layers.len() {
            nodes.push(self.layers[i].get_nodes());
        }
        nodes
    }

    pub fn shuffle_vector(vec: &mut Vec<[Vec<f64>; 2]>) {
        let mut rng = rand::thread_rng();
        vec.shuffle(&mut rng);
    }

    pub fn save_model(&self, name: &str) {
        let model: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) = (self.get_weights(), self.get_biases());
        let serialized = serde_json::to_string(&model).unwrap();
        let mut json = File::create(format!("{}.json", name)).unwrap();
        json.write_all(serialized.as_bytes()).expect("Error writing bytes to JSON");
    }

    pub fn load_model(&mut self, name: &str) {
        let mut str = String::new();
        let _ = File::open(format!("{}.json", name)).expect("Error opening file").read_to_string(&mut str);
        let model: (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) = serde_json::from_str(&str).unwrap();
        for i in 0..self.layers.len() {
            self.layers[i].set_params(model.0[i].clone(), model.1[i].clone());
        }
    }
}