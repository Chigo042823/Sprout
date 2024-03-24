use rand::seq::SliceRandom;
use serde_derive::{Serialize, Deserialize};

use crate::{layer::{Layer, LayerType}, loss_function::{LossFunction, LossType}};
use std::{fs::File, io::{Read, Write}};

#[derive(Serialize, Deserialize, PartialEq)]
pub enum NetworkType {
    FCN,
    CNN
}

#[derive(Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub cost: f64,
    pub print_progress: bool,
    pub network_type: NetworkType,
    pub loss_function: LossFunction
}

impl Network {
    pub fn new(layers: Vec<Layer>, learning_rate: f64, batch_size: usize, loss_type: LossType) -> Self {
        let mut network_type = NetworkType::FCN;
        for i in 0..layers.len() {
            if layers[i].layer_type == LayerType::Convolutional {
                network_type = NetworkType::CNN;
            }
        }
        Network {
            layers,
            learning_rate,
            batch_size,
            cost: 0.0,
            print_progress: false,
            network_type,
            loss_function: LossFunction::new(loss_type)
        }
    }

    pub fn print_progress(&mut self, value: bool) {
        self.print_progress = value
    }

    pub fn dense_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut current: Vec<f64> = inputs; 
        for i in 0..self.layers.len() {
            current = self.layers[i].dense_forward(current);
        }
        current
    }

    pub fn conv_forward(&mut self, inputs: Vec<Vec<Vec<f64>>>) -> Vec<f64> {
        let mut conv_current = inputs; 
        let mut dense_current = vec![]; 
        for i in 0..self.layers.len() {
            match self.layers[i].layer_type {
                crate::layer::LayerType::Dense => 
                    {
                        if i != 0 {
                            if self.layers[i - 1].layer_type == LayerType::Convolutional {
                                dense_current = Self::flatten(&conv_current);
                            }
                            dense_current = self.layers[i].dense_forward(dense_current);
                        } 
                    },
                crate::layer::LayerType::Convolutional => 
                    {
                        conv_current = self.layers[i].conv_forward(conv_current);
                    },
            }
        }
        dense_current
    }

    pub fn flatten(inputs: &Vec<Vec<Vec<f64>>>) -> Vec<f64> {
        let flat = inputs.iter()
            .flat_map(|row| row.iter())
            .flat_map(|col| col.iter())
            .cloned() // Cloned to avoid borrowing issues
            .collect();
        flat
    }

    pub fn reshape(input: Vec<f64>, channels: usize, rows: usize, cols: usize) -> Vec<Vec<Vec<f64>>> {
        let mut output = vec![vec![vec![0.0; cols]; rows]; channels];
        
        for i in 0..channels {
            let mut index = 0;
            for j in 0..rows {
                for k in 0..cols {
                    output[i][j][k] = input[index].clone();
                    index += 1;
                }
            }
        }
    
        output
    }

    pub fn conv_backward(&mut self, loss_gradient: Vec<f64>) {
        let mut delta_output = loss_gradient.clone();
        let mut conv_delta = vec![];

        for i in (0..self.layers.len()).rev() {
            match self.layers[i].layer_type {
                crate::layer::LayerType::Dense => 
                    {
                        let layer = &mut self.layers[i];
                        delta_output = layer.dense_backward(delta_output.clone(), self.learning_rate);  
                        if i - 1 >= 0 as usize {
                            if self.layers[i - 1].layer_type == LayerType::Convolutional {
                                let params = self.layers[i - 1].conv_params.as_ref().unwrap();
                                conv_delta = Self::reshape(delta_output.clone(), 
                                    params.outputs.len(), 
                                    params.outputs[0].len(), 
                                    params.outputs[0][0].len()
                                );
                                // println!("{:#?}", conv_delta);
                            }
                        } 
                    },
                crate::layer::LayerType::Convolutional => 
                    {
                        let layer = &mut self.layers[i];
                        layer.conv_backward(conv_delta.clone(), self.learning_rate);
                    },
            }
        }
    }

    pub fn dense_backward(&mut self, loss_gradient: Vec<f64>) {
        let mut delta_output = loss_gradient.clone();

        for i in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[i];
            delta_output = layer.dense_backward(delta_output.clone(), self.learning_rate);
        }
    }

    pub fn conv_train(&mut self, mut data: Vec<(Vec<Vec<Vec<f64>>>, Vec<f64>)>, epochs: usize) {
        let samples = data.len() as f64;
        for i in 0..epochs {
            if i % 1000 == 0 && self.print_progress {
                println!("Progress: {}%", 100.0 * (i as f64 / epochs as f64));
            }
            
            Self::shuffle_tensor(&mut data);
            self.cost = 0.0; // Reset cost for each epoch
    
            for sample in &data {
                let output = self.conv_forward(sample.0.clone());
                let target = &sample.1;
                self.cost += self.loss_function.function(&output, &target);
    
                let mut loss_gradient: Vec<f64> = self.loss_function.derivative(&output, target);
                
    
                self.conv_backward(loss_gradient);
            }
    
            self.cost /= samples; // Compute average cost per sample
        }
    
        if self.print_progress {
            println!("Training Complete");
        }
    }

    pub fn dense_train(&mut self, mut data: Vec<[Vec<f64>; 2]>, epochs: usize) {
        let samples = data.len() as f64;
    
        for i in 0..epochs {
            if i % 1000 == 0 && self.print_progress {
                println!("Progress: {}%", 100.0 * (i as f64 / epochs as f64));
            }
            
            Self::shuffle_vector(&mut data);
            self.cost = 0.0; // Reset cost for each epoch
    
            for sample in &data {
                let output = self.dense_forward(sample[0].clone());
                let target = &sample[1];
                self.cost += self.loss_function.function(&output, &target);
    
                let mut loss_gradient: Vec<f64> = vec![0.0; target.len()];
                for l in 0..target.len() {
                    loss_gradient[l] += 2.0 * (output[l] - target[l]);
                }
    
                self.dense_backward(loss_gradient);
            }
    
            self.cost /= samples; // Compute average cost per sample
        }
    
        if self.print_progress {
            println!("Training Complete");
        }
    }

    pub fn normalgd_train(&mut self, mut data: Vec<[Vec<f64>; 2]>, epochs: usize) {
        // let samples = data.len() as f64;
    
        // for i in 0..epochs {
        //     if i % 1000 == 0 && self.print_progress {
        //         println!("Progress: {}%", 100.0 * (i as f64 / epochs as f64));
        //     }
    
        //     self.cost = 0.0; // Reset cost for each epoch
        //     let mut loss_gradient: Vec<f64> = vec![0.0; data[0][1].len()];
    
        //     for sample in &data {
        //         let output = self.forward(sample[0].clone());
        //         let target = &sample[1];
        //         self.cost += Self::get_cost(target, &output);
    
        //         for l in 0..target.len() {
        //             loss_gradient[l] += 2.0 * (output[l] - target[l]);
        //             loss_gradient[l] /= samples;
        //         }    
        //     }
        //     self.backward(loss_gradient);
    
        //     self.cost /= samples; // Compute average cost per sample
        // }
    
        // if self.print_progress {
        //     println!("Training Complete");
        // }
    }

    pub fn batch_train(&mut self, mut data: Vec<[Vec<f64>; 2]>, epochs: usize) {
        // for i in 0..epochs { // Epochs
        //     Self::shuffle_vector(&mut data);
        //     if i % 1000 == 0 && self.print_progress {
        //         println!("Progress: {}%", 100.0 * (i as f32 / epochs as f32));
        //     }
        //     let samples = data.len();
        //     let batches = f64::ceil(samples as f64 / self.batch_size as f64);

        //     for j in 0..batches as usize { // Batches
        //         let batch_start = j * self.batch_size;
        //         let batch_end = (batch_start + self.batch_size).min(samples);
        //         let samples_per_batch = batch_end - batch_start;
        //         let mut batch_targets: Vec<Vec<f64>> = vec![];
        //         let mut batch_outputs: Vec<Vec<f64>> = vec![];

        //         for k in batch_start..batch_end {
        //             batch_outputs.push(self.forward(data[k][0].clone()));
        //             batch_targets.push(data[k][1].clone());
        //         }

        //         let mut loss_gradient: Vec<f64> = vec![0.0; batch_targets[0].len()];

        //         for k in 0..batch_targets.len() { // Each Sample
        //             self.cost += Self::get_cost(&batch_targets[k], &batch_outputs[k]);
        //             for l in 0..batch_targets[k].len() { // Each Output Node
        //                 loss_gradient[l] += (2.0 * (batch_outputs[k][l] - batch_targets[k][l])) * (1.0 / samples_per_batch as f64);
        //             }
        //         }
        //         self.backward(loss_gradient);
        //     }
        //     self.cost /= data.len() as f64;
        // }
        // if self.print_progress {
        //     println!("Training Complete");
        // }
    }

    pub fn reset(&mut self) {
        for i in 0..self.layers.len() {
            self.layers[i].reset();
        }
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

    pub fn shuffle_tensor(vec: &mut Vec<(Vec<Vec<Vec<f64>>>, Vec<f64>)>) {
        let mut rng = rand::thread_rng();
        vec.shuffle(&mut rng);
    }

    pub fn save_model(&self, name: &str) {
        let model: &Network = self;
        let serialized = serde_json::to_string(&model).unwrap();
        let mut json = File::create(format!("{}.json", name)).unwrap();
        json.write_all(serialized.as_bytes()).expect("Error writing bytes to JSON");
    }

    pub fn load_model(&mut self, name: &str) {
        let mut str = String::new();
        let _ = File::open(format!("{}.json", name)).expect("Error opening file").read_to_string(&mut str);
        let mut model: Network = serde_json::from_str(&str).unwrap();
        std::mem::swap(self, &mut model);
    }
}