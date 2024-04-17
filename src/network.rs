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
    pub loss_function: LossFunction,
    pub grad_threshold: f64,
}

impl Network {
    pub fn new(layers: Vec<Layer>, learning_rate: f64, batch_size: usize, loss_type: LossType) -> Self {
        let mut network_type = NetworkType::FCN;
        for i in 0..layers.len() {
            if layers[i].layer_type == LayerType::Convolutional || layers[i].layer_type == LayerType::Pooling {
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
            loss_function: LossFunction::new(loss_type),
            grad_threshold: 0.2,
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
                LayerType::Dense => 
                    {
                        if i != 0 {
                            if self.layers[i - 1].layer_type != LayerType::Dense {
                                dense_current = Self::flatten(&conv_current);
                            }
                            dense_current = self.layers[i].dense_forward(dense_current);
                        } 
                    },
                LayerType::Convolutional => 
                    {
                        conv_current = self.layers[i].conv_forward(conv_current);
                    },
                LayerType::Pooling => 
                    {
                        conv_current = self.layers[i].pool_forward(conv_current);
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
                LayerType::Dense => 
                    {
                        let layer = &mut self.layers[i];
                        delta_output = layer.dense_backward(delta_output.clone(), self.learning_rate);  
                        if i - 1 >= 0 as usize {
                            if self.layers[i - 1].layer_type != LayerType::Dense {
                                let params = self.layers[i - 1].conv_params.as_ref().unwrap();
                                conv_delta = Self::reshape(delta_output.clone(), 
                                    params.outputs.len(), 
                                    params.outputs[0].len(), 
                                    params.outputs[0][0].len()
                                );
                            }
                        } 
                    },
                LayerType::Convolutional => 
                    {
                        let layer = &mut self.layers[i];
                        conv_delta = layer.conv_backward(conv_delta.clone(), self.learning_rate);
                    },
                LayerType::Pooling => 
                    {
                        let layer = &mut self.layers[i];
                        conv_delta = layer.pool_backward(conv_delta.clone());
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

            self.cost = 0.0; // Reset cost
            
            Self::shuffle_tensor(&mut data);
            
            let batches = (samples / self.batch_size as f64).ceil() as usize;
    
            for batch in 0..batches { //each batch
                let mut loss_gradient: Vec<f64> = vec![0.0; data[0].1.len()]; //output nodes
                let mut sample_max = self.batch_size;
                let current_sample = batch * self.batch_size;
                if  current_sample + self.batch_size >= samples as usize && self.batch_size != 1 {
                    sample_max = samples as usize - current_sample;
                }
                for s in 0..sample_max { //each sample
                    let sample = data[current_sample + s].clone();
                    let output = self.conv_forward(sample.0.clone());
                    let target = &sample.1;
                    let mut true_index = 0;
                    for i in 0..target.len() {
                        if target[i] == 1.0 {
                            true_index = i;
                            break;
                        }
                    }
                    let mut cost = self.loss_function.function(&output, &target, true_index);
                    if cost.is_nan() || cost.is_infinite() {
                        cost = 0.0;
                    }

                    self.cost += cost;
        
                    let sample_loss = self.loss_function.derivative(&output, target, true_index);

                    for i in 0..loss_gradient.len() { //Sum Gradients
                        loss_gradient[i] += sample_loss[i];
                    }

                    let l2_norm = loss_gradient.iter().map(|x| x.powf(2.0)).sum::<f64>().sqrt();

                    if l2_norm > self.grad_threshold {
                        let scale = self.grad_threshold / l2_norm;
                        for i in 0..loss_gradient.len() {
                            loss_gradient[i] *= scale;
                        }
                    }
                    
                }
                // for i in 0..loss_gradient.len() {
                //     loss_gradient[i] /= self.batch_size as f64;
                // }

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

            let batches = (samples / self.batch_size as f64).ceil() as usize;
    
            for batch in 0..batches { //each batch
                let mut loss_gradient: Vec<f64> = vec![0.0; data[0][1].len()]; //output nodes
                let mut sample_max = self.batch_size;
                let current_sample = batch * self.batch_size;
                if  current_sample + self.batch_size >= samples as usize  && self.batch_size != 1 {
                    sample_max = samples as usize - current_sample;
                }
                for s in 0..sample_max { //each sample
                    let sample = data[batch * self.batch_size + s].clone();
                    let output = self.dense_forward(sample[0].clone());
                    let target = &sample[1];
                    let mut true_index = 0;
                    for i in 0..target.len() {
                        if target[i] == 1.0 {
                            true_index = i;
                            break;
                        }
                    }
                    self.cost += self.loss_function.function(&output, &target, true_index);
        
                    let sample_loss = self.loss_function.derivative(&output, target, true_index);

                    for i in 0..loss_gradient.len() { //Sum Gradients
                        loss_gradient[i] += sample_loss[i];
                    }

                    let l2_norm = loss_gradient.iter().map(|x| x.powf(2.0)).sum::<f64>().sqrt();

                    if l2_norm > self.grad_threshold {
                        let scale = self.grad_threshold / l2_norm;
                        for i in 0..loss_gradient.len() {
                            loss_gradient[i] *= scale;
                        }
                    }
                }
                let _ = loss_gradient.iter().map(|x| x / self.batch_size as f64).collect::<Vec<f64>>();
                
                self.dense_backward(loss_gradient);
        
                self.cost /= self.batch_size as f64; // Compute average cost per sample
            }
        }
    
        if self.print_progress {
            println!("Training Complete");
        }
    }

    pub fn reset(&mut self) {
        self.cost = 0.0;
        for i in 0..self.layers.len() {
            self.layers[i].reset();
        }
    }

    pub fn print_weights(&self) {
        for i in 0..self.layers.len() {
            println!("{:#?}", self.layers[i].get_dense_weights());
        }
    }

    pub fn print_biases(&self) {
        for i in 0..self.layers.len() {
            println!("{:#?}", self.layers[i].get_dense_biases());
        }
    }

    pub fn get_weights(&self) -> (Vec<Vec<Vec<Vec<f64>>>>, Vec<Vec<Vec<f64>>>) {
        let mut dense_weights = vec![];
        let mut conv_weights = vec![];
        for i in 0..self.layers.len() {
            if self.layers[i].layer_type == LayerType::Dense {
                dense_weights.push(self.layers[i].get_dense_weights());
            } else {
                conv_weights.push(self.layers[i].get_conv_weights());
            }
        }
        (conv_weights, dense_weights)
    }

    pub fn get_biases(&self) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut dense_biases = vec![];
        let mut conv_biases = vec![];
        for i in 0..self.layers.len() {
            if self.layers[i].layer_type == LayerType::Dense {
                dense_biases.push(self.layers[i].get_dense_biases());
            } else {
                conv_biases.push(self.layers[i].get_conv_biases());
            }
        }
        (conv_biases, dense_biases)
    }

    pub fn get_conv_outputs(&self) -> Vec<Vec<Vec<Vec<f64>>>> {
        let mut outputs = vec![];
        for i in 0..self.layers.len() {
            if self.layers[i].layer_type == LayerType::Dense {
                break;
            }
            let output = self.layers[i].get_conv_outputs();
            outputs.push(output)
        }
        outputs
    }

    pub fn get_nodes(&self) -> Vec<usize>{
        let mut nodes = vec![];
        for i in 0..self.layers.len() {
            if self.layers[i].layer_type == LayerType::Convolutional {
                continue;
            }
            nodes.push(self.layers[i].get_input_nodes());
        }
        nodes.push(self.layers[self.layers.len() - 1].get_nodes());
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

    pub fn from_load(name: &str) -> Self {
        let mut str = String::new();
        let _ = File::open(format!("{}.json", name)).expect("Error opening file").read_to_string(&mut str);
        let model: Network = serde_json::from_str(&str).unwrap();
        model
    }
}