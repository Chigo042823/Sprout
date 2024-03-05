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

    pub fn backward(&mut self, inputs: Vec<f64>, outputs: Vec<f64>, targets: Vec<f64>) {
        if outputs.len() != targets.len() {
            println!("Ouputs: {} and targets: {} are not compatible", outputs.len(), targets.len());
            panic!();
        }
        let mut loss_gradient = Vec::with_capacity(outputs.len());
        for i in 0..outputs.len() {
            loss_gradient.push(2.0 * (outputs[i] - targets[i]));
        }
        // let sum: f64 = loss_gradient.iter().sum();
        // println!("Loss: {}", sum);

        let mut next_loss_gradient = loss_gradient.clone();

        for i in (1..self.layers.len()).rev() {
            let input = self.layers[i-1].get_outputs();
            let layer = self.layers[i].as_mut();
            next_loss_gradient = layer.backward(input, next_loss_gradient, self.learning_rate);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: usize) {
        for i in 0..epochs {
            // println!("Epoch: {}", i + 1);
            for j in 0..inputs.len() {
                let outputs = self.forward(inputs[j].clone());
                self.backward(inputs[j].clone(), outputs, targets[j].clone());
            }
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
}