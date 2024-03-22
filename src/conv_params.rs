use serde_derive::{Serialize, Deserialize};
use rand::prelude::*;


#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PaddingType {
    Same,
    Valid,
    Full
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvParams {
    pub kernel: usize,
    pub padding_type: PaddingType,
    pub padding: usize,
    pub stride: usize,
    pub data: Vec<Vec<Vec<f64>>>, //channel > img matrix
    pub weights: Vec<Vec<Vec<f64>>>, //1 kernel > depth of kernel -> kernel weights
    pub bias: f64, //bias for each kernel not including depth
    pub outputs: Vec<Vec<Vec<f64>>>, //channel > mat
    pub inputs: Vec<Vec<Vec<f64>>>, //channel > mat
}

impl ConvParams {
    pub fn new(kernel: usize, padding_type: PaddingType, stride: usize) -> Self {
        
        let mut cp = ConvParams {
            kernel,
            padding_type,
            padding: 0,
            stride,
            data: vec![],
            weights: vec![],
            bias: 0.0,
            outputs: vec![],
            inputs: vec![],
        };
        cp
    }

    pub fn init(&mut self) {
        for i in 0..self.data.len() {
            self.weights.push(vec![vec![0.0; self.kernel]; self.kernel]);
        }

        self.bias = thread_rng().gen_range(-0.2..0.2);

        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                for k in 0..self.weights[i][j].len() {
                    self.weights[i][j][k] = thread_rng().gen_range(-0.2..0.2);
                }
            }
        }
    }

    pub fn add_padding(&mut self) {
        if self.padding_type == PaddingType::Valid {
            self.data = self.inputs.clone();
            self.init();
            return;
        }
        let mut padded_image = vec![];
        let mut padding = (self.kernel - 1) / 2;

        if self.padding_type == PaddingType::Full {
            padding = self.kernel - 1;
        }

        self.padding = padding;
        let height = self.inputs[0].len();
        let width = self.inputs[0][0].len();
        let padded_height = height + 2 * self.padding;
        let padded_width = width + 2 * self.padding;
        
        for i in 0..self.inputs.len() {
            padded_image.push(vec![vec![0.0; padded_width]; padded_height]);
        }

        for i in 0..padded_image.len() {
            for j in 0..height {
                for k in 0..width {
                    padded_image[i][j + self.padding][k + self.padding] = self.inputs[i][j][k];
                }
            }
        }

        self.data = padded_image;
        self.init();
    }

    pub fn get_output_dims(&self) -> [usize; 2] {
        let height = self.data[0].len();
        let width = self.data[0][0].len();

        let out_width = width - self.kernel + 1;
        let out_height = height - self.kernel + 1;

        [out_width, out_height]
    }
    
    pub fn print_kernels(&self) {
        println!("--------------------------\nKernel Dimensions: {} x {}", self.kernel, self.kernel);
        println!("Weights: \n{:#?}", self.weights);
        println!("Biases: \n{:#?}\n-------------------------------", self.bias);
    }
}