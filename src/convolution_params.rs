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
    pub data: Vec<Vec<f64>>,
    pub weights: Vec<Vec<f64>>,
    pub bias: f64,
    pub outputs: Vec<Vec<f64>>,
    pub inputs: Vec<Vec<f64>>
}

impl ConvParams {
    pub fn new(kernel: usize, padding_type: PaddingType, stride: usize) -> Self {
        let mut weights = vec![vec![0.0; kernel]; kernel];
        let bias = thread_rng().gen_range(-0.2..0.2);

        for i in 0..weights.len() {
            for j in 0..weights[i].len() {
                    weights[i][j] = thread_rng().gen_range(-0.2..0.2);
            }
        }
        
        ConvParams {
            kernel,
            padding_type,
            padding: 0,
            stride,
            data: vec![],
            weights,
            bias,
            outputs: vec![],
            inputs: vec![],
        }
    }

    pub fn add_padding(&mut self) {
        if self.padding_type == PaddingType::Valid {
            return;
        }
        let mut padding = (self.kernel - 1) / 2;
        if self.padding_type == PaddingType::Full {
            padding = self.kernel - 1;
        }
        self.padding = padding;
        let height = self.inputs.len();
        let width = self.inputs[0].len();
        let padded_height = height + 2 * self.padding;
        let padded_width = width + 2 * self.padding;
        
        let mut padded_image = vec![vec![0.0; padded_width]; padded_height];

        for j in 0..height {
            for k in 0..width {
                padded_image[j + self.padding][k + self.padding] = self.inputs[j][k];
            }
        }
        self.data = padded_image;
    }

    pub fn get_output_dims(&self) -> [usize; 2] {
        let height = self.data.len();
        let width = self.data[0].len();

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