use crate::{activation::{Activation, ActivationFunction}, convolution_params::PaddingType, layer::Layer};

pub struct LayerBuilder {
    kernels: Vec<usize>,
    paddings: Vec<PaddingType>,
    strides: Vec<usize>,
    activations: Vec<ActivationFunction>,
    cn_layers: usize,
    img: [usize; 2],
    dense_layers: Vec<usize>,
}

impl LayerBuilder {
    pub fn new() -> Self {
        LayerBuilder {
            kernels: vec![],
            paddings: vec![],
            strides: vec![],
            activations: vec![],
            cn_layers: 0,
            img: [0; 2],
            dense_layers: vec![],
        }
    }

    pub fn set_kernels(&mut self, kernels: Vec<usize>) {
        self.kernels = kernels;
    }

    pub fn set_paddings(&mut self, paddings: Vec<PaddingType>) {
        self.paddings = paddings;
    }

    pub fn set_strides(&mut self, strides: Vec<usize>) {
        self.strides = strides;
    }

    pub fn set_activations(&mut self, activations: Vec<ActivationFunction>) {
        self.activations = activations;
    }

    pub fn set_cn_layers(&mut self, cn_layers: usize) {
        self.cn_layers = cn_layers;
    }

    pub fn set_img(&mut self, img: [usize; 2]) {
        self.img = img;
    }

    pub fn set_dense_layers(&mut self, dense_layers: Vec<usize>) {
        self.dense_layers = dense_layers;
    }

    pub fn cnn(&self) -> Vec<Layer> {
        let width = self.img[0];
        let width = self.img[1];
        let mut layers = vec![
            Layer::conv(self.kernels[0].clone(), self.paddings[0].clone(), self.strides[0].clone(), self.activations[0].clone())
        ];
        let layer_count = self.cn_layers + self.dense_layers.len();
        let mut switch = false;
        for i in 1..layer_count {
            if i == self.cn_layers {
                switch = true;
            }
            if !switch {
                layers.push(
                    Layer::conv(self.kernels[i].clone(), self.paddings[i].clone(), self.strides[i].clone(), self.activations[i].clone())
                );
            } else {
                let nodes = [self.dense_layers[i - 1], self.dense_layers[i]];
                layers.push(
                    Layer::dense(nodes, self.activations[i].clone())
                );
            }
        }

        vec![]
    }
}