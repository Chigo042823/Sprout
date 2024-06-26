use std::vec;
use serde_derive::*;

use crate::
    {activation::
        {Activation, ActivationFunction}, 
        conv_params::
            {ConvParams, PaddingType}, 
        dense_params::
            DenseParams
    };

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LayerType {
    Dense,
    Convolutional,
    Pooling
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub activation: Activation,
    pub layer_type: LayerType,
    pub conv_params: Option<ConvParams>,
    pub dense_params: Option<DenseParams>,
}

impl Layer {
    pub fn dense(nodes: [usize; 2], activation_fn: ActivationFunction) -> Self {
        let dense_params = Some(DenseParams::new(nodes[0], nodes[1]));

        let mut layer = Layer {
            dense_params,
            activation: Activation::new(activation_fn),
            layer_type: LayerType::Dense,
            conv_params: None
        };
        layer.dense_params.as_mut().unwrap().init(layer.activation.function.clone());
        layer
    }

    pub fn conv(kernel: usize, padding_type: PaddingType, stride: usize, activation_fn: ActivationFunction) -> Self {
        let conv_params = Some(ConvParams::new(kernel, padding_type, stride));
        let layer = Layer {
            dense_params: None,
            activation: Activation::new(activation_fn),
            layer_type: LayerType::Convolutional,
            conv_params
        };
        layer
    }

    pub fn pool(kernel: usize, stride: usize) -> Self {
        let conv_params = Some(ConvParams::new(kernel, PaddingType::Valid, stride));
        let layer = Layer {
            dense_params: None,
            activation: Activation::new(ActivationFunction::ReLU),
            layer_type: LayerType::Pooling,
            conv_params
        };
        layer
    }

    pub fn conv_forward(&mut self, inputs: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        let params = self.conv_params.as_mut().unwrap();
        params.inputs = inputs;
        if params.padding_type == PaddingType::Valid {
            params.data = params.inputs.clone();
        }
        params.add_padding(self.activation.function.clone());
        let output_dims = params.get_output_dims();

        let mut weighted_inputs = vec![vec![vec![0.0; output_dims[0]]; output_dims[1]]; params.data.len()];
        let img = &params.data;

        let mut activation = vec![vec![vec![0.0; output_dims[0]]; output_dims[1]]; params.data.len()];

        for i in 0..img.len() { //each channel
            for j in (0..weighted_inputs[i].len()) { //each img row
                if j + params.kernel > params.data[i].len() {
                    break;
                }
                for k in (0..weighted_inputs[i][j].len()) { //each img column
                    if k + params.kernel > params.data[i][0].len() {
                        break;
                    }
                    for kern_row in 0..params.kernel { //Kernel rows
                        for kern_col in 0..params.kernel { //Kernel Columns
                            weighted_inputs[i][j][k] += (img[i][j * params.stride + kern_row][k * params.stride + kern_col] * params.weights[i][kern_row][kern_col]);
                            weighted_inputs[i][j][k] += params.bias;
                        }
                    }
                }
            }
            for j in 0..weighted_inputs[i].len() { 
                activation[i][j] = self.activation.function(weighted_inputs[i][j].clone());
            }
        }
        params.outputs = activation.clone();
        activation
    }

    pub fn pool_forward(&mut self, inputs: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        let params = self.conv_params.as_mut().unwrap();
        params.inputs = inputs;
        params.add_padding(self.activation.function.clone());

        let output_dims = params.get_output_dims();
        let img = params.inputs.clone();

        let mut output = vec![vec![vec![0.0; output_dims[0]]; output_dims[1]]; params.data.len()];

        for i in 0..img.len() { //each channel
            for j in (0..output[i].len()) { //each output img row
                if (j) * params.kernel >= img[i].len() {
                    break;
                }
                for k in (0..output[i][j].len()) { //each output img column
                    if (k) * params.kernel >= img[i][0].len() {
                        break;
                    }
                    let mut max = f64::NEG_INFINITY;
                    for kern_row in 0..params.kernel { //Kernel rows
                        let row_i = j * params.stride + kern_row;
                        if row_i >= img[i].len() {
                            break;
                        }
                        for kern_col in 0..params.kernel { //Kernel Columns
                            let col_i = k * params.stride + kern_col;
                            if col_i >= img[i][0].len() {
                                break;
                            }
                            let val = img[i][row_i][col_i];
                            if val > max {
                                max = val;
                            }
                        }
                    }
                    output[i][j][k] = max;
                }
            }
        }
        params.outputs = output.clone();
        output
    }

    pub fn dense_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        self.dense_params.as_mut().unwrap().inputs = inputs.clone();
        let mut weighted_inputs = self.dense_params.as_mut().unwrap().biases.clone();
        for i in 0..self.dense_params.as_mut().unwrap().nodes_out {
            for j in 0..self.dense_params.as_mut().unwrap().nodes_in {
                weighted_inputs[i] += inputs[j] * self.dense_params.as_mut().unwrap().weights[j][i];
            }
        }

        let activation = self.activation.function(weighted_inputs);
        self.dense_params.as_mut().unwrap().outputs = activation.clone();
        activation
    }

    pub fn conv_backward(&mut self, errors: Vec<Vec<Vec<f64>>>, learning_rate: f64) -> Vec<Vec<Vec<f64>>> {
        let params = self.conv_params.as_mut().unwrap();
        let channels = params.data.len();
        let mut delta_output = errors;
        let mut weight_gradients = vec![vec![vec![0.0; params.kernel]; params.kernel]; channels];
        let img = params.data.clone();
        let mut avg_bias_gradient = 0.0;
        let mut next_delta = vec![]; //3x3
        let kernel = params.kernel;

        for i in 0..channels { //each channel
            for j in 0..delta_output[i].len() {
                if self.activation.function == ActivationFunction::SoftMax {
                    break;
                }
                let activation_derivatives = self.activation.derivative(params.outputs[i][j].clone());
                for k in 0..delta_output[i][j].len() {
                    delta_output[i][j][k] *= activation_derivatives[k];
                }
            }

            for j in (0..delta_output[i].len()) { //each img row
                if j + kernel == img[i].len() {
                    break;
                }
                for k in (0..delta_output[i][j].len()) { //each img column
                    if k + kernel == img[i][j].len() {
                        break;
                    }
                    for kern_row in 0..kernel { //Kernel rows
                        for kern_col in 0..kernel { //Kernel Columns
                            weight_gradients[i][kern_row][kern_col] += (img[i][j * params.stride + kern_row][k * params.stride + kern_col] * delta_output[i][j][k]);
                        }
                    }
                }
            }
            for j in 0..params.weights[i].len() {
                for k in 0..params.weights[i][j].len() {
                    params.weights[i][j][k] -= learning_rate *  weight_gradients[i][j][k];
                }
            }
            // Update bias using the average of the gradients
            for j in 0..delta_output[i].len() {
                for k in 0..delta_output[i][j].len() {
                    avg_bias_gradient += delta_output[i][j][k];
                }
            }
            avg_bias_gradient /= (delta_output[i].len() * delta_output[i][0].len()) as f64;

            let padded_gradients = Self::add_padding_matrix(kernel - 1, &delta_output[i]);

            let rotated_kernel = Self::rotate180(&params.weights[i]);
            let mut gradient_channel = vec![];
            for j in (0..padded_gradients.len()) { //each img row
                if j + params.kernel > padded_gradients.len() {
                    break;
                }
                let mut gradient_row = vec![];
                for k in (0..padded_gradients[j].len()) { //each img column
                    if k + params.kernel > padded_gradients[0].len() {
                        break;
                    }
                    let mut sum = 0.0;
                    for kern_row in 0..params.kernel { //Kernel rows
                        for kern_col in 0..params.kernel { //Kernel Columns
                            sum += (rotated_kernel[kern_row][kern_col] * padded_gradients[j * params.stride + kern_row][k * params.stride + kern_col]);
                        }
                    }
                    gradient_row.push(sum);
                }
                gradient_channel.push(gradient_row);
            }
            next_delta.push(gradient_channel);
        }
        params.bias -= learning_rate * avg_bias_gradient;

        next_delta
    }

    pub fn pool_backward(&mut self, errors: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        let params = self.conv_params.as_mut().unwrap();
        let channels = params.data.len();
        let delta_output = errors;
        let img = params.data.clone();
        let kernel = params.kernel;
        let inputs = params.inputs.clone();
        let mut indecies = vec![[0, 0]; delta_output.len() * delta_output[0].len() + delta_output[0][0].len()];

        let mut next_delta = vec![vec![vec![0.0; inputs[0][0].len()]; inputs[0].len()]; inputs.len()]; //4x4

        for i in 0..channels { //each channel
            for j in (0..delta_output[i].len()) { //each img row
                if j * params.stride >= img[i].len() {
                    break;
                }
                for k in (0..delta_output[i][j].len()) { //each img column
                    if k * params.stride >= img[i][j].len() {
                        break;
                    }
                    let mut max = f64::NEG_INFINITY;
                    let mut max_indx = [j * params.stride, k * params.stride];
                    for kern_row in 0..kernel { //Kernel rows
                        let row_i = j * params.stride + kern_row;
                        if row_i >= img[i].len() {
                            break;
                        }
                        for kern_col in 0..kernel { //Kernel Columns
                            let col_i = k * params.stride + kern_col;
                            if col_i >= img[i][0].len() {
                                break;
                            }
                            let val = img[i][row_i][col_i];
                            if val > max {
                                max_indx = [row_i, col_i]; //each kernel
                                max = val;
                            }
                        }
                        // indecies.push(max_indx); //each channel
                    }
                    next_delta[i][max_indx[0]][max_indx[1]] = delta_output[i][j][k];
                }
            }
        }
        next_delta
    }

    pub fn rotate180(mat: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut new_mat = mat.clone();
        for row in &mut new_mat {
            row.reverse();
        }
        new_mat.reverse();
        new_mat
    }

    pub fn add_padding_matrix(padding: usize, matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let height = matrix.len();
        let width = matrix[0].len();
        let padded_height = height + 2 * padding;
        let padded_width = width + 2 * padding;
        
        let mut padded_image = vec![vec![0.0; padded_width]; padded_height];

        for i in 0..height {
            for j in 0..width {
                padded_image[i + padding][j + padding] = matrix[i][j];
            }
        }
        padded_image
    }

    pub fn dense_backward(&mut self, errors: Vec<f64>, learning_rate: f64) -> Vec<f64> {
        let mut delta_output = errors.clone();

        let activation_gradients = self.activation.derivative(self.dense_params.as_mut().unwrap().outputs.clone());
        for i in 0..delta_output.len() {
            if self.activation.function == ActivationFunction::SoftMax {
                break;
            }
            delta_output[i] *= activation_gradients[i];
        }

        for i in 0..self.dense_params.as_mut().unwrap().weights.len() {
            for j in 0..self.dense_params.as_mut().unwrap().weights[i].len() {
                self.dense_params.as_mut().unwrap().weights[i][j] -= learning_rate * (self.dense_params.as_mut().unwrap().inputs[i] * delta_output[j]);
            }
        }

        for i in 0..self.dense_params.as_mut().unwrap().biases.len() {
            self.dense_params.as_mut().unwrap().biases[i] -= learning_rate * delta_output[i];
        }

        let mut next_delta = vec![0.0; self.dense_params.as_mut().unwrap().nodes_in];
        for i in 0..self.dense_params.as_mut().unwrap().weights.len() {
            for j in 0..self.dense_params.as_mut().unwrap().weights[i].len() {
                next_delta[i] += (self.dense_params.as_mut().unwrap().weights[i][j] * delta_output[j] * self.dense_params.as_mut().unwrap().inputs[i]);
            }   
        }

        next_delta
    }

    pub fn get_dense_weights(&self) -> Vec<Vec<f64>> {
        return self.dense_params.as_ref().unwrap().weights.clone();
    }

    pub fn get_dense_biases(&self) -> Vec<f64> {
        return self.dense_params.as_ref().unwrap().biases.clone();
    }

    pub fn get_conv_weights(&self) -> Vec<Vec<Vec<f64>>> {
        return self.conv_params.as_ref().unwrap().weights.clone();
    }

    pub fn get_conv_biases(&self) -> f64 {
        return self.conv_params.as_ref().unwrap().bias.clone();
    }

    pub fn get_nodes(&self) -> usize {
        if self.layer_type == LayerType::Dense {
            return self.dense_params.as_ref().unwrap().nodes_out.clone();
        }
        0
    }

    pub fn get_input_nodes(&self) -> usize {
        if self.layer_type == LayerType::Dense {
            return self.dense_params.as_ref().unwrap().nodes_in.clone();
        }
        0
    }

    pub fn get_dense_outputs(&self) -> Vec<f64> {
        self.dense_params.as_ref().unwrap().outputs.clone()
    }

    pub fn get_conv_outputs(&self) -> Vec<Vec<Vec<f64>>> {
        self.conv_params.as_ref().unwrap().outputs.clone()
    }

    pub fn get_layer_type(&self) -> LayerType {
        self.layer_type.clone()
    }

    pub fn reset(&mut self) {
        if self.layer_type == LayerType::Dense {
            self.dense_params.as_mut().unwrap().inputs = vec![];
            self.dense_params.as_mut().unwrap().outputs = vec![];
            self.dense_params.as_mut().unwrap().init(self.activation.function.clone());
        } else {
            self.conv_params.as_mut().unwrap().inputs = vec![];
            self.conv_params.as_mut().unwrap().outputs = vec![];
            self.conv_params.as_mut().unwrap().init(self.activation.function.clone());
        }
    }

    pub fn set_params(&mut self, weights: Vec<Vec<f64>>, biases: Vec<f64>) {
        self.dense_params.as_mut().unwrap().weights = weights;
        self.dense_params.as_mut().unwrap().biases = biases;
    }
}