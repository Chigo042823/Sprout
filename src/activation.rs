use serde_derive::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    TanH,
    SoftMax
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Activation {
    pub function: ActivationFunction,
}

impl Activation {
    pub fn new(function: ActivationFunction) -> Self {
        Activation {
            function
        }
    }
    pub fn function(&self, inp: Vec<f64>) -> Vec<f64> {
        let inputs = inp.clone();
        match self.function {
            ActivationFunction::Sigmoid => 
                {
                    let mut outputs = vec![0.0; inputs.len()];
                    for i in 0..outputs.len() {
                        outputs[i] = 1.0 / (1.0 + ((-inputs[i]).exp()));
                    }
                    outputs
                },
            ActivationFunction::ReLU => 
                {
                    let mut outputs = vec![0.0; inputs.len()];
                    for i in 0..outputs.len() {
                        let x = inputs[i];
                        outputs[i] = if x > 0.0 {
                            x
                        } else {
                            x * 0.01
                        };
                    }
                    outputs  
                },
            ActivationFunction::TanH =>
                {
                    let mut outputs = vec![0.0; inputs.len()];
                    for i in 0..outputs.len() {
                        let x = inputs[i];
                        outputs[i] = x.tanh()
                    }
                    outputs   
                },
            ActivationFunction::SoftMax => 
                {
                    let max_input = *inputs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                    let sum_exp: f64 = inputs.iter().map(|&x| (x - max_input).exp()).sum();

                    let mut outputs = vec![0.0; inputs.len()];
                    for i in 0..outputs.len() {
                        outputs[i] = ((inputs[i] - max_input).exp()) / sum_exp;
                    }
                    outputs 
                },
        }
    }

    pub fn derivative(&self, outs: Vec<f64>, true_index: usize) -> Vec<f64> {
        let outputs = outs.clone();
        match self.function {
            ActivationFunction::Sigmoid => 
                {
                    let mut gradients = vec![0.0; outputs.len()];
                    for i in 0..outputs.len() {
                        let x: f64 = outputs[i];
                        gradients[i] = x * (1.0 - x);
                    }
                    gradients
                },
            ActivationFunction::ReLU => 
                {
                    let mut gradients = vec![0.0; outputs.len()];
                    for i in 0..outputs.len() {
                        let x = outputs[i];
                        gradients[i] = if x < 0.0 {
                            0.01
                        } else {
                            1.0
                        };
                    }
                    gradients
                },
            ActivationFunction::TanH => 
            {
                let mut gradients = vec![0.0; outputs.len()];
                for i in 0..outputs.len() {
                    let x = outputs[i];
                    gradients[i] = 1.0 - (x*x);
                }
                gradients
            },
            ActivationFunction::SoftMax => 
                {
                    let true_val = outputs[true_index];
                    let mut gradients = vec![0.0; outputs.len()];
                    for i in 0..outputs.len() {
                        if i == true_index {
                            continue;
                        }
                        let x = outputs[i];
                        gradients[i] = -true_val * x;
                    }
                    gradients[true_index] = true_val * (1.0 - true_val);
                    gradients                
                },
        }
    }
}