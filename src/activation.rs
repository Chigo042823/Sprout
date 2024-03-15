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
    pub fn function(&self, x: f64) -> f64 {
        match self.function {
            ActivationFunction::Sigmoid => 
                1.0 / (1.0 + ((-x).exp())),
            ActivationFunction::ReLU => 
                if x < 0.0 {
                    x * 0.01
                } else {
                    x
                },
            ActivationFunction::TanH =>
                x.tanh(),
            ActivationFunction::SoftMax => todo!(),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self.function {
            ActivationFunction::Sigmoid => 
                x * (1.0 - x),
            ActivationFunction::ReLU => 
                if x < 0.0 {
                    0.01
                } else {
                    1.0
                },
            ActivationFunction::TanH => 
                1.0 - (x*x),
            ActivationFunction::SoftMax => todo!(),
        }
    }
}