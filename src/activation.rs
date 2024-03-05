pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    SoftMax
}

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
                1.0 / (1.0 + (-x.exp())),
            ActivationFunction::ReLU => 
                if x < 0.0 {0.0} else {x},
            ActivationFunction::SoftMax => todo!(),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self.function {
            ActivationFunction::Sigmoid => 
                x * (1.0 - x),
            ActivationFunction::ReLU => 
                if x < 0.0 {0.0} else {1.0},
            ActivationFunction::SoftMax => todo!(),
        }
    }
}