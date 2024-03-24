use serde_derive::{Deserialize, Serialize};


#[derive(Serialize, Deserialize, PartialEq)]
pub enum LossType {
    MSE,
    CEL
}

#[derive(Serialize, Deserialize)]
pub struct LossFunction {
    pub loss_type: LossType
}


impl LossFunction {
    pub fn new(loss_type: LossType) -> Self {
        LossFunction {
            loss_type
        }
    }

    pub fn function(&self, outputs: &Vec<f64>, targets: &Vec<f64>) -> f64 {
        match self.loss_type {
            LossType::MSE => 
                {   
                    let mut cost = 0.0;
                    for i in 0..targets.len() {
                        cost += (outputs[i] -  targets[i]).powf(2.0);
                    }
                    cost
                },
            LossType::CEL => 
            {
                let mut true_index = 0;
                for i in 0..targets.len() {
                    if targets[i] == 1.0 {
                        true_index = i;
                    }
                }
                -(outputs[true_index].ln())
            },
        }
    }

    pub fn derivative(&self, outputs: &Vec<f64>, targets: &Vec<f64>) -> Vec<f64> {
        match self.loss_type {
            LossType::MSE => 
                {   
                    let mut loss_gradient: Vec<f64> = vec![0.0; targets.len()];
                    for l in 0..targets.len() {
                        loss_gradient[l] += 2.0 * (outputs[l] - targets[l]);
                    }
                    loss_gradient
                },
            LossType::CEL => 
                {
                    let mut true_index = 0;
                    for i in 0..targets.len() {
                        if targets[i] == 1.0 {
                            true_index = i;
                        }
                    }
                    let mut gradients = vec![outputs[true_index]; targets.len()];
                    gradients[true_index] = outputs[true_index] - 1.0;
                    gradients
                },
        }
    }
}