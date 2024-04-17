    use serde_derive::{Deserialize, Serialize};


#[derive(Serialize, Deserialize, PartialEq, Clone)]
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

    pub fn function(&self, outputs: &Vec<f64>, targets: &Vec<f64>, true_index: usize) -> f64 {
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
                -outputs[true_index].ln()
            },
        }
    }

    pub fn derivative(&self, outputs: &Vec<f64>, targets: &Vec<f64>, true_index: usize) -> Vec<f64> {
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
                    let mut gradients = outputs.clone();
                    gradients[true_index] -= 1.0;
                    gradients
                },
        }
    }
}