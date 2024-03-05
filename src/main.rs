use std::time::{self, Duration};

use ml_library::{activation::ActivationFunction::*, layer::{DenseLayer, Layer}, network::Network};

fn main() {
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(DenseLayer::new(2, 3, Sigmoid)),
        Box::new(DenseLayer::new(3, 2, Sigmoid)),
        Box::new(DenseLayer::new(2, 2, Sigmoid)),
    ];
    let inputs = vec![
        vec![1.0, 0.0],
        vec![0.0, 0.0],
    ];
    let targets = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ];
    let mut nn = Network::new(layers, 0.001);
    let time = time::Instant::now();

    nn.train(inputs.clone(), targets, 100);
    println!("1 0 = {:?}", nn.forward(inputs[0].clone()));
    println!("0 0 = {:?}", nn.forward(inputs[1].clone()));
    // nn.print_biases();

    let delta = time.elapsed();
    println!("Time Elapsed: {:?}", delta);
}
