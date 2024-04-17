<p align="center">
    <img height="50%" width="50%" src="assets/Logo.png" alt="Sprout Logo">
</p>

<h1>About</h1>
Sprout is a <b>Simple Machine Learning library</b> in <b>Rust</b> made with no pre-existing ML or linear algebra libraries.
I made Sprout to get a better understanding of ML concepts.
<h1>Key Features</h1>
<ul>
    <li>Fully Connected Layers</li>
    <li>Convolution Layers</li>
    <li>Mini-Batch Gradient Descent</li>
    <li>Normalizations</li>
    <li>Model Saving/Loading to JSON</li>
</ul>
<h1>How To Use</h1>
Sprout uses a Vec of the included Layer struct which is passed into the Network struct as shown here:

    use Sprouts::{Layer::{Layer, LayerType}, network::Network, activation::ActivationFunction::*, loss_function::LossType::*}
    
    let layers = vec![
        Layer::dense([2, 3], Sigmoid),
        Layer::dense([3, 1], Sigmoid),
    ];
    
    // Network::new(layers, learning_rate, batch_size, loss_function);
    let nn = Network::new(layers, 0.2, 1, MSE);
    
    //Prints network's loss and epoch progress in the terminal
    nn.dense_train(true);

    //data: Vec<[Inputs, Outputs]>
    let data: Vec<[Vec<f64>; 2]> = vec![
        [vec![1.0, 0.0], vec![0.0]],
        [vec![0.0, 0.0], vec![1.0]],
        [vec![1.0, 1.0], vec![1.0]],
        [vec![0.0, 1.0], vec![0.0]],
    ];  

    //dense_train(data, epochs)
    nn.dense_train(data.clone(), 10000);

    for i in 0..data.len() {
        println!("Input: {:?} || Output: {:?} || Target: {:?}",data[i][0].clone(), nn.dense_forward(data[i][0].clone()), data[i][1].clone());
    }
    
As of now the only supported layers are conv and dense layers, pooling layers are next on the agenda.

will expound readme soon...

## License

This project is licensed under the [MIT License](LICENSE).
