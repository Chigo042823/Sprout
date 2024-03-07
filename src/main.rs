use std::time;
use image::*;

use ml_library::{activation::ActivationFunction::*, layer::{DenseLayer, Layer}, network::Network};

fn main() {
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(DenseLayer::new(2, 7, Sigmoid)),
        Box::new(DenseLayer::new(7, 7, Sigmoid)),
        Box::new(DenseLayer::new(7, 1, Sigmoid)),
    ];
    let mut nn = Network::new(layers, 0.8, 1);
    
    // let inputs: Vec<Vec<f64>> = vec![
    //     vec![1.0, 0.0],
    //     vec![0.0, 0.0],
    //     vec![1.0, 1.0],
    //     vec![0.0, 1.0],
    // ]; 
    // let targets: Vec<Vec<f64>> = vec![
    //     vec![1.0, 0.0],
    //     vec![0.0, 1.0],
    //     vec![0.0, 1.0],
    //     vec![1.0, 0.0],
    // ]; 

    // for i in 0..targets.len() {
    //     println!("Input: {:?} // Output: {:?} // Target: {:?}",inputs[i].clone(), nn.forward(inputs[i].clone()), targets[i].clone());
    // }

    let mut inputs = vec![];
    let mut targets = vec![];

    let img = image::open("mnist_8.png").unwrap();

    for y in 0..img.dimensions().1 {
        for x in 0..img.dimensions().0 {
            let intensity;
            let pixel = img.get_pixel(x, y).0;
            intensity = (pixel[0] / 3) + (pixel[1] / 3) + (pixel[2] / 3);
            inputs.push(vec![x as f64, y as f64]);
            targets.push(vec![(intensity as f64 / 255 as f64)]);
        }
    }

    let time = time::Instant::now();

    nn.train(inputs.clone(), targets.clone(), 600_000);

    let delta = time.elapsed();

    for y in 0..img.dimensions().1 as usize {
        for x in 0..img.dimensions().0 as usize {
            if targets[x + (y * img.dimensions().0 as usize)][0] * 255.0 > 10.0 {
                print!("#");
            } else {
                print!(" ");
            }
        }
        println!();
    }


    println!("-----------------------------");

    let mut new_image = RgbImage::new(img.dimensions().0, img.dimensions().1);

    for y in 0..img.dimensions().1 as usize {
        for x in 0..img.dimensions().0 as usize {
            let int = (nn.forward(vec![x as f64, y as f64])[0] * 255.0) as u8;
            let rgb = Rgb([int, int, int]);
            new_image.put_pixel(x as u32, y as u32, rgb);
            if int > 10 {
                print!("#");
            } else {
                print!(" ");
            }
        }
        println!();
    }

    let _  = new_image.save("Output.png").unwrap();

    println!("Training Time Elapsed: {:?}", delta);
}
