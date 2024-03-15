use std::time;
use image::*;

use ml_library::{activation::ActivationFunction::*, layer::{Layer, LayerType::Dense}, network::Network};

fn main() {
    let layers: Vec<Layer> = vec![
        Layer::new(2, 3, Dense, Sigmoid),
        Layer::new(3, 1, Dense, Sigmoid),
    ];
    let mut nn = Network::new(layers, 0.7, 2);

    let time = time::Instant::now();
    xor_mode(&mut nn, 10_000);
    // digit_model(nn, 1000);
    // nn.save_model("test1");

    let delta = time.elapsed();

    println!("Training Time Elapsed: {:?}", delta);
}

pub fn xor_mode(nn: &mut Network, epochs: usize) {
    let data: Vec<[Vec<f64>; 2]> = vec![
        [vec![1.0, 0.0], vec![0.0]],
        [vec![0.0, 0.0], vec![1.0]],
        [vec![1.0, 1.0], vec![1.0]],
        [vec![0.0, 1.0], vec![0.0]],
    ];  

    nn.train(data.clone(), epochs);
    // nn.load_model("test1");

    for i in 0..data.len() {
        println!("Input: {:?} // Output: {:?} // Target: {:?}",data[i][0].clone(), nn.forward(data[i][0].clone()), data[i][1].clone());
    }
}

pub fn digit_model(mut nn: Network, epochs: usize) {
    let mut data: Vec<[Vec<f64>; 2]> = vec![];

    let img = image::open("mnist_8.png").unwrap();

    for y in 0..img.dimensions().1 {
        for x in 0..img.dimensions().0 {
            let intensity;
            let pixel = img.get_pixel(x, y).0;
            intensity = (pixel[0] / 3) + (pixel[1] / 3) + (pixel[2] / 3);
            data.push([vec![x as f64, y as f64], vec![(intensity as f64 / 255 as f64)]]);
        }
    }

    nn.train(data.clone(), epochs);

    // for y in 0..img.dimensions().1 as usize {
    //     for x in 0..img.dimensions().0 as usize {
    //         if data[x + (y * img.dimensions().0 as usize)][1][0] * 255.0 > 10.0 {
    //             print!("#");
    //         } else {
    //             print!(" ");
    //         }
    //     }
    //     println!();
    // }


    // println!("-----------------------------");

    let mut new_image = RgbImage::new(img.dimensions().0, img.dimensions().1);

    for y in 0..img.dimensions().1 as usize {
        for x in 0..img.dimensions().0 as usize {
            let int = (nn.forward(vec![x as f64, y as f64])[0] * 255.0) as u8;
            let rgb = Rgb([int, int, int]);
            new_image.put_pixel(x as u32, y as u32, rgb);
            // if int > 10 {
            //     print!("#");
            // } else {
            //     print!(" ");
            // }
        }
        // println!();
    }

    let _  = new_image.save("Output.png").unwrap();
}
