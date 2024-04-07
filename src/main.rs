use std::time;
use image::*;

use ml_library::{conv_params::PaddingType::*, activation::ActivationFunction::*, loss_function::LossType::*, conv_params::PaddingType, layer::{Layer, LayerType::*}, network::Network};

fn main() {
    let time = time::Instant::now();

    conv_model();

    let delta = time.elapsed();

    println!("Training Time Elapsed: {:?}", delta);
}

pub fn conv_model() {

    let layers = vec![
        Layer::conv(3, Valid, 1, ReLU),
        Layer::pool(2, 2),
        Layer::dense([4, 3], Sigmoid),
    ];

    let data = vec![
        (vec![
            vec![
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ], 
        vec![0.0, 1.0, 0.0]
        ),
        (vec![
            vec![
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ], 
        vec![1.0, 0.0, 0.0,]
        ),
        (vec![
            vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ], 
        vec![0.0, 0.0, 1.0,]
        ),
    ];

    let mut nn = Network::new(layers, 0.01, 3, MSE);
    nn.conv_train(data.clone(), 10000);
    println!("Samples: {} || Channels: {} || Rows: {} || Cols: {}", data.len(), data[0].0.len(), data[0].0[0].len(), data[0].0[0][0].len());
    println!("Output 1: {:?}", nn.conv_forward(data[0].0.clone()));
    println!("Output 2: {:?}", nn.conv_forward(data[1].0.clone()));
    println!("Output 3: {:?}", nn.conv_forward(data[2].0.clone()));
}

pub fn xor_mode(epochs: usize) {

    let layers = vec![
        Layer::dense([2, 3], Sigmoid),
        Layer::dense([3, 1], Sigmoid),
    ];

    let mut nn = Network::new(layers, 0.5, 3, MSE);

    let data: Vec<[Vec<f64>; 2]> = vec![
        [vec![1.0, 0.0], vec![0.0]],
        [vec![0.0, 0.0], vec![1.0]],
        [vec![1.0, 1.0], vec![1.0]],
        [vec![0.0, 1.0], vec![0.0]],
    ];  

    nn.dense_train(data.clone(), epochs);
    // nn.load_model("test1");

    for i in 0..data.len() {
        println!("Input: {:?} // Output: {:?} // Target: {:?}",data[i][0].clone(), nn.dense_forward(data[i][0].clone()), data[i][1].clone());
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

    nn.dense_train(data.clone(), epochs);

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
            let int = (nn.dense_forward(vec![x as f64, y as f64])[0] * 255.0) as u8;
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
