This is a simple Machine Learning library in rust made with no pre-existing ML or linear algebra libraries.
You can easily build your network by defining layers within a vec and pasing it into the network struct:

    let layers = vec![
        Layer::conv(25, Valid, 1, ReLU),
        Layer::dense([16, 12], Sigmoid),
        Layer::dense([12, 10], SoftMax),
    ];
    let nn = Network::new(layers, 0.0002, 1, CEL);
As of now the only supported layers are conv and dense layers, pooling layers are next on the agenda.

will expound readme soon...
