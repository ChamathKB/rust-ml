use tch::{nn, nn::Module, nn::OptimizerConfig, nn::Linear, nn::Func, nn::Optimizer, Tensor};

fn main() {
    // Define the dimensions of our input and output layers
    let input_size = 784;
    let hidden_size = 128;
    let output_size = 10;

    // Define the neural network
    let net  = nn::seq()
            .add(nn::Linear(input_size, hidden_size, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::Linear(hidden_size, output_size, Default::default()));

    // Load the MNIST dataset
    let (train_images, train_labels, test_images, test_labels) = tch::vision::mnist::load();

    // create an optimizer
    let mut optimizer = nn::Adam::default().build(&net, 1e-3).unwrap();

    // Train the neural network
    for epoch in 1..=10 {
        let loss = net
            .forward(&train_images)
            .cross_entropy_for_logits(&train_labels)
            .mean();
        
        optimizer.backward_step(&loss);

        println!("Epoch: {}, Loss: {:?}", epoch, loss);
    }

    // Test the neural network
    let test_accuracy = net
        .forward(&test_images)
        .softmax(-1, tch::Kind::Float)
        .accuracy_for_logits(&test_labels);

    println!("Test Accuracy: {:?}", test_accuracy);
}
