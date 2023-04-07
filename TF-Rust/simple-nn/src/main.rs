use tensorflow::{Graph, Session};
use tensorflow::ops::{Placeholder, MatMul, BiasAdd, Relu, Softmax};
use ndarray::prelude::*;

fn main() {
    // Define the input and output dimentions
    let input_size = 784;
    let hidden_size = 128;
    let output_size = 10;

    // Create a Tensorflow graph
    let mut graph = Graph::new();

    // Define the input and output placeholders
    let input = graph.new_placeholder(f32::to_datatype(&graph.operation_scope("input")), &[-1, input_size]).unwrap();
    let labels = graph.new_placeholder(f32::to_datatype(&graph.operation_scope("labels")), &[-1, output_size]).unwrap();

    // Define the weights and biases for the neural network
    let weight1 = graph.new_variable(&[input_size, hidden_size], &ndarray::Array::random((input_size, hidden_size), ndarray::StandardNormal)).unwrap();
    let biases1 = graph.new_variable(&[hidden_size], &ndarray::Array::zeros(hidden_size)).unwrap();
    let weights2 = graph.new_variable(&[hidden_size, output_size], &ndarray::Array::random((hidden_size, output_size), ndarray::StandardNormal)).unwrap();
    let biases2 = graph.new_variable(&[output_size], &ndarray::Array::zeros(output_size)).unwrap();

    // Define the operations for the neural network
    let hidden_layer = graph.apply(MatMul::new(input.clone(), weights1.clone(), "hidden_layer")).unwrap();
    let hidden_layer = graph.apply(BiasAdd::new(hidden_layer, biases1.clone(), "hidden_layer_bias")).unwrap();
    let hidden_layer = graph.apply(Relu::new(hidden_layer, "hidden_layer_activation")).unwrap();
    let output_layer = graph.apply(MatMul::new(hidden_layer, weights2.clone(), "output_layer")).unwrap();
    let output_layer = graph.apply(BiasAdd::new(output_layer, biases2.clone(), "output_layer_bias")).unwrap();
    let output_layer = graph.apply(Softmax::new(output_layer, "output_layer_activation")).unwrap();

    // Define the loss function and optimizer
    let cross_entropy = graph.reduce_mean(graph.softmax_cross_with_logits(&output_layer, &labels, None).unwrap(), &[0], false, "cross_entropy").unwrap();
    let optimizer = graph.train_adam(&cross_entropy, 0.001f32, 0.9f32, 0.999f32, 1e-8f32);

    // Create a new TensorFlow session
    let mut session = Session::new(&graph, &Default::default()).unwrap();

    // Load the MNIST dataset
    let (train_images, train_labels, test_images, test_labels) = tch::vision::mnist::load();

     // Train the neural network
     for epoch in 1..=10 {
        let mut loss = 0.0;

        for (batch_images, batch_labels) in train_images.iter().zip(train_labels.iter()) {
            let feed_dict = graph::FeedDict::new()
                .add(&input, &batch_images)
                .add(&labels, &batch_labels);
            let mut run_args = graph::RunArgs::new();
            run_args.add_target(&optimizer);
            run_args.add_target(&cross_entropy);
            session.run(&mut run_args, &feed_dict).unwrap();

            loss += run_args.fetch(&cross_entropy).unwrap()[0];
        }

        loss /= train_images.len() as f32;
        println!("Epoch {}: loss = {}", epoch, loss);
    }

    // Test the neural network
    let mut correct = 0;

    for (test_image, test_label) in test_images.iter().zip(test_labels.iter()) {
        let feed_dict = graph::FeedDict::new()
            .add(&input, &test_image.expand_dims(0).unwrap().into_dyn())
            .add(&labels, &ndarray::arr1(&[*test_label as i32]).into_dyn());
        let mut run_args = graph::RunArgs::new();
        run_args.add_target(&output_layer);
        let output = session.run(&mut run_args, &feed_dict).unwrap().take::<f32>(&output_layer).unwrap();

        if output.argmax().unwrap() == *test_label as usize {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / test_images.len() as f32;
    println!("Test accuracy = {}", accuracy);
}
