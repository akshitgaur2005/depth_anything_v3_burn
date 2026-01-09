mod model;

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::LibTorch;
use burn::tensor;
use model::my_model::Model;

fn main() {
    let device = LibTorchDevice::default();
    println!("Using device: {device:?}");

    // Create model instance and load weights from target dir default device
    let model: Model<LibTorch<f32>> = Model::default();

    // Create input tensor (replace with your actual input)
    let input = tensor::Tensor::<LibTorch<f32>, 4>::zeros([1, 3, 280, 504], &device);

    println!("Beginning computation...");
    // Perform inference
    let output = model.forward(input);

    println!("Model output: {:?}", output);
}
