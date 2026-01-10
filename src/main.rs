mod model;
mod utils;

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::LibTorch;
use burn::module::Module;
use burn::prelude::Backend;
use burn::{tensor, Tensor};
use model::my_model::Model;
use utils::*;

type Back = LibTorch<f32>;

fn main() {
    let device = LibTorchDevice::Cuda(0);
    println!("Using device: {device:?}");

    // Create model instance and load weights from target dir default device
    // let model: Model<Back> = Model::default().to_device(&device);
    let model: Model<Back> =
        Model::from_file(concat!(env!("OUT_DIR"), "/model/my_model.bpk"), &device);

    // Create input tensor (replace with your actual input)
    let input = tensor::Tensor::<Back, 4>::zeros([1, 3, 280, 504], &device);

    println!("Beginning computation...");
    // Perform inference
    let output = model.forward(input);

    println!("Model output: {:?}", output);
}

fn inference<B: Backend>(model: Model<Back>, image: Tensor<B, 4>) {}
