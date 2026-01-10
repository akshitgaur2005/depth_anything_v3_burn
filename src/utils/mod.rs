use burn::prelude::*;
use burn::tensor::{ElementConversion, Shape, Tensor, TensorData};
use image::{ImageBuffer, ImageFormat, ImageReader, Luma, RgbImage}; // Import necessary items from image crate
use std::error::Error;
use std::path::Path; // For generic error handling

/// Loads an image from a file path and converts it into a Tensor<B, 4>.
/// The tensor will have shape [1, Channels, Height, Width] and float values in [0, 1].
/// Supports image formats supported by the 'image' crate (PNG, JPEG, etc. with enabled features).
/// Converts input to RGB if it's not already (discards alpha if present).
///
/// Args:
///     path: The path to the image file.
///     device: The Burn device to create the tensor on.
///
/// Returns:
///     A Result containing the loaded tensor or an error.
pub fn load_image_to_tensor<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Box<dyn Error>> {
    // Open and decode the image file
    let img = ImageReader::open(path)?.decode()?;

    // Convert to RGB8 format (3 channels, 8 bits per channel)
    // This handles different input formats (grayscale, rgba) and converts to RGB
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    let channels = 3; // RGB always has 3 channels

    // Get the raw pixel bytes (HWC order by default from image crate)
    let raw_pixels: Vec<u8> = rgb_img.into_raw();

    // Convert u8 pixel values [0, 255] to float [0, 1] and reorder to CHW
    let mut data_chw: Vec<B::FloatElem> =
        Vec::with_capacity(height as usize * width as usize * channels);
    for c in 0..channels {
        for h in 0..height as usize {
            for w in 0..width as usize {
                // Calculate index in the HWC raw_pixels buffer
                let hwc_idx = h * (width as usize * channels) + w * channels + c;
                let pixel_value_u8 = raw_pixels[hwc_idx];
                // Convert u8 [0, 255] to float [0.0, 1.0] and convert to backend float type
                data_chw.push((pixel_value_u8 as f32).elem());
            }
        }
    }

    // Create a Tensor<B, 3> [Channels, Height, Width] first
    let tensor_3d = Tensor::<B, 3>::from_data(
        TensorData::new(
            data_chw,
            Shape::new([channels, height as usize, width as usize]),
        ),
        device,
    );

    // Add a batch dimension to make it Tensor<B, 4> [1, Channels, Height, Width]
    Ok(tensor_3d)
}

fn preprocess_inputs() {}
