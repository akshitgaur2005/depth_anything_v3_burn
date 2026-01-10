use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/my_model.onnx")
        .embed_states(true)
        .out_dir("model/")
        .run_from_script();
}
