use gtk4::{
    prelude::*, Application, ApplicationWindow, Box as GtkBox, Button, DrawingArea, GestureDrag, Label, Orientation,
};
use image::{ImageBuffer, Rgba, RgbaImage};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use tch::{nn, Device, Tensor};

#[derive(Debug)]
struct Net {
    fc1: nn::Conv2D,
    fc2: nn::Conv2D,
	fc3: nn::Conv2D,
	fc4: nn::Conv2D,
    fc5: nn::Linear,
    device: Device,
}

impl Net {
    fn new(vs: &nn::Path, device: Device) -> Net {
        let conv_config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let fc1 = nn::conv2d(vs / "fc1", 1, 16, 3, conv_config);
		let conv_config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let fc2 = nn::conv2d(vs / "fc2", 16, 32, 3, conv_config);
		let fc3 = nn::conv2d(vs / "fc3", 32, 64, 3, conv_config);
        let fc4 = nn::conv2d(vs / "fc4", 64, 128, 3, conv_config);
		let fc5 = nn::linear(vs / "fc5", 128, 10, Default::default());
        Net {
            fc1,
            fc2,
            fc3,
			fc4,
			fc5,
            device,
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1)
            .relu()
			.avg_pool2d(2, [], 0, false, true, None)
            .apply(&self.fc2)
            .relu()
			.avg_pool2d(2, [], 0, false, true, None)
			.apply(&self.fc3)
            .relu()
			.avg_pool2d(2, [], 0, false, true, None)
            .apply(&self.fc4)
			.relu()
			.avg_pool2d(2, [], 0, false, true, None)
			.flatten(1, -1)
			.apply(&self.fc5)
            .log_softmax(1, tch::Kind::Float)
    }
}

fn load_model(model_path: &str, device: Device) -> Result<(Net, nn::VarStore), String> {
    let mut vs = nn::VarStore::new(device);
    let net = Net::new(&vs.root(), device);

    // Try loading via CModule first (handles LibTorch C++ torch::save(net, "mnist.pt") archives)
    match tch::CModule::load_on_device(model_path, device) {
        Ok(cmod) => {
            let params = cmod
                .named_parameters()
                .map_err(|e| format!("Failed to read parameters from CModule: {}", e))?;
            let mut vars = vs.variables();
            tch::no_grad(|| {
                for (name, param_tensor) in params {
                    if let Some(var) = vars.get_mut(&name) {
                        var.copy_(&param_tensor);
                    } else {
                        eprintln!("Warning: parameter {} not found in Net", name);
                    }
                }
            });
            Ok((net, vs))
        }
        Err(cmod_err) => {
            // Fallback to direct VarStore loading (e.g. safetensors / varstore binary formats)
            match vs.load(model_path) {
                Ok(()) => Ok((net, vs)),
                Err(vs_err) => Err(format!(
                    "Failed to load model from '{}':\n - CModule error: {}\n - VarStore error: {}",
                    model_path, cmod_err, vs_err
                )),
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_model>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let device = Device::cuda_if_available(); 
    let (net, _vs) = match load_model(model_path, device) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };
    let net = Arc::new(net);

    let app = Application::builder()
        .application_id("com.example.mnist")
        .build();

    app.connect_activate(move |app| {
        build_ui(app, net.clone());
    });

    app.run_with_args(&args[..1]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_mnist_model() {
        let device = Device::Cpu;
        let (net, _vs) = load_model("../mnist.pt", device).expect("Model should load successfully");
        let dummy_input = Tensor::zeros([1, 1, 28, 28], (tch::Kind::Float, device));
        let output = net.forward(&dummy_input);
        assert_eq!(output.size(), vec![1, 10]);
    }
}


fn build_ui(app: &Application, net: Arc<Net>) {
    let window = ApplicationWindow::builder()
        .application(app)
        .title("MNIST Digit Recognition")
        .default_width(400)
        .default_height(500)
        .build();

    let main_box = GtkBox::new(Orientation::Vertical, 10);
    main_box.set_margin_top(10);
    main_box.set_margin_bottom(10);
    main_box.set_margin_start(10);
    main_box.set_margin_end(10);

    let label = Label::new(Some("Draw a digit (0-9)"));
    main_box.append(&label);

    let drawing_area = DrawingArea::new();
    drawing_area.set_content_width(280);
    drawing_area.set_content_height(280);
    drawing_area.set_css_classes(&["drawing-area"]);

    let pixels = Rc::new(RefCell::new(vec![255u8; 280 * 280]));
    let pixels_clone = pixels.clone();

    drawing_area.set_draw_func(move |_, context, _, _| {
        let pixels = pixels_clone.borrow();
        let mut surface = cairo::ImageSurface::create(
            cairo::Format::A8,
            280,
            280,
        ).unwrap();
        {
            let mut data = surface.data().unwrap();
            data.copy_from_slice(&pixels);
        }
        context.set_source_surface(&surface, 0.0, 0.0).unwrap();
        context.paint().unwrap();
    });

    let gesture = GestureDrag::new();
    let pixels_draw = pixels.clone();
    let drawing_area_clone = drawing_area.clone();
    gesture.connect_drag_begin(move |_, x, y| {
        draw_on_canvas(&pixels_draw, x as i32, y as i32, 10);
        drawing_area_clone.queue_draw();
    });

    let pixels_draw = pixels.clone();
    let drawing_area_clone = drawing_area.clone();
    gesture.connect_drag_update(move |gesture, offset_x, offset_y| {
        if let Some((start_x, start_y)) = gesture.start_point() {
            let x = start_x + offset_x;
            let y = start_y + offset_y;
            draw_on_canvas(&pixels_draw, x as i32, y as i32, 10);
            drawing_area_clone.queue_draw();
        }
    });

    drawing_area.add_controller(gesture);

    drawing_area.set_can_focus(true);

    main_box.append(&drawing_area);

    let button_box = GtkBox::new(Orientation::Horizontal, 10);

    let predict_button = Button::with_label("Recognize");
    let label_clone = label.clone();
    let pixels_clone = pixels.clone();
    let net_clone = net.clone();

    predict_button.connect_clicked(move |_| {
        let pixels = pixels_clone.borrow();
        match recognize_digit(&net_clone, &pixels) {
            Ok(digit) => {
                label_clone.set_text(&format!("Recognized digit: {}", digit));
            }
            Err(e) => {
                label_clone.set_text(&format!("Error: {}", e));
            }
        }
    });

    button_box.append(&predict_button);

    let clear_button = Button::with_label("Clear");
    let pixels_clone = pixels.clone();
    let drawing_area_clone = drawing_area.clone();

    clear_button.connect_clicked(move |_| {
        let mut pixels = pixels_clone.borrow_mut();
        pixels.fill(255);
        drawing_area_clone.queue_draw();
    });

    button_box.append(&clear_button);
    main_box.append(&button_box);

    window.set_child(Some(&main_box));
    window.present();
}

fn draw_on_canvas(pixels: &Rc<RefCell<Vec<u8>>>, x: i32, y: i32, radius: i32) {
    let mut pixels = pixels.borrow_mut();
    for dx in -radius..=radius {
        for dy in -radius..=radius {
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && nx < 280 && ny >= 0 && ny < 280 {
                let idx = (ny * 280 + nx) as usize;
                if idx < pixels.len() {
                    pixels[idx] = 0; // Black
                }
            }
        }
    }
}

fn recognize_digit(net: &Net, canvas_pixels: &[u8]) -> Result<u32, String> {
    // Resize canvas (280x280) to MNIST (28x28)
    let img_rgba: RgbaImage = ImageBuffer::from_fn(280, 280, |x, y| {
        let idx = (y * 280 + x) as usize;
        let val = canvas_pixels[idx];
        Rgba([val, val, val, 255])
    });

    let img_small = image::imageops::resize(&img_rgba, 28, 28, image::imageops::FilterType::Lanczos3);

    // Convert to normalized grayscale (0.0 to 1.0)
    let mut input_data: Vec<f32> = vec![0.0; 28 * 28];
    for (i, pixel) in img_small.pixels().enumerate() {
        // Convert to grayscale and normalize
        input_data[i] = (pixel[0] as f32) / 255.0;
    }

    let inverted_data: Vec<f32> = input_data.into_iter().map(|x| 1.0 - x).collect();

    // Create tensor of shape [1, 1, 28, 28] on model's device
    let input = Tensor::from_slice(&inverted_data)
        .view([1, 1, 28, 28])
        .to_device(net.device);

    let output = tch::no_grad(|| net.forward(&input));

    let pred = output.argmax(Some(-1), false).int64_value(&[0]) as u32;

    Ok(pred)
}
