#include <torch/torch.h>

struct Block : torch::nn::Module {
	Block(int in_channels, int out_channels, int stride = 1) {
		conv1 = register_module("conv1", torch::nn::Conv2d{torch::nn::Conv2dOptions{
			in_channels, out_channels, 3
		}.padding(1).stride(stride)});
		bn1 = register_module("bn1", torch::nn::BatchNorm2d{out_channels});
		conv2 = register_module("conv2", torch::nn::Conv2d{torch::nn::Conv2dOptions{
			out_channels, out_channels, 3
		}.padding(1).stride(1)});
		bn2 = register_module("bn2", torch::nn::BatchNorm2d{out_channels});

		if (stride != 1 || in_channels != out_channels) {
			proj = true;
			shortcut = register_module("shortcut", torch::nn::Conv2d{torch::nn::Conv2dOptions{
				in_channels, out_channels, 1
			}.stride(stride)});
			bn_shortcut = register_module("bn_shortcut", torch::nn::BatchNorm2d{out_channels});
		}
	}

	torch::Tensor forward(torch::Tensor x) {
		torch::Tensor y = x;
		x = conv1->forward(x);
		x = bn1->forward(x);
		x = torch::relu(x);
		x = conv2->forward(x);
		x = bn2->forward(x);
		
		if(proj) {
			y = shortcut->forward(y);
			y = bn_shortcut->forward(y);
		}
		
		return torch::relu(x + y);
	}

	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Conv2d conv2{nullptr};
	torch::nn::BatchNorm2d bn2{nullptr};
	torch::nn::Conv2d shortcut{nullptr};
	torch::nn::BatchNorm2d bn_shortcut{nullptr};
	bool proj = false;
};

// Define a new Module.
struct Net : torch::nn::Module {
	Net(int num_classes = 10) : num_classes(num_classes) {
		conv1 = register_module("conv1", torch::nn::Conv2d{torch::nn::Conv2dOptions{
			1, 64, 3
		}.padding(1).stride(1).bias(false)});
		bn1 = register_module("bn1", torch::nn::BatchNorm2d{64});

		b1 = register_module("block1", std::make_shared<Block>(64, 64));
		b2 = register_module("block2", std::make_shared<Block>(64, 64));
		b3 = register_module("block3", std::make_shared<Block>(64, 128, 2));
		b4 = register_module("block4", std::make_shared<Block>(128, 128));
		b5 = register_module("block5", std::make_shared<Block>(128, 256, 2));
		b6 = register_module("block6", std::make_shared<Block>(256, 256));
		b7 = register_module("block7", std::make_shared<Block>(256, 512, 2));
		b8 = register_module("block8", std::make_shared<Block>(512, 512));

		avg = register_module("avg", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions{{1,1}}));

		fc = register_module("fc", torch::nn::Linear(512, num_classes));
	}

	// Implement the Net's algorithm.
	torch::Tensor forward(torch::Tensor x) {
		x = conv1->forward(x);
		x = bn1->forward(x);
		x = torch::relu(x);

		x = b1->forward(x);
		x = b2->forward(x);
		x = b3->forward(x);
		x = b4->forward(x);
		x = b5->forward(x);
		x = b6->forward(x);
		x = b7->forward(x);
		x = b8->forward(x);

		x = avg->forward(x);
		x = x.view({x.size(0), -1});
		x = fc->forward(x);

		return torch::log_softmax(x, 1);
	}

	// Use one of many "standard library" modules.
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::AdaptiveAvgPool2d avg{nullptr};
	std::shared_ptr<Block> b1{nullptr};
	std::shared_ptr<Block> b2{nullptr};
	std::shared_ptr<Block> b3{nullptr};
	std::shared_ptr<Block> b4{nullptr};
	std::shared_ptr<Block> b5{nullptr};
	std::shared_ptr<Block> b6{nullptr};
	std::shared_ptr<Block> b7{nullptr};
	std::shared_ptr<Block> b8{nullptr};
	torch::nn::Linear fc{nullptr};

	int num_classes;
};

static torch::Device get_best_device() {
	if(torch::cuda::is_available()) {
		return torch::Device(torch::kCUDA);
	}
	if(torch::mps::is_available()) {
		return torch::Device(torch::kMPS);
	}
	if(torch::is_vulkan_available()) {
		return torch::Device(torch::kVulkan);
	}
	return torch::Device(torch::kCPU);
}