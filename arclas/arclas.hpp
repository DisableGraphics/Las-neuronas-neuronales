#pragma once
#include <torch/torch.h>

struct Arclas : torch::nn::Module {
	Arclas(size_t n_classes) {
		// Construct and register two Linear submodules.
		auto conv_options = torch::nn::Conv2dOptions(3, 64, 5)
			.padding(1);
		auto module = torch::nn::Conv2d(conv_options);

		fc1 = register_module("fc1", std::move(module));
		auto conv_options2 = torch::nn::Conv2dOptions(64, 32, 3)
			.padding(1);
		auto module2 = torch::nn::Conv2d(conv_options2);
		fc2 = register_module("fc2", std::move(module2));

		auto conv_options3 = torch::nn::Conv2dOptions(32, 16, 3)
			.padding(1);
		auto module3 = torch::nn::Conv2d(conv_options3);
		fc3 = register_module("fc3", std::move(module3));

		fc7 = register_module("fc7", torch::nn::Linear(12544, 512));
		fc8 = register_module("fc8", torch::nn::Linear(512, n_classes));
	}

	// Implement the Net's algorithm.
	torch::Tensor forward(torch::Tensor x) {
		// conv
		x = fc1->forward(x);
		x = torch::relu(x);
		// yet another conv
		x = torch::relu(fc2->forward(x));
		x = torch::avg_pool2d(x, 3);
		// Yet another conv
		x = torch::relu(fc3->forward(x));
		x = torch::avg_pool2d(x, 3);
		x = x.flatten(1);
		x = torch::relu(fc7->forward(x));
		x = torch::dropout(x, 0.5, is_training());
		x = torch::log_softmax(fc8->forward(x), /*dim=*/1);
		return x;
	}

	// Use one of many "standard library" modules.
	torch::nn::Conv2d fc1{nullptr};
	torch::nn::Conv2d fc2{nullptr};
	torch::nn::Conv2d fc3{nullptr};
	torch::nn::Linear fc7{nullptr};
	torch::nn::Linear fc8{nullptr};
};