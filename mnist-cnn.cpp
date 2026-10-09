#include <torch/torch.h>
#include "common.hpp"

// Define a new Module.
struct Net : torch::nn::Module {
	Net() {
		// Construct and register two Linear submodules.
		auto conv_options = torch::nn::Conv2dOptions(1, 16, 3)
			.padding(1);
		auto module = torch::nn::Conv2d(conv_options);

		fc1 = register_module("fc1", std::move(module));
		auto conv_options2 = torch::nn::Conv2dOptions(16, 32, 3)
			.padding(1);
		auto module2 = torch::nn::Conv2d(conv_options2);
		fc2 = register_module("fc2", std::move(module2));

		auto conv_options3 = torch::nn::Conv2dOptions(32, 64, 3)
			.padding(1);
		auto module3 = torch::nn::Conv2d(conv_options3);
		fc3 = register_module("fc3", std::move(module3));
		auto conv_options4 = torch::nn::Conv2dOptions(64, 128, 3)
			.padding(1);
		auto module4 = torch::nn::Conv2d(conv_options4);
		fc4 = register_module("fc4", std::move(module4));
		fc5 = register_module("fc5", torch::nn::Linear(128, 10));
	}

	// Implement the Net's algorithm.
	torch::Tensor forward(torch::Tensor x) {
		// conv
		x = torch::relu(fc1->forward(x));
		x = torch::avg_pool2d(x, 2);
		// yet another conv
		x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
		x = torch::relu(fc2->forward(x));
		x = torch::avg_pool2d(x, 2);
		// Yet another conv
		x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
		x = torch::relu(fc3->forward(x));
		x = torch::avg_pool2d(x, 2);
		// More convs lol
		x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
		x = torch::relu(fc4->forward(x));
		x = torch::avg_pool2d(x, 2);
		// Linear
		x = x.flatten(1);
		x = torch::log_softmax(fc5->forward(x), /*dim=*/1);
		return x;
	}

	// Use one of many "standard library" modules.
	torch::nn::Conv2d fc1{nullptr};
	torch::nn::Conv2d fc2{nullptr};
	torch::nn::Conv2d fc3{nullptr};
	torch::nn::Conv2d fc4{nullptr};
	torch::nn::Linear fc5{nullptr};
};

int main() {
	torch::Device device(torch::cuda::is_available()
        ? torch::kCUDA
        : torch::kCPU);
	// Create a new Net.
	auto net = std::make_shared<Net>();
	net->to(device);

	// Create a multi-threaded data loader for the MNIST dataset.
	auto train_loader = torch::data::make_data_loader(
		torch::data::datasets::MNIST("./data", 
			torch::data::datasets::MNIST::Mode::kTrain).map(
			torch::data::transforms::Stack<>()),
		/*batch_size=*/64);
	
	// Test set
	auto test_loader = torch::data::make_data_loader(
		torch::data::datasets::MNIST("./data", 
			torch::data::datasets::MNIST::Mode::kTest).map(
			torch::data::transforms::Stack<>()),
		/*batch_size=*/64);

	// Instantiate an SGD optimization algorithm to update our Net's parameters.
	torch::optim::AdamW optimizer(net->parameters());

	float last_test_loss = INFINITY;
	constexpr size_t N_EPOCHS_WITHOUT_IMPR = 2;
	size_t without_impr = 0;

	for (size_t epoch = 1; epoch <= 10; ++epoch) {
		if(without_impr == N_EPOCHS_WITHOUT_IMPR) {
			break;
		}
		size_t batch_index = 0;
		// Iterate the data loader to yield batches from the dataset.
		for (auto& batch : *train_loader) {
			batch.data = batch.data.to(device);
			batch.target = batch.target.to(device);
			// Reset gradients.
			optimizer.zero_grad();
			// Execute the model on the input data.
			torch::Tensor prediction = net->forward(batch.data);
			// Compute a loss value to judge the prediction of our model.
			torch::Tensor loss = torch::nll_loss(prediction, batch.target);
			// Compute gradients of the loss w.r.t. the parameters of our model.
			loss.backward();
			// Update the parameters based on the calculated gradients.
			optimizer.step();
			// Output the loss and checkpoint every 100 batches.
			if (++batch_index % 100 == 0) {
				std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
						<< " | Loss: " << loss.item<float>() << " | Accuracy: " 
						<< accuracy(prediction, batch.target).item<float>() << " | Recall: " 
						<< recall_macro(prediction, batch.target, 10).item<float>()  << " | F score: " 
						<< f_score(prediction, batch.target, 10).item<float>() << std::endl;
			}
		}

		net->eval();
		torch::NoGradGuard no_grad;
		
		for (auto& batch : *test_loader) {
			batch.data = batch.data.to(device);
			batch.target = batch.target.to(device);
			torch::Tensor prediction = net->forward(batch.data);
			torch::Tensor loss = torch::nll_loss(prediction, batch.target);

			if (++batch_index % 100 == 0) {
				std::cout << "Test Epoch: " << epoch << " | Test Batch: " << batch_index
					<< " | Test Loss: " << loss.item<float>() << " | Test Accuracy: " 
					<< accuracy(prediction, batch.target).item<float>() << " | Test Recall: " 
					<< recall_macro(prediction, batch.target, 10).item<float>()  << " | Test F score: " 
					<< f_score(prediction, batch.target, 10).item<float>() << std::endl;
					if(loss.item<float>() < last_test_loss) {
						last_test_loss = loss.item<float>();
						torch::save(net, "mnist.pt");
					} else {
						without_impr++;
					}
			}
		}
		
		net->train();
	}
}
