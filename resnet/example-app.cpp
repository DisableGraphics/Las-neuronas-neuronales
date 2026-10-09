#include <ATen/core/TensorBody.h>
#include <ATen/ops/batch_norm.h>
#include <ATen/ops/is_vulkan_available.h>
#include <ATen/ops/relu.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <cmath>
#include <cstddef>
#include <string>
#include <torch/cuda.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/mps.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/pooling.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/options/pooling.h>
#include <torch/optim/sgd.h>
#include <torch/torch.h>
#include "resnet.hpp"

int main() {
	torch::Device dev{get_best_device()};

	// Create a new Net.
	auto net = std::make_shared<Net>();
	net->to(dev);

	// Create a multi-threaded data loader for the MNIST dataset.
	auto data_loader = torch::data::make_data_loader(
			torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTrain).map(
					torch::data::transforms::Stack<>()),
			/*batch_size=*/64);
	auto test_loader = torch::data::make_data_loader(
		torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTest)
		.map(
			torch::data::transforms::Stack<>()
		),
		/*batch_size=*/64);

	// Instantiate an SGD optimization algorithm to update our Net's parameters.
	torch::optim::SGDOptions opt{0.01};
	opt.momentum(0.9);
	opt.weight_decay(1e-4);
	torch::optim::SGD optimizer(net->parameters(), opt);
	float prev_loss = +INFINITY;
	auto start = std::chrono::steady_clock::now();

	for (size_t epoch = 1; epoch <= 20; ++epoch) {
		size_t batch_index = 0;
		// Iterate the data loader to yield batches from the dataset.
		for (auto& batch : *data_loader) {
			// Reset gradients.
			optimizer.zero_grad();
			// Execute the model on the input data.
			torch::Tensor prediction = net->forward(batch.data.to(dev));
			// Compute a loss value to judge the prediction of our model.
			torch::Tensor loss = torch::nll_loss(prediction, batch.target.to(dev));
			// Compute gradients of the loss w.r.t. the parameters of our model.
			loss.backward();
			// Update the parameters based on the calculated gradients.
			optimizer.step();
			// Output the loss and checkpoint every 100 batches.
			if (++batch_index % 100 == 0) {
				std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
									<< " | Loss: " << loss.item<float>() << std::endl;				
			}
		}
		float total_loss = 0.0f;
		size_t n = 0;
		net->eval();
		{
			torch::NoGradGuard no_grad;
			for(auto& batch : *test_loader) {
				torch::Tensor prediction = net->forward(batch.data.to(dev));
				torch::Tensor loss = torch::nll_loss(prediction, batch.target.to(dev));
				total_loss += loss.item<float>();
				n++;
				if (++batch_index % 100 == 0) {
					std::cout << "Epoch: " << epoch << " | Test Batch: " << batch_index
										<< " | Test Loss: " << loss.item<float>() << std::endl;
				}
			}
			float avg_losss = total_loss / n;

			if(avg_losss < prev_loss) {
				auto tmpstamp = start.time_since_epoch().count();
				torch::save(net, std::string("net") + std::to_string(tmpstamp) + ".pt");
				prev_loss = avg_losss;
			}
		}
		net->train();
	}
	auto end = std::chrono::steady_clock::now();

	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
		end - start
	).count();

	long long hours = elapsed / 3600;
	long long minutes = (elapsed % 3600) / 60;
	long long seconds = elapsed % 60;

	std::cout << hours << " hours, "
			<< minutes << " minutes, "
			<< seconds << " seconds\n";

}