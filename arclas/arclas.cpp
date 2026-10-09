#include <filesystem>
#include <torch/torch.h>
#include "../common.hpp"
#include <fstream>
#include "arclas.hpp"

struct ArclasDataset : torch::data::datasets::Dataset<ArclasDataset> {
	public:
		ArclasDataset(const std::string& x_tensor, const std::string& y_tensor, size_t n_samples) {
			std::unordered_set<int> labels_set;
			
			std::vector<int> labels;
			{
				std::ifstream ifs{"arclas/labels"};
				std::string line;
				while(std::getline(ifs, line)) {
					auto lbl = std::stoi(line);
					labels.push_back(lbl);
					labels_set.insert(lbl);
				}
			}
			
			labels_unique = std::vector<int>(labels_set.begin(), labels_set.end());
			for(size_t i = 0; i < labels.size(); i++) {
				const auto& label = labels[i];
				auto it = std::find(labels_unique.begin(), labels_unique.end(), label);
				if (it != labels_unique.end()) {
					int index = std::distance(labels_unique.begin(), it);
					auto tensor = torch::tensor({index}, torch::kLong);
					label_tensors.emplace_back(tensor);
				}
			}

			for (size_t i = 0; i < n_samples; i++) {
				torch::Tensor x;
				torch::load(x, std::format("arclas/tensors/x_{}.tensor", i));
				x = x.to(torch::kFloat32).div(255.0);			
				data_tensors.emplace_back(x);
			}
		}

		ArclasDataset(
			const ArclasDataset& original,
			const std::vector<size_t>& indices
		) {
			labels_unique = original.labels_unique;
			for (size_t i : indices) {
				data_tensors.push_back(original.data_tensors[i]);
				label_tensors.push_back(original.label_tensors[i]);
			}
		}

		void to(torch::Device dev) {
			for (auto& tensor : data_tensors) {
				tensor = tensor.to(dev);
			}
		
			for (auto& tensor : label_tensors) {
				tensor = tensor.to(dev);
			}
		}

		size_t get_n_classes() const {
			return labels_unique.size();
		}
		
		
		torch::optional<size_t> size() const override {
			return data_tensors.size();
		}

		virtual ExampleType get(size_t index) override {
			return {data_tensors[index], label_tensors[index]};
		}
		std::vector<int> labels_unique;
	private:
		std::vector<torch::Tensor> data_tensors;
		std::vector<torch::Tensor> label_tensors;

};

size_t n_samples(const std::string& dir) {
	size_t n = 0;
	for(auto& entry : std::filesystem::directory_iterator(dir)) {
		if(entry.is_regular_file() && entry.path().extension() == ".tensor") {
			n++;
		}
	}
	return n;
}

int main() {
	torch::Device device(torch::cuda::is_available()
        ? torch::kCUDA
        : torch::kCPU);
	
	ArclasDataset alldata{"x.tensor", "y.tensor", n_samples("arclas/tensors")};

	// Create a new Net.
	auto net = std::make_shared<Arclas>(alldata.get_n_classes());
	net->to(device);

	// Instantiate an SGD optimization algorithm to update our Net's parameters.
	torch::optim::SGDOptions opt{0.01};
	opt.momentum(0.9);
	opt.weight_decay(1e-4);
	torch::optim::SGD optimizer(net->parameters(), opt);

	const size_t total = alldata.size().value();
	const size_t train_size = static_cast<size_t>(total * 0.8);

	auto indices = torch::randperm(total, torch::kLong);

	std::vector<size_t> train_indices;
	std::vector<size_t> test_indices;

	for (size_t i = 0; i < train_size; ++i)
		train_indices.push_back(indices[i].item<int64_t>());

	for (size_t i = train_size; i < total; ++i)
		test_indices.push_back(indices[i].item<int64_t>());

	ArclasDataset train_loader(alldata, train_indices);
	ArclasDataset test_loader(alldata, test_indices);

	float last_test_loss = INFINITY;
	constexpr size_t N_EPOCHS_WITHOUT_IMPR = 20;
	size_t without_impr = 0;

	std::vector<torch::Tensor> predictions;
	std::vector<torch::Tensor> targets;


	for (size_t epoch = 1; epoch <= 500; ++epoch) {
		if(without_impr >= N_EPOCHS_WITHOUT_IMPR) break;
		size_t batch_index = 0;
		std::vector<torch::Tensor> predictions;
		std::vector<torch::Tensor> targets;

		float total_loss = 0;
		size_t n = 0;
		float avg_loss;

		// Iterate the data loader to yield batches from the dataset.
		for (size_t i = 0; i < train_loader.size(); i++) {
			auto batch = train_loader.get(i);
			batch.data = batch.data
				.unsqueeze(0)
				.to(device);

			batch.target = batch.target.to(device);
			// Reset gradients.
			optimizer.zero_grad();
			// Execute the model on the input data.
			torch::Tensor prediction = net->forward(batch.data);
			predictions.push_back(prediction);
			targets.push_back(batch.target);
			// Compute a loss value to judge the prediction of our model.
			torch::Tensor loss = torch::nll_loss(prediction, batch.target);
			// Compute gradients of the loss w.r.t. the parameters of our model.
			loss.backward();
			// Update the parameters based on the calculated gradients.
			optimizer.step();
			total_loss += loss.item<float>();
			n++;
		}
		avg_loss = total_loss/n;

		auto all_predictions = torch::cat(predictions, 0);
		auto all_targets = torch::cat(targets, 0);
		std::cout << "Epoch: " << epoch << std::endl;

		std::cout << "Loss: " << avg_loss <<
		" | Accuracy: " << accuracy(all_predictions, all_targets).item<float>() <<
		" | Recall: " << recall_macro(all_predictions, all_targets, alldata.get_n_classes()).item<float>() <<
		" | F score: " << f_score(all_predictions, all_targets, alldata.get_n_classes()).item<float>() << std::endl;

		predictions.clear();
		targets.clear();

		net->eval();
		torch::NoGradGuard no_grad;
		total_loss = 0;
		n = 0;
		
		for (size_t i = 0; i < test_loader.size(); i++) {
			auto batch = test_loader.get(i);
			batch.data = batch.data
				.unsqueeze(0)
				.to(device);

			batch.target = batch.target.to(device);
			torch::Tensor prediction = net->forward(batch.data);
			torch::Tensor loss = torch::nll_loss(prediction, batch.target);
			predictions.push_back(prediction);
			targets.push_back(batch.target);
			total_loss += loss.item<float>();
			n++;
		}

		all_predictions = torch::cat(predictions, 0);
		all_targets = torch::cat(targets, 0);

		avg_loss = total_loss / n;

		if(avg_loss < last_test_loss) {
			last_test_loss = avg_loss;
			without_impr = 0;
			torch::save(net, "arclas/arclas.pt");
		} else {
			without_impr++;
		}

		std::cout << "Test average Loss: " << avg_loss << 
		" | Test Accuracy: " << accuracy(all_predictions, all_targets).item<float>() <<
		" | Test Recall: " << recall_macro(all_predictions, all_targets, alldata.get_n_classes()).item<float>() <<
		" | Test F score: " << f_score(all_predictions, all_targets, alldata.get_n_classes()).item<float>() << std::endl;
		std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
		
		predictions.clear();
		targets.clear();	
		net->train();
	}
}
