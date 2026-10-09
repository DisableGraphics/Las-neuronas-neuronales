#include <random>
#include <torch/torch.h>
#include <fstream>
#include <filesystem>
#include "common.hpp"
const std::string allowed_characters =
	"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;_'";

const size_t n_letters = allowed_characters.size();


std::vector<std::string> read_lines(const std::string& path) {
	std::vector<std::string> ret;
	std::ifstream ifs{path};
	std::string line;
	while(std::getline(ifs, line)) {
		ret.emplace_back(line);
	}
	return ret;
}

long letterToIndex(char letter) {
	auto pos = allowed_characters.find(letter);

	if (pos != std::string::npos) {
		return static_cast<long>(pos);
	}

	return allowed_characters.size() - 2;
}

std::pair<std::string, int64_t> label_from_output(
	const torch::Tensor& output,
	const std::vector<std::string>& labels)
{
	auto result = output.topk(1);
	auto top_i = std::get<1>(result);
	int64_t label_i = top_i[0].item<int64_t>();
	return {labels[label_i], label_i};
}

torch::Tensor lineToTensor(const std::string& line) {
	auto ret = torch::zeros({(long)line.size(), 1, (long)n_letters});
	for(size_t i = 0; i < line.size(); i++) {
		ret[i][0][letterToIndex(line[i])] = 1;
	}
	return ret;
}

struct ArclasDataset : torch::data::datasets::Dataset<ArclasDataset> {
	public:
		ArclasDataset(const std::string& data_dir) {
			std::unordered_set<std::string> labels_set;
			for(auto& file : std::filesystem::directory_iterator(data_dir)) {
				if(file.is_regular_file() && file.path().extension() == ".txt") {
					const auto label = file.path().stem().string();
					labels_set.insert(label);
					std::vector<std::string> lines = read_lines(file.path());
					for(auto& line : lines) {
						data_tensors.push_back(lineToTensor(line));
						data.emplace_back(line);
						labels.emplace_back(label);
					}
				}
			}

			labels_unique = std::vector<std::string>(labels_set.begin(), labels_set.end());
			for(size_t i = 0; i < labels.size(); i++) {
				const auto& label = labels[i];
				auto it = std::find(labels_unique.begin(), labels_unique.end(), label);

				if (it != labels_unique.end()) {
					int index = std::distance(labels_unique.begin(), it);
					auto tensor = torch::tensor({index}, torch::kLong);
					label_tensors.emplace_back(tensor);
				}
			}
		}

		ArclasDataset(
			const ArclasDataset& original,
			const std::vector<size_t>& indices
		) {
			labels_unique = original.labels_unique;
			for (size_t i : indices) {
				data.push_back(original.data[i]);
				labels.push_back(original.labels[i]);
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
		
		
		torch::optional<size_t> size() const override {
			return data_tensors.size();
		}

		virtual ExampleType get(size_t index) override {
			return {data_tensors[index], label_tensors[index]};
		}

		std::vector<std::string> labels_unique;
	private:
		std::vector<std::string> data;
		std::vector<std::string> labels;
		std::vector<torch::Tensor> data_tensors;
		std::vector<torch::Tensor> label_tensors;

};

// Define a new Module.
struct CharRNN : torch::nn::Module {
	CharRNN(long input_size, long hidden_size, long output_size) {
		// Construct and register two Linear submodules.
		rnn = register_module("rnn", torch::nn::GRU(input_size, hidden_size));
		h2o = register_module("h2o", torch::nn::Linear(hidden_size, output_size));
	}

	// Implement the Net's algorithm.
	torch::Tensor forward(torch::Tensor x) {
		auto inner = rnn->forward(x);
	
		auto hidden = std::get<1>(inner);
	
		// [num_layers, batch_size, hidden_size]
		// -> [batch_size, hidden_size]
		hidden = hidden.squeeze(0);
	
		auto output = h2o->forward(hidden);
	
		return torch::log_softmax(output, /*dim=*/1);
	}	

	// Use one of many "standard library" modules.
	torch::nn::GRU rnn{nullptr};
	torch::nn::Linear h2o{nullptr};
};

std::vector<double> train(
	std::shared_ptr<CharRNN>& rnn,
	ArclasDataset& training_data,
	torch::Device dev,
	int n_epoch,
	int n_batch_size,
	int report_every,
	torch::nn::NLLLoss& criterion,
	torch::optim::Optimizer& optimizer)
{
	std::vector<double> all_losses;

	for (int epoch = 1; epoch <= n_epoch; ++epoch) {

		rnn->zero_grad();

		// Create shuffled indices
		std::vector<int64_t> indices(training_data.size().value());
		std::iota(indices.begin(), indices.end(), 0);

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(indices.begin(), indices.end(), g);

		double current_loss = 0.0;

		// Process minibatches
		for (size_t start = 0; start < indices.size(); start += n_batch_size) {

			size_t end = std::min(
				start + static_cast<size_t>(n_batch_size),
				indices.size()
			);

			torch::Tensor batch_loss = torch::zeros({}, torch::TensorOptions().device(dev));

			for (size_t j = start; j < end; ++j) {

				int64_t i = indices[j];
			
				const auto& example = training_data.get(i);
			
				auto input = example.data.to(dev);
				auto target = example.target.to(dev);
			
				auto output = rnn->forward(input);
			
				auto loss = criterion(
					output,
					target
				);
			
				batch_loss += loss;
			}
			

			// Backpropagate
			batch_loss.backward();

			// Clip gradients
			torch::nn::utils::clip_grad_norm_(
				rnn->parameters(),
				3.0
			);

			// Update parameters
			optimizer.step();
			optimizer.zero_grad();

			current_loss +=
				batch_loss.item<double>() /
				static_cast<double>(end - start);
		}

		double epoch_loss =
			current_loss /
			static_cast<double>(
				(double)(training_data.size().value() + n_batch_size - 1) / n_batch_size
			);

		all_losses.push_back(epoch_loss);

		if (epoch % report_every == 0) {
			std::cout
				<< epoch
				<< " ("
				<< static_cast<int>(
					100.0 * epoch / n_epoch
				)
				<< "%):\t average batch loss = "
				<< epoch_loss
				<< '\n';
		}
	}

	return all_losses;
}

void evaluate(
	std::shared_ptr<CharRNN>& rnn,
	ArclasDataset& testing_data,
	const std::vector<std::string>& classes,
	torch::Device device)
{
	const int64_t n_classes = classes.size();

	// Confusion matrix lives on CPU since we're only using it for statistics.
	auto confusion = torch::zeros(
		{n_classes, n_classes},
		torch::kFloat32
	);

	rnn->eval();

	{
		torch::NoGradGuard no_grad;

		for (size_t i = 0; i < testing_data.size().value(); ++i) {

			const auto& example = testing_data.get(i);

			// Dataset tensors may already be on the device, but this also
			// makes the function work regardless of how the dataset was created.
			auto text = example.data.to(device);
			auto target = example.target.to(device);

			auto output = rnn->forward(text);

			auto [guess, guess_i] =
				label_from_output(output, classes);

			// target has shape [1]
			int64_t label_i = target.item<int64_t>();

			// confusion is CPU, so update it with scalar indexing.
			confusion[label_i][guess_i] += 1.0f;
		}
	}

	// Normalize each row.
	for (int64_t i = 0; i < n_classes; ++i) {
		auto denom = confusion[i].sum().item<float>();

		if (denom > 0.0f) {
			confusion[i] /= denom;
		}
	}

	// Print the matrix.
	std::cout << "\nConfusion matrix:\n\n";

	// Header
	std::cout << std::setw(15) << "Actual \\ Pred";

	for (const auto& class_name : classes) {
		std::cout << std::setw(10) << class_name;
	}

	std::cout << '\n';

	// Separator
	std::cout << std::string(
		15 + 10 * n_classes,
		'-'
	) << '\n';

	// Rows
	for (int64_t i = 0; i < n_classes; ++i) {

		std::cout << std::setw(15) << classes[i];

		for (int64_t j = 0; j < n_classes; ++j) {

			float value = confusion[i][j].item<float>();

			std::cout
				<< std::setw(10)
				<< std::fixed
				<< std::setprecision(2)
				<< value;
		}

		std::cout << '\n';
	}

	rnn->train();
}

int main() {
	torch::Device device(torch::cuda::is_available()
		? torch::kCUDA
		: torch::kCPU);
	// Create a new Net.
	auto net = std::make_shared<CharRNN>(n_letters, 128, 18);
	net->to(device);

	auto alldata = ArclasDataset("data/names");

	torch::manual_seed(2024);

	const size_t total = alldata.size().value();
	const size_t train_size = static_cast<size_t>(total * 0.8);

	auto indices = torch::randperm(total, torch::kLong);

	std::vector<size_t> train_indices;
	std::vector<size_t> test_indices;

	for (size_t i = 0; i < train_size; ++i)
		train_indices.push_back(indices[i].item<int64_t>());

	for (size_t i = train_size; i < total; ++i)
		test_indices.push_back(indices[i].item<int64_t>());

	ArclasDataset train_set(alldata, train_indices);
	ArclasDataset test_set(alldata, test_indices);

	train_set.to(device);
	test_set.to(device);

	std::cout << "train examples = " << train_set.size().value() << '\n';
	std::cout << "validation examples = " << test_set.size().value() << '\n';

	auto input = lineToTensor("Albert");
	input = input.to(device);
	auto output = net->forward(input);
	std::cout << label_from_output(output, alldata.labels_unique) << std::endl;

	auto opt = torch::optim::SGD(net->parameters(), {0.15});
	auto nlloss = torch::nn::NLLLoss();

	auto trained = train(net, train_set, device, 27, 256, 5, nlloss, opt);
	evaluate(net, test_set, test_set.labels_unique, device);
	torch::save(net, "names.pt");
}
