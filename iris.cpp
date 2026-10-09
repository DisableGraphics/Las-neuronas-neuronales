#include "common.hpp"

#include <torch/torch.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#define HIDDEN_SIZE 10
#define INPUT_SIZE 4
#define OUTPUT_SIZE 3
#define NUM_CLASSES 3

struct Net : torch::nn::Module {
	Net() {
		fc1 = register_module("fc1", torch::nn::Linear(INPUT_SIZE, HIDDEN_SIZE));
		fc2 = register_module("fc2", torch::nn::Linear(HIDDEN_SIZE, HIDDEN_SIZE));
		fc3 = register_module("fc3", torch::nn::Linear(HIDDEN_SIZE, OUTPUT_SIZE));
	}

	// Implement the Net's algorithm.
	torch::Tensor forward(torch::Tensor x) {
		x = torch::tanh(fc1->forward(x.reshape({x.size(0), INPUT_SIZE})));
		x = torch::sin(fc2->forward(x));
		x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
		return x;
	}

	// Use one of many "standard library" modules.
	torch::nn::Linear fc1{nullptr};
	torch::nn::Linear fc2{nullptr}, fc3{nullptr};
};

struct IrisDataset {
    torch::Tensor X, X_val;
    torch::Tensor y, y_val;
};

IrisDataset load_iris_csv(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + filename);
    }

    std::vector<float> features;
    std::vector<int64_t> labels;

    // Map string labels to integers.
    const std::unordered_map<std::string, int64_t> label_map = {
        {"Setosa", 0},
        {"Versicolor", 1},
        {"Virginica", 2}
    };

    std::string line;

    // Skip header.
    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty");
    }

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);

        std::string field;
        std::vector<std::string> fields;

        while (std::getline(ss, field, ',')) {
            // Remove surrounding quotes.
            if (field.size() >= 2 &&
                field.front() == '"' &&
                field.back() == '"') {
                field = field.substr(1, field.size() - 2);
            }

            fields.push_back(field);
        }

        if (fields.size() != 5) {
            throw std::runtime_error(
                "Expected 5 columns, got " +
                std::to_string(fields.size()) +
                " in line: " + line
            );
        }

        // First four columns are numerical.
        for (int i = 0; i < 4; ++i) {
            features.push_back(std::stof(fields[i]));
        }

        // Last column is the class.
        auto it = label_map.find(fields[4]);

        if (it == label_map.end()) {
            throw std::runtime_error(
                "Unknown label: " + fields[4]
            );
        }

        labels.push_back(it->second);
    }

    const int64_t num_samples = static_cast<int64_t>(labels.size());

    if (features.size() != num_samples * 4) {
        throw std::runtime_error("Invalid feature/label count");
    }

    auto X = torch::from_blob(
		features.data(),
		{num_samples, 4},
		torch::TensorOptions().dtype(torch::kFloat32)
	).clone();
	
	auto y = torch::from_blob(
		labels.data(),
		{num_samples},
		torch::TensorOptions().dtype(torch::kInt64)
	).clone();
	
	auto permutation = torch::randperm(num_samples);
	
	X = X.index_select(0, permutation);
	y = y.index_select(0, permutation);
	
	const int64_t n_samples_train = num_samples * 0.7;
	const int64_t n_samples_val = num_samples - n_samples_train;
	
	auto X_train = X.narrow(0, 0, n_samples_train);
	auto y_train = y.narrow(0, 0, n_samples_train);
	
	auto X_val = X.narrow(0, n_samples_train, n_samples_val);
	auto y_val = y.narrow(0, n_samples_train, n_samples_val);

    return {X, X_val, y, y_val};
}


int main() {
	torch::Device device(torch::cuda::is_available()
        ? torch::kCPU
        : torch::kCPU);

    std::cout << "Using device: "
              << (device.is_cuda() ? "CUDA" : "CPU")
              << std::endl;
	// Create a new Net.
	auto net = std::make_shared<Net>();
	net->to(device);

	torch::optim::AdamW optimizer(net->parameters());

	IrisDataset id = load_iris_csv("iris.csv");
	id.X = id.X.to(device);
    id.y = id.y.to(device);
    id.X_val = id.X_val.to(device);
    id.y_val = id.y_val.to(device);

	double prev_val_loss = +INFINITY;

	for (size_t iteration = 1; iteration <= 1500; ++iteration) {
		optimizer.zero_grad();
		torch::Tensor prediction = net->forward(id.X);
		torch::Tensor loss = torch::nll_loss(prediction, id.y);
		loss.backward();
		optimizer.step();

		torch::Tensor pred2 = net->forward(id.X_val);
		torch::Tensor valid_loss = torch::nll_loss(pred2, id.y_val);
		if (iteration % 100 == 0) {
			std::cout << "Epoch: " << iteration
					<< " \n\t- Loss: " << loss.item<float>() << "  \n\t- Validation loss: " << valid_loss.item<float>() << "  \n\t- Training Accuracy: " 
					<< accuracy(prediction, id.y).item<float>()  << "  \n\t- Validation Accuracy: " 
					<< accuracy(pred2, id.y_val).item<float>() << "  \n\t- Training Recall: " 
					<< recall_macro(prediction, id.y, NUM_CLASSES).item<float>() << "  \n\t- Validation Recall: " 
					<< recall_macro(pred2, id.y_val, NUM_CLASSES).item<float>()  << "  \n\t- Training F score: " 
					<< f_score(prediction, id.y, NUM_CLASSES).item<float>() << "  \n\t- Validation F score: " 
					<< f_score(pred2, id.y_val, NUM_CLASSES).item<float>() << std::endl;
		}
		float valid = valid_loss.item<float>();
		if(prev_val_loss > valid) {
			torch::save(net, "iris.pt");
		}
	}
}