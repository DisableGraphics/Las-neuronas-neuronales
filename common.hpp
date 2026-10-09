#include <torch/torch.h>

torch::Tensor accuracy(torch::Tensor predictions, torch::Tensor targets);
torch::Tensor recall(torch::Tensor predictions, torch::Tensor targets, int64_t num_classes);
torch::Tensor recall_macro(torch::Tensor predictions, torch::Tensor targets, int64_t num_classes);
torch::Tensor f_score(torch::Tensor predictions, torch::Tensor targets, int64_t num_classes);